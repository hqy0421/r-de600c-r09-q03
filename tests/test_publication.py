"""Publication state machine: re-openable batches, read gating, recovery.

A batch (one ``run_id``) spans two backends that cannot commit atomically:
ClickHouse holds the values, PostgreSQL the ``runs`` row and the
``run_publication`` state. The contract these tests pin:

* a write leaves its publication ``confirmed`` and immediately readable;
* a batch whose values are in ClickHouse but whose confirmation was never
  recorded stays hidden from reads until recovery confirms it — the "data
  written, run record not written" window;
* a confirmed batch can be *reopened*, which withdraws its version from reads
  without deleting it, and ``resume_publication`` re-confirms it;
* recovery can *locate* a batch by run_id and finish it, and sweep incomplete
  batches by workflow without aborting on one that cannot be finished.

No-DB tests cover the gate predicate and the state-transition helpers; the live
tests exercise the two-backend flow against real PostgreSQL and ClickHouse.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import energydb as edb
import polars as pl
import pytest
from energydb import Client
from energydb import runs as runs_mod
from energydb._release import _gate_predicate, gate_td
from energydb.errors import (
    PublicationConfirmationError,
    PublicationNotFoundError,
    PublicationStateError,
)
from energydb.models import SQL_SCHEMA_PREFIX as P

pytestmark_live = pytest.mark.skipif(
    not (os.environ.get("TIMEDB_PG_DSN") and os.environ.get("TIMEDB_CH_URL")),
    reason="TIMEDB_PG_DSN / TIMEDB_CH_URL not set",
)

BASE_VT = datetime(2026, 1, 1, tzinfo=UTC)
KT_1 = datetime(2026, 1, 1, 6, tzinfo=UTC)
KT_2 = datetime(2026, 1, 1, 7, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Gate predicate + client wrapping (no DB needed)
# ---------------------------------------------------------------------------


def test_gate_predicate_lists_the_excluded_ids():
    assert _gate_predicate([5, 7]) == "run_id NOT IN (5, 7)"
    assert _gate_predicate([42]) == "run_id NOT IN (42)"


def test_gate_td_is_a_noop_with_nothing_to_exclude():
    """Steady state: an empty exclusion set must return the very client object,
    so reads pay neither a wrapper nor a settings merge."""
    sentinel = object()
    assert gate_td(sentinel, []) is sentinel


def test_gate_td_swaps_only_the_ch_client():
    """The gate must shallow-copy so the shared client (used by writes and other
    readers) is untouched, swapping only ``_ch``."""

    class _FakeTD:
        def __init__(self):
            self._ch = object()

    td = _FakeTD()
    original_ch = td._ch
    gated = gate_td(td, [9])
    assert gated is not td
    assert td._ch is original_ch  # original untouched
    # The wrapper exposes the filter as a preset setting merged into queries.
    assert gated._ch._settings["additional_table_filters"]["series_values"] == "run_id NOT IN (9)"
    assert gated._ch._settings["additional_table_filters"]["run_series"] == "run_id NOT IN (9)"


def test_gate_wrapper_presets_win_over_per_call_settings():
    """A caller must not be able to override the gate with their own settings."""

    class _Inner:
        def __init__(self):
            self.seen = None

        def query(self, sql, *, settings=None, **kwargs):
            self.seen = settings
            return "ok"

    inner = _Inner()
    from energydb._release import _FilteredCH

    wrapped = _FilteredCH(inner, {"additional_table_filters": {"series_values": "run_id NOT IN (1)"}})
    wrapped.query("SELECT 1", settings={"max_threads": 1})
    assert wrapped._inner.seen["max_threads"] == 1
    assert wrapped._inner.seen["additional_table_filters"]["series_values"] == "run_id NOT IN (1)"


# ---------------------------------------------------------------------------
# Publication helpers against a raw PG connection (no CH needed)
# ---------------------------------------------------------------------------


@pytestmark_live
class TestPublicationStateHelpers:
    @staticmethod
    def _fresh() -> Client:
        c = Client()
        c.delete()
        c.create()
        return c

    def test_a_folded_run_upsert_creates_a_pending_publication(self):
        """The folded CTE writes run + publication in one statement; before any
        values land the state is ``pending``, which reads exclude."""
        client = self._fresh()
        try:
            async def go():
                async with client._async._pool.connection() as conn:
                    await runs_mod.upsert_run(conn, run_id=77)
                    await conn.commit()
                    return await runs_mod.get_publication_state(conn, 77)

            assert client._portal.run(go()) == runs_mod.PENDING
        finally:
            client.delete()
            client.close()

    def test_confirm_and_reopen_round_trip(self):
        client = self._fresh()
        try:
            async def go():
                async with client._async._pool.connection() as conn:
                    await runs_mod.upsert_run(conn, run_id=88)
                    await runs_mod.confirm_publication(conn, 88)
                    confirmed = await runs_mod.get_publication_state(conn, 88)
                    await runs_mod.reopen_publication(conn, 88)
                    reopened = await runs_mod.get_publication_state(conn, 88)
                    await conn.commit()
                    return confirmed, reopened

            assert client._portal.run(go()) == (runs_mod.CONFIRMED, runs_mod.REOPENED)
        finally:
            client.delete()
            client.close()

    def test_reopening_a_pending_batch_raises_state_error(self):
        client = self._fresh()
        try:
            async def go():
                async with client._async._pool.connection() as conn:
                    await runs_mod.upsert_run(conn, run_id=99)
                    await conn.commit()
                    with pytest.raises(PublicationStateError) as excinfo:
                        await runs_mod.reopen_publication(conn, 99)
                    return excinfo.value

            err = client._portal.run(go())
            assert err.run_id == 99
            assert err.state == runs_mod.PENDING
        finally:
            client.delete()
            client.close()

    def test_reopening_a_missing_batch_raises_not_found_state_error(self):
        client = self._fresh()
        try:
            async def go():
                async with client._async._pool.connection() as conn:
                    with pytest.raises(PublicationStateError):
                        await runs_mod.reopen_publication(conn, 123456)
                    return await runs_mod.get_publication_state(conn, 123456)

            assert client._portal.run(go()) is None
        finally:
            client.delete()
            client.close()

    def test_replaying_a_confirmed_write_does_not_reset_it(self):
        """Idempotent retry: the folded upsert's conflict clause preserves a
        confirmed batch (same write re-run must stay readable)."""
        client = self._fresh()
        try:
            async def go():
                async with client._async._pool.connection() as conn:
                    await runs_mod.upsert_run(conn, run_id=55)
                    await runs_mod.confirm_publication(conn, 55)
                    await conn.commit()
                    # Same run row upserted again (a replay).
                    await runs_mod.upsert_run(conn, run_id=55)
                    await conn.commit()
                    return await runs_mod.get_publication_state(conn, 55)

            assert client._portal.run(go()) == runs_mod.CONFIRMED
        finally:
            client.delete()
            client.close()

    def test_deleting_the_run_row_cascades_the_publication(self):
        """The write path's orphan-run compensation is a DELETE on runs; the FK
        cascade must clear the publication with it."""
        client = self._fresh()
        try:
            async def go():
                async with client._async._pool.connection() as conn:
                    await runs_mod.upsert_run(conn, run_id=66)
                    await conn.commit()
                    await conn.execute(f"DELETE FROM {P}runs WHERE run_id = %s", (66,))
                    await conn.commit()
                    return await runs_mod.get_publication_state(conn, 66)

            assert client._portal.run(go()) is None
        finally:
            client.delete()
            client.close()


# ---------------------------------------------------------------------------
# End-to-end against real PG + CH
# ---------------------------------------------------------------------------


@pytest.fixture
def live_client():
    client = Client()
    client.delete()
    client.create()
    root = edb.Portfolio(name="P", members=[edb.wind.WindTurbine(name="T1", capacity=3.0)])
    client.register_tree(root)
    client.get_node("P", "T1").register_series(
        name="power", canonical_unit="MW", data_type="actual", timeseries_type="FLAT", retention="medium"
    )
    yield client
    client.delete()
    client.close()


def _manifest(values: list[float], *, values_start: int = 0) -> pl.DataFrame:
    base = BASE_VT + timedelta(hours=values_start)
    return pl.DataFrame(
        {
            "path": ["P/T1"] * len(values),
            "data_type": ["actual"] * len(values),
            "name": ["power"] * len(values),
            "valid_time": pl.datetime_range(
                base, base + timedelta(hours=len(values) - 1), interval="1h", time_unit="us", time_zone="UTC", eager=True
            ),
            "value": values,
        }
    )


def _read_latest(client: Client) -> list[float]:
    m = pl.DataFrame({"path": ["P/T1"], "data_type": ["actual"], "name": ["power"]})
    return client.read(m)["value"].to_list()


@pytestmark_live
def test_a_successful_write_is_confirmed_and_readable(live_client):
    res = live_client.write(_manifest([1.0, 2.0]), knowledge_time=KT_1)
    assert (res.written, res.skipped) == (2, 0)
    pub = live_client.resume_publication(int(res))  # idempotent: stays confirmed
    assert pub["state"] == runs_mod.CONFIRMED
    assert _read_latest(live_client) == [1.0, 2.0]


@pytestmark_live
def test_unconfirmed_values_are_hidden_from_reads(live_client):
    """**The read gate.** Write two vintages; force the newer one back to
    ``pending`` (as the "run record not written" window would leave it) and
    confirm the gate hides it, then ``resume_publication`` restores it."""
    first = int(live_client.write(_manifest([1.0, 2.0]), knowledge_time=KT_1))
    second = int(live_client.write(_manifest([10.0, 20.0]), knowledge_time=KT_2))
    # The latest read serves the newer vintage.
    assert _read_latest(live_client) == [10.0, 20.0]

    # Simulate the CH-values-landed-but-not-confirmed window for `second`.
    async def demote():
        async with live_client._async._pool.connection() as conn:
            await conn.execute(
                f"UPDATE {P}run_publication SET state = 'pending' WHERE run_id = %s", (second,)
            )
            await conn.commit()

    live_client._portal.run(demote())

    # Hidden: reads fall back to the older confirmed vintage, not the newer one.
    assert _read_latest(live_client) == [1.0, 2.0]

    # And the run is not reported as having touched the series.
    sid = _series_id(live_client)
    reported = {r["run_id"] for r in live_client.read_runs_for_series(series_id=sid)}
    assert second not in reported and first in reported

    # Recovery locates it by run_id and finishes it: values are present.
    pub = live_client.resume_publication(second)
    assert pub["state"] == runs_mod.CONFIRMED
    assert _read_latest(live_client) == [10.0, 20.0]


@pytestmark_live
def test_reopening_withdraws_the_version_and_resume_restores_it(live_client):
    """Re-openable state: a confirmed batch can be withdrawn (hidden again)
    without losing its values, then re-confirmed by recovery."""
    run_id = int(live_client.write(_manifest([3.0, 4.0]), knowledge_time=KT_1))
    assert _read_latest(live_client) == [3.0, 4.0]

    pub = live_client.reopen_publication(run_id)
    assert pub["state"] == runs_mod.REOPENED
    assert _read_latest(live_client) == []  # withdrawn and hidden

    live_client.resume_publication(run_id)
    assert _read_latest(live_client) == [3.0, 4.0]


@pytestmark_live
def test_reopening_a_pending_batch_by_client_raises_state_error(live_client):
    """The client-level reopen goes through the state machine, so a batch that
    is not yet servable is rejected rather than silently rewritten."""

    async def register_only():
        async with live_client._async._pool.connection() as conn:
            await runs_mod.upsert_run(conn, run_id=606060)
            await conn.commit()

    live_client._portal.run(register_only())
    with pytest.raises(PublicationStateError) as excinfo:
        live_client.reopen_publication(606060)
    assert excinfo.value.run_id == 606060
    assert excinfo.value.state == runs_mod.PENDING


@pytestmark_live
def test_recovery_locates_a_batch_that_has_no_values(live_client):
    """A batch whose values never reached ClickHouse cannot be confirmed by
    recovery; it is marked failed and the error names the run_id to re-run."""
    async def register_only():
        async with live_client._async._pool.connection() as conn:
            await runs_mod.upsert_run(conn, run_id=424242, workflow_id="wf_x")
            await conn.commit()

    live_client._portal.run(register_only())

    with pytest.raises(PublicationStateError) as excinfo:
        live_client.resume_publication(424242)
    assert excinfo.value.run_id == 424242

    async def state():
        async with live_client._async._pool.connection() as conn:
            return await runs_mod.get_publication_state(conn, 424242)

    assert live_client._portal.run(state()) == runs_mod.FAILED


@pytestmark_live
def test_resume_repairs_a_missing_run_series_mapping(live_client):
    """The values lane and the run_series lane insert concurrently and can fail
    independently. If the values landed but the mapping did not, recovery
    derives the mapping from the stored values and still confirms."""
    run_id = int(live_client.write(_manifest([13.0, 14.0]), knowledge_time=KT_1))

    # Simulate the run_series lane never having committed.
    live_client.td._ch.command(
        "DELETE FROM run_series WHERE run_id = {rid:UInt64}", parameters={"rid": run_id}
    )
    assert live_client.td.read_run_series(series_id=_series_id(live_client)) == []

    async def demote():
        async with live_client._async._pool.connection() as conn:
            await conn.execute(f"UPDATE {P}run_publication SET state = 'pending' WHERE run_id = %s", (run_id,))
            await conn.commit()

    live_client._portal.run(demote())

    pub = live_client.resume_publication(run_id)
    assert pub["state"] == runs_mod.CONFIRMED
    # The mapping was repaired from the values, so the run is reported again.
    assert run_id in live_client.td.read_run_series(series_id=_series_id(live_client))


@pytestmark_live
def test_resume_unknown_run_id_raises_not_found(live_client):
    with pytest.raises(PublicationNotFoundError) as excinfo:
        live_client.resume_publication(999999999)
    assert excinfo.value.run_id == 999999999


@pytestmark_live
def test_recover_incomplete_sweeps_by_workflow(live_client):
    """The bulk recovery path: locate every incomplete batch for a workflow and
    finish the ones whose values exist, skipping (not aborting on) the rest."""
    good = int(live_client.write(_manifest([5.0, 6.0]), knowledge_time=KT_1, workflow_id="wf_sweep"))

    async def seed():
        async with live_client._async._pool.connection() as conn:
            # A batch with values but unconfirmed: demote the real write.
            await conn.execute(f"UPDATE {P}run_publication SET state = 'pending' WHERE run_id = %s", (good,))
            # A batch with no values at all (never written to CH).
            await runs_mod.upsert_run(conn, run_id=515151, workflow_id="wf_sweep")
            await conn.commit()

    live_client._portal.run(seed())

    report = live_client.recover_incomplete(workflow_id="wf_sweep")
    by_id = {r["run_id"]: r for r in report}
    assert by_id[good]["action"] == "confirmed"
    assert by_id[515151]["action"] == "skipped"
    assert _read_latest(live_client) == [5.0, 6.0]


@pytestmark_live
def test_recover_incomplete_dry_run_does_not_mutate(live_client):
    run_id = int(live_client.write(_manifest([7.0]), knowledge_time=KT_1))

    async def demote():
        async with live_client._async._pool.connection() as conn:
            await conn.execute(f"UPDATE {P}run_publication SET state = 'pending' WHERE run_id = %s", (run_id,))
            await conn.commit()

    live_client._portal.run(demote())
    report = live_client.recover_incomplete(dry_run=True)
    entry = next(r for r in report if r["run_id"] == run_id)
    assert entry["action"] == "would-confirm"

    async def state():
        async with live_client._async._pool.connection() as conn:
            return await runs_mod.get_publication_state(conn, run_id)

    assert live_client._portal.run(state()) == runs_mod.PENDING  # untouched


@pytestmark_live
def test_confirmation_failure_raises_and_leaves_the_batch_hidden(live_client, monkeypatch):
    """The "values written, confirmation not recorded" window: if PG is
    unreachable right after the CH write, the write raises
    ``PublicationConfirmationError`` carrying the run_id, and the batch stays
    pending (hidden) so recovery can finish it later."""
    from energydb import runs as _runs

    async def boom(*args, **kwargs):
        raise RuntimeError("simulated PG outage during confirmation")

    # Fail the PG write inside the confirmation, not the wrapper, so the
    # wrapper's own error path (the PublicationConfirmationError) is exercised.
    monkeypatch.setattr(_runs, "confirm_publication", boom)

    with pytest.raises(PublicationConfirmationError) as excinfo:
        live_client.write(_manifest([8.0, 9.0]), knowledge_time=KT_1)
    run_id = excinfo.value.run_id
    assert excinfo.value.state == runs_mod.PENDING

    # Values did reach ClickHouse.
    evidence = live_client.td._ch.query(
        "SELECT count() FROM series_values WHERE run_id = {rid:UInt64}", parameters={"rid": run_id}
    ).result_rows[0][0]
    assert int(evidence) == 2

    # But reads exclude it: the batch is pending.
    assert _read_latest(live_client) == []

    # Recovery finishes it once PG is reachable again.
    monkeypatch.undo()
    pub = live_client.resume_publication(run_id)
    assert pub["state"] == runs_mod.CONFIRMED
    assert _read_latest(live_client) == [8.0, 9.0]


@pytestmark_live
def test_value_write_failure_leaves_the_batch_failed_and_findable(live_client, monkeypatch):
    """The inverse window: the run record commits, then the CH write raises. The
    publication is marked ``failed`` (not lost), the original error propagates,
    and the batch is still locatable by run_id for recovery."""
    from energydb import _io as _io_mod

    monkeypatch.setattr(_io_mod.asyncio, "to_thread", _make_to_thread_writer_boom())

    with pytest.raises(RuntimeError, match="simulated ClickHouse outage"):
        live_client.write(_manifest([11.0, 12.0]), knowledge_time=KT_1, workflow_id="wf_fail")

    # The batch is findable and marked failed, not silently gone.
    report = live_client.recover_incomplete(workflow_id="wf_fail")
    assert len(report) == 1
    assert report[0]["state"] == runs_mod.FAILED
    assert report[0]["action"] == "skipped"  # no values reached CH


def _make_to_thread_writer_boom():
    """A ``to_thread`` replacement that fails only the ``td.write`` call.

    Everything else (the CH evidence probes, etc.) still runs for real, so the
    test exercises the write-failure path without breaking recovery.
    """
    from energydb import _io as _io_mod

    real = _io_mod.asyncio.to_thread

    def fake(fn, *args, **kwargs):
        if getattr(fn, "__name__", "") == "write":
            async def boom():
                raise RuntimeError("simulated ClickHouse outage")

            return boom()
        return real(fn, *args, **kwargs)

    return fake


def _uuid_of(client: Client, path: str):
    import psycopg

    with psycopg.connect(os.environ["TIMEDB_PG_DSN"]) as conn:
        return conn.execute(f"SELECT uuid FROM {P}node WHERE path = %s", (path,)).fetchone()[0]


def _series_id(client: Client) -> int:
    return client.list_series(_uuid_of(client, "P/T1"))[0]["series_id"]
