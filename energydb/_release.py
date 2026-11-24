"""Publication lifecycle: the read-side gate and the recovery path.

Two jobs, both in service of the invariant *a version is readable only once its
PostgreSQL run record and its ClickHouse values are both confirmed*
(``run_publication.state = 'confirmed'``):

* **Gate** (:func:`gate_td`): wrap a :class:`timedb.TimeDBClient` so every value
  read carries a ClickHouse ``additional_table_filters`` predicate excluding the
  run_ids whose publication is not yet confirmed. The filter is applied
  server-side on both ``series_values`` and ``run_series``, so a half-written
  batch's values can neither shadow an older confirmed value in a "latest" read
  nor appear as a run that touched a series. Applied only when the unconfirmed
  set is non-empty, so steady-state reads pay nothing.

* **Recovery** (:func:`resume_publication`, :func:`recover_incomplete`): locate a
  batch by ``run_id`` (or sweep every incomplete one by ``workflow_id``) and
  finish it. Finishing means: probe ClickHouse for the batch's values, and if
  they are present, repair the ``run_series`` mapping if the concurrent insert
  lane failed and flip the publication to ``confirmed``. A batch whose values
  never reached ClickHouse is left ``failed`` with guidance to re-run the write
  idempotently under the same ``run_id`` — the write path is safe to repeat.

The gate is the read-time half of the design; the write path
(:func:`energydb._io.write_manifest`) is the write-time half, flipping the
publication to ``confirmed`` immediately after ClickHouse acknowledges.
"""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Sequence
from typing import Any

from energydb import runs as runs_mod
from energydb.errors import PublicationNotFoundError, PublicationStateError

# The two CH relations a value read can touch, both keyed by run_id.
_GATED_RELATIONS = ("series_values", "run_series")


class _FilteredCH:
    """A ClickHouse client wrapper that merges preset query ``settings`` in.

    timedb's read functions take no ``settings`` argument, so the only seam for
    a server-side predicate is the client they call ``query``/``query_arrow``
    on. Preset settings win over per-call ones (the gate must not be
    overridable by a caller's settings), and everything else delegates
    unchanged, so ``insert_arrow``, ``command``, etc. behave exactly as before.
    """

    __slots__ = ("_inner", "_settings")

    def __init__(self, inner: Any, settings: dict[str, Any]) -> None:
        self._inner = inner
        self._settings = settings

    def _merge(self, settings: dict[str, Any] | None) -> dict[str, Any]:
        return {**(settings or {}), **self._settings}

    def query(self, *args: Any, settings: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        return self._inner.query(*args, settings=self._merge(settings), **kwargs)

    def query_arrow(self, *args: Any, settings: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        return self._inner.query_arrow(*args, settings=self._merge(settings), **kwargs)

    def command(self, *args: Any, settings: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        return self._inner.command(*args, settings=self._merge(settings), **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _gate_predicate(exclude_run_ids: Sequence[int]) -> str:
    """``run_id NOT IN (...)`` for the excluded ids.

    The ids are integers from a BIGINT column, so inlining them is injection-safe
    and keeps the predicate a plain constant (no bound params to thread through
    timedb's read functions, which do not accept them for settings).
    """
    return "run_id NOT IN (" + ", ".join(str(int(r)) for r in exclude_run_ids) + ")"


def gate_td(td: Any, exclude_run_ids: Sequence[int]) -> Any:
    """Return ``td`` with unconfirmed run_ids filtered out of every read, or
    ``td`` unchanged when there is nothing to exclude.

    Shallow-copies the client and swaps only ``_ch``, so the original (shared by
    every other reader and by writes) is untouched. An empty exclusion set is a
    no-op: steady-state reads keep the untouched, unwrapped client.
    """
    if not exclude_run_ids:
        return td
    predicate = _gate_predicate(exclude_run_ids)
    settings = {"additional_table_filters": {rel: predicate for rel in _GATED_RELATIONS}}
    clone = copy.copy(td)
    clone._ch = _FilteredCH(td._ch, settings)
    return clone


# ---------------------------------------------------------------------------
# ClickHouse evidence probe + repair (synchronous; run in a worker thread)
# ---------------------------------------------------------------------------


def _ch_evidence(td: Any, run_id: int) -> dict[str, int]:
    """Count the batch's rows in ClickHouse: ``{"values": n, "run_series": m}``.

    ``values > 0`` is the evidence that the CH half of the publication landed;
    ``run_series`` records whether the mapping lane also succeeded (the two
    insert lanes run concurrently and can fail independently).
    """
    result = td._ch.query(
        "SELECT "
        "(SELECT count() FROM series_values WHERE run_id = {rid:UInt64}) AS n_values, "
        "(SELECT count() FROM run_series FINAL WHERE run_id = {rid:UInt64}) AS n_runs",
        parameters={"rid": int(run_id)},
    )
    n_values, n_runs = result.result_rows[0]
    return {"values": int(n_values), "run_series": int(n_runs)}


def _backfill_run_series(td: Any, run_id: int) -> None:
    """Re-insert the ``run_series`` mapping for a batch from its stored values.

    Used when the values lane succeeded but the concurrent mapping lane failed:
    the values are complete, so the mapping is derivable from them. ``DISTINCT``
    keeps the insert one row per series, matching what the write path produces.
    """
    td._ch.command(
        "INSERT INTO run_series (series_id, run_id) "
        "SELECT DISTINCT series_id, run_id FROM series_values WHERE run_id = {rid:UInt64}",
        parameters={"rid": int(run_id)},
    )


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------


async def resume_publication(client, run_id: int) -> dict[str, Any]:
    """Finish a specific batch: confirm it if its values are present.

    The recovery counterpart to a crashed write. Idempotent: a ``confirmed``
    batch returns its row unchanged. For anything else it probes ClickHouse for
    the values and

    * values present → repair the ``run_series`` mapping if missing, flip the
      publication to ``confirmed``, return the updated row;
    * values absent → mark the publication ``failed`` and raise
      :class:`~energydb.errors.PublicationStateError`: the values never reached
      ClickHouse, so the batch cannot be confirmed from stored state. Re-running
      the original write with ``run_id=`` is idempotent and is the way to finish
      it.

    Raises :class:`~energydb.errors.PublicationNotFoundError` if no publication
    exists for ``run_id``.
    """
    async with client._conn() as conn:
        pub = await runs_mod.get_publication(conn, run_id)
    if pub is None:
        raise PublicationNotFoundError(
            f"no publication exists for run_id={run_id}; nothing to resume", run_id=run_id
        )
    if pub["state"] == runs_mod.CONFIRMED:
        return pub

    evidence = await asyncio.to_thread(_ch_evidence, client.td, run_id)
    if evidence["values"] == 0:
        message = (
            f"cannot resume run_id={run_id}: no values are stored in ClickHouse, so the batch "
            "cannot be confirmed from stored state. Re-run the original write with "
            f"run_id={run_id} (the write path is idempotent) to finish it, or leave it failed."
        )
        async with client._conn() as conn:
            await runs_mod.mark_publication_failed(conn, run_id, "no values in ClickHouse at resume time")
            await conn.commit()
        raise PublicationStateError(message, run_id=run_id, state=pub["state"])

    if evidence["run_series"] == 0:
        await asyncio.to_thread(_backfill_run_series, client.td, run_id)

    async with client._conn() as conn:
        await runs_mod.confirm_publication(conn, run_id)
        await conn.commit()
        updated = await runs_mod.get_publication(conn, run_id)
    assert updated is not None  # confirmed the row we just read
    return updated


async def recover_incomplete(
    client,
    *,
    workflow_id: str | None = None,
    limit: int = 100,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Sweep incomplete publications, resuming each; return a per-batch report.

    Each report entry is ``{run_id, state (before), action, detail}`` where
    ``action`` is ``"confirmed"``, ``"skipped"`` (values absent), or
    ``"would-confirm"``/``"would-skip"`` under ``dry_run``. One bad batch never
    aborts the sweep: its failure is recorded in that entry's ``detail`` and the
    sweep continues, so a single un-resumable batch cannot block the rest.
    """
    async with client._conn() as conn:
        incomplete = await runs_mod.list_incomplete_publications(conn, workflow_id=workflow_id, limit=limit)

    report: list[dict[str, Any]] = []
    for entry in incomplete:
        run_id = entry["run_id"]
        before = entry["state"]
        if dry_run:
            evidence = await asyncio.to_thread(_ch_evidence, client.td, run_id)
            action = "would-confirm" if evidence["values"] else "would-skip"
            report.append(
                {"run_id": run_id, "state": before, "action": action, "detail": f"{evidence['values']} value rows"}
            )
            continue
        try:
            updated = await resume_publication(client, run_id)
        except PublicationStateError as exc:
            report.append({"run_id": run_id, "state": before, "action": "skipped", "detail": str(exc)})
        else:
            report.append(
                {"run_id": run_id, "state": before, "action": "confirmed", "detail": f"state={updated['state']}"}
            )
    return report
