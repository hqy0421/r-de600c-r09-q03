"""Run metadata and publication state on the ``runs`` / ``run_publication`` tables.

A *publication* is one batch (one ``run_id``) across both backends: its ``runs``
row in PostgreSQL and the values stamped with that id in ClickHouse. They live
in separate systems and cannot commit atomically, so the two failure windows —

* values written, run record not (the CH<->PG gap), and
* run record written, values not (a CH failure after the PG commit) —

would otherwise leave a batch that is neither readable nor locatable. The
``run_publication`` row is the explicit, re-openable state machine that makes a
half-finished batch identifiable and lets a scheduler finish it:

``pending`` → (values in CH) → ``confirmed``, with ``failed`` (write raised) and
``reopened`` (a confirmed batch withdrawn for repair) as the recovery and
withdrawal states. Reads serve a version only once the state is ``confirmed``,
which by construction means both halves are present (see
:mod:`energydb._release`).

``run_id`` itself is client-generated (uuid7 truncated to UInt64), so the write
path mints an id, upserts a ``pending`` row for it, writes the values, then
flips it to ``confirmed`` — no round-trip spent on id allocation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, NamedTuple

from psycopg.types.json import Jsonb
from uuid6 import uuid7

from energydb.errors import PublicationStateError, ValidationError
from energydb.models import SQL_SCHEMA_PREFIX as P

# The state vocabulary, mirrored by the CHECK constraint on the table. Kept in
# one tuple so the SQL and the Python agree and the error messages can list it.
PUBLICATION_STATES: tuple[str, ...] = ("pending", "confirmed", "failed", "reopened")
CONFIRMED = "confirmed"
PENDING = "pending"
FAILED = "failed"
REOPENED = "reopened"


class RunRow(NamedTuple):
    """The columns of a ``runs`` upsert, carried as one value.

    Lets the write path fold the run upsert into the manifest-resolve query
    (one round-trip) via :func:`run_upsert_cte`, or run it standalone via
    :func:`upsert_run_row`."""

    run_id: int
    workflow_id: str | None = None
    model_name: str | None = None
    run_start_time: datetime | None = None
    run_finish_time: datetime | None = None
    run_params: dict | None = None


# Shared by the standalone upsert and the foldable CTE. Idempotent under retry;
# run identity is immutable and keyed by run_id.
_RUN_UPSERT_BODY = f"""INSERT INTO {P}runs
            (run_id, workflow_id, model_name, run_start_time, run_finish_time, run_params)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (run_id) DO UPDATE SET
            workflow_id     = EXCLUDED.workflow_id,
            model_name      = EXCLUDED.model_name,
            run_start_time  = EXCLUDED.run_start_time,
            run_finish_time = EXCLUDED.run_finish_time,
            run_params      = EXCLUDED.run_params"""

# The publication row rides *after* the run row in the same statement via a
# data-modifying CTE, so registering a batch is still a single round-trip that
# cannot record a publication without its run row.
#
# The conflict clause is deliberately asymmetric: a re-run of the *same*
# write (idempotent retry) must not knock a confirmed batch back to pending,
# but any other state — pending, failed, reopened — resets to a clean pending
# with the previous attempt's evidence cleared, so a retried batch starts from
# a known baseline. Self-references use the target's own alias (``pub``),
# which PostgreSQL accepts in the ``DO UPDATE`` expressions.
_PUBLICATION_RESET_BODY = f"""INSERT INTO {P}run_publication AS pub (run_id, state)
        SELECT run_id, 'pending' FROM run_ins
        ON CONFLICT (run_id) DO UPDATE SET
            state = CASE WHEN pub.state = 'confirmed' THEN 'confirmed' ELSE 'pending' END,
            values_written_at = CASE WHEN pub.state = 'confirmed' THEN pub.values_written_at ELSE NULL END,
            confirmed_at = CASE WHEN pub.state = 'confirmed' THEN pub.confirmed_at ELSE NULL END,
            last_error = CASE WHEN pub.state = 'confirmed' THEN pub.last_error ELSE NULL END,
            updated_at = CASE WHEN pub.state = 'confirmed' THEN pub.updated_at ELSE now() END"""


def _run_params(run: RunRow) -> tuple:
    if run.run_start_time is not None and run.run_start_time.tzinfo is None:
        raise ValidationError("run_start_time must be timezone-aware")
    if run.run_finish_time is not None and run.run_finish_time.tzinfo is None:
        raise ValidationError("run_finish_time must be timezone-aware")
    return (
        run.run_id,
        run.workflow_id,
        run.model_name,
        run.run_start_time,
        run.run_finish_time,
        Jsonb(run.run_params or {}),
    )


def run_upsert_cte(run: RunRow) -> tuple[str, tuple]:
    """A data-modifying CTE that upserts ``run`` (and its ``pending``
    publication row), to prepend to another query.

    Postgres executes a ``WITH … AS (INSERT …)`` exactly once, even when the
    main query never references it, so the run and its publication are
    recorded regardless of what the folded SELECT returns (preserves "an
    all-skipped write still records a run"). The publication CTE reads the run
    upsert's own output (``SELECT … FROM run_ins``), so the two rows are
    written in one statement and the FK holds. Returns ``(cte_sql, params)``;
    the params bind the run columns once (``run_ins``'s RETURNING feeds the
    publication insert, whose row is not parameterized) and bind before the
    rest of the statement."""
    return (
        f"WITH run_ins AS (\n        {_RUN_UPSERT_BODY}\n        RETURNING run_id\n    ), "
        f"pub_ins AS (\n        {_PUBLICATION_RESET_BODY}\n    )\n",
        _run_params(run),
    )


async def upsert_run_row(conn, run: RunRow) -> None:
    """Standalone run upsert (run-metadata updates outside a write's folded
    resolve). Registers the publication ``pending`` alongside it, exactly as the
    folded CTE does."""
    await conn.execute(
        f"WITH run_ins AS (\n        {_RUN_UPSERT_BODY}\n        RETURNING run_id\n    ), "
        f"pub_ins AS (\n        {_PUBLICATION_RESET_BODY}\n    )\n"
        "SELECT 1",
        _run_params(run),
    )


def generate_run_id() -> int:
    """Client-side BIGINT run id: top 63 bits of a uuid7.

    uuid7's 128-bit layout keeps the 48-bit ms timestamp in the high bits.
    Shifting right by 65 preserves timestamp sortability and keeps the result
    within a signed 63-bit range (fits ``BIGINT`` / polars ``Int64``).
    """
    return uuid7().int >> 65


async def upsert_run(
    conn,
    *,
    run_id: int,
    workflow_id: str | None = None,
    model_name: str | None = None,
    run_start_time: datetime | None = None,
    run_finish_time: datetime | None = None,
    run_params: dict | None = None,
) -> None:
    """Insert or update a run row (and its publication state). Idempotent under retry."""
    await upsert_run_row(
        conn,
        RunRow(run_id, workflow_id, model_name, run_start_time, run_finish_time, run_params),
    )


async def get_runs(conn, run_ids: list[int]) -> list[dict[str, Any]]:
    """Hydrate run metadata by id. Returns matched rows ordered by
    ``inserted_at`` descending (latest run first)."""
    if not run_ids:
        return []
    rows = await (
        await conn.execute(
            f"""
            SELECT run_id, workflow_id, model_name, run_start_time, run_finish_time,
                   run_params, inserted_at
            FROM {P}runs
            WHERE run_id = ANY(%s)
            ORDER BY inserted_at DESC
            """,
            (run_ids,),
        )
    ).fetchall()
    return [
        {
            "run_id": r[0],
            "workflow_id": r[1],
            "model_name": r[2],
            "run_start_time": r[3],
            "run_finish_time": r[4],
            "run_params": r[5],
            "inserted_at": r[6],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Publication state transitions
# ---------------------------------------------------------------------------


async def mark_values_written(conn, run_id: int) -> None:
    """Record that ClickHouse acknowledged the batch's values (still pending
    confirmation, which is the separate PG-recording step)."""
    await conn.execute(
        f"UPDATE {P}run_publication SET values_written_at = now(), updated_at = now() "
        "WHERE run_id = %s AND state <> 'confirmed'",
        (run_id,),
    )


async def confirm_publication(conn, run_id: int) -> None:
    """Flip a batch to ``confirmed``: from here reads will serve its version.

    Idempotent (confirming an already-confirmed batch is a no-op). The
    ``values_written_at`` stamp is advisory bookkeeping for the CH half; the
    authoritative check that both halves are present is the recovery probe
    (:func:`energydb._release.confirm_publication`), not this row alone.
    """
    await conn.execute(
        f"UPDATE {P}run_publication SET state = 'confirmed', confirmed_at = now(), "
        "updated_at = now(), last_error = NULL "
        "WHERE run_id = %s AND state <> 'confirmed'",
        (run_id,),
    )


async def mark_publication_failed(conn, run_id: int, error: str) -> None:
    """Record that the value write raised, leaving the batch for recovery.

    Best-effort: a failure to record a failure must not mask the original
    error, so callers suppress exceptions from here (the batch state stuck at
    ``pending`` is still findable by run_id).
    """
    await conn.execute(
        f"UPDATE {P}run_publication SET state = 'failed', last_error = %s, updated_at = now() "
        "WHERE run_id = %s AND state <> 'confirmed'",
        (error[:2000], run_id),
    )


async def reopen_publication(conn, run_id: int) -> None:
    """Withdraw a confirmed batch so its version stops being served, pending repair.

    Raises :class:`~energydb.errors.PublicationStateError` if the publication is
    already ``reopened``/``failed`` (nothing to withdraw) or absent.
    """
    row = await (
        await conn.execute(
            f"UPDATE {P}run_publication SET state = 'reopened', updated_at = now() "
            "WHERE run_id = %s AND state = 'confirmed' RETURNING run_id",
            (run_id,),
        )
    ).fetchone()
    if row is None:
        await _raise_bad_transition(conn, run_id, "reopen", want=CONFIRMED)


async def _raise_bad_transition(conn, run_id: int, action: str, *, want: str | None = None) -> None:
    """Raise the right :class:`PublicationStateError` for a no-op transition.

    Distinguishes "no such publication" from "exists but in the wrong state", so
    a scheduler can tell a typo from a double-reopen.
    """
    state = await get_publication_state(conn, run_id)
    if state is None:
        raise PublicationStateError(
            f"no publication exists for run_id={run_id}", run_id=run_id, state=None,
        )
    expected = f" (expected {want!r})" if want is not None else ""
    raise PublicationStateError(
        f"cannot {action} publication for run_id={run_id} in state {state!r}{expected}",
        run_id=run_id,
        state=state,
    )


async def get_publication_state(conn, run_id: int) -> str | None:
    """The publication ``state`` for ``run_id``, or ``None`` if there is no row."""
    row = await (
        await conn.execute(
            f"SELECT state FROM {P}run_publication WHERE run_id = %s",
            (run_id,),
        )
    ).fetchone()
    return None if row is None else row[0]


async def get_publication(conn, run_id: int) -> dict[str, Any] | None:
    """The full publication row for ``run_id`` plus its run metadata, or ``None``.

    One LEFT JOIN so a caller (recovery) gets everything it needs to describe a
    batch in a single round-trip."""
    row = await (
        await conn.execute(
            f"""
            SELECT p.run_id, p.state, p.values_written_at, p.confirmed_at,
                   p.created_at, p.updated_at, p.last_error,
                   r.workflow_id, r.model_name, r.run_start_time, r.run_finish_time,
                   r.run_params, r.inserted_at
            FROM {P}run_publication p
            LEFT JOIN {P}runs r ON r.run_id = p.run_id
            WHERE p.run_id = %s
            """,
            (run_id,),
        )
    ).fetchone()
    if row is None:
        return None
    return {
        "run_id": row[0],
        "state": row[1],
        "values_written_at": row[2],
        "confirmed_at": row[3],
        "created_at": row[4],
        "updated_at": row[5],
        "last_error": row[6],
        "workflow_id": row[7],
        "model_name": row[8],
        "run_start_time": row[9],
        "run_finish_time": row[10],
        "run_params": row[11],
        "inserted_at": row[12],
    }


async def list_incomplete_publications(
    conn,
    *,
    workflow_id: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Batches whose publication is not ``confirmed``, oldest first.

    The recovery entry point's index: every row here is a batch a scheduler
    started but did not finish (a ``pending`` orphan, a ``failed`` write, or a
    ``reopened`` withdrawal). ``workflow_id`` narrows to one producer. Ordered
    by ``created_at`` so a recovery sweep drains oldest-first and does not
    starve behind a retry storm.
    """
    params: list[Any] = []
    where = "p.state <> 'confirmed'"
    if workflow_id is not None:
        where += " AND r.workflow_id = %s"
        params.append(workflow_id)
    params.append(limit)
    rows = await (
        await conn.execute(
            f"""
            SELECT p.run_id, p.state, p.values_written_at, p.confirmed_at,
                   p.created_at, p.updated_at, p.last_error,
                   r.workflow_id, r.model_name
            FROM {P}run_publication p
            LEFT JOIN {P}runs r ON r.run_id = p.run_id
            WHERE {where}
            ORDER BY p.created_at ASC
            LIMIT %s
            """,
            params,
        )
    ).fetchall()
    return [
        {
            "run_id": r[0],
            "state": r[1],
            "values_written_at": r[2],
            "confirmed_at": r[3],
            "created_at": r[4],
            "updated_at": r[5],
            "last_error": r[6],
            "workflow_id": r[7],
            "model_name": r[8],
        }
        for r in rows
    ]


async def unconfirmed_run_ids(conn) -> list[int]:
    """Every ``run_id`` whose publication is not ``confirmed``.

    Feeds the read-side gate: these ids are excluded from value reads so a
    half-finished batch is never served. Small by construction — the partial
    index keeps the query over only the exception set.
    """
    rows = await (
        await conn.execute(f"SELECT run_id FROM {P}run_publication WHERE state <> 'confirmed'")
    ).fetchall()
    return [r[0] for r in rows]
