"""``_td_call`` forwards only the read kwargs the installed timedb accepts.

energydb and timedb are versioned separately, so a deployment can pin a timedb
whose read surface predates a kwarg this energydb offers. Forwarding blindly
would turn *every* read into a ``TypeError``; the factory instead drops an
unsupported argument whose value requests nothing and refuses one the caller
actually asked for.

The fake readers therefore declare *explicit* signatures (no ``**kwargs``):
introspection is exactly what the factory acts on.
"""

from __future__ import annotations

import pytest
from energydb._io import _td_call
from energydb.errors import ValidationError


class _NoBucket:
    """A pre-bucketing timedb: ``read`` has no ``bucket_us``/``bucket_dedup``."""

    def __init__(self):
        self.calls: list[dict] = []

    def read(self, *, series_ids, retention, meta_source=None, start_valid=None):
        self.calls.append({"series_ids": series_ids, "retention": retention, "start_valid": start_valid})
        return "frame"


class _Full:
    """A timedb with the whole modern read surface."""

    def __init__(self):
        self.calls: list[dict] = []

    def read(self, *, series_ids, retention, meta_source=None, bucket_us=None, bucket_dedup=False, start_valid=None):
        self.calls.append(
            {
                "series_ids": series_ids,
                "retention": retention,
                "bucket_us": bucket_us,
                "bucket_dedup": bucket_dedup,
                "start_valid": start_valid,
            }
        )
        return "frame"


def test_unsupported_none_valued_kwarg_is_dropped():
    """``bucket_us=None`` is the default; an installed timedb without it must not
    fail every read, while a supported kwarg still gets through."""
    td = _NoBucket()
    call = _td_call(td, relative=False, kwargs={"bucket_us": None, "start_valid": None})
    call([1], ["medium"])
    assert td.calls[0] == {"series_ids": [1], "retention": ["medium"], "start_valid": None}


def test_unsupported_requested_kwarg_raises():
    """A dropped ``bucket_us=5000000`` would change the shape of the answer, so
    a caller who asked for it gets a clear error rather than silent bucketing
    loss."""
    td = _NoBucket()
    with pytest.raises(ValidationError, match="bucket_us"):
        _td_call(td, relative=False, kwargs={"bucket_us": 5_000_000})
    assert td.calls == []


def test_supported_kwargs_pass_through_untouched():
    td = _Full()
    call = _td_call(td, relative=False, kwargs={"bucket_us": 5_000_000, "bucket_dedup": False, "start_valid": None})
    call([1], ["medium"])
    assert td.calls[0]["bucket_us"] == 5_000_000
    assert td.calls[0]["bucket_dedup"] is False


def test_a_reader_taking_var_kwargs_accepts_everything():
    """A future timedb with ``**kwargs`` is treated as accepting all of them."""

    class _VarKwReader:
        def __init__(self):
            self.seen = None

        def read(self, *, series_ids, retention, meta_source=None, **kwargs):
            self.seen = kwargs
            return "frame"

    td = _VarKwReader()
    call = _td_call(td, relative=False, kwargs={"bucket_us": 5_000_000})
    assert call([1], ["medium"]) == "frame"
    assert td.seen["bucket_us"] == 5_000_000


def test_bucket_dedup_is_dropped_even_when_true():
    """``bucket_dedup`` only modifies a ``bucket_us`` request, so dropping it
    when unsupported is always harmless — unlike a True-valued ``bucket_us``."""
    td = _NoBucket()
    call = _td_call(td, relative=False, kwargs={"bucket_dedup": True})
    call([1], ["medium"])
    assert td.calls[0] == {"series_ids": [1], "retention": ["medium"], "start_valid": None}


def test_relative_filters_against_read_relative_not_read():
    """The factory picks the path's reader, so filtering follows that method's
    signature: ``start_valid`` is fine on ``read`` but absent on
    ``read_relative`` here, so it is dropped on the relative path."""

    class _Asym(_NoBucket):
        def read_relative(self, *, series_ids, retention, meta_source=None):
            self.calls.append({"series_ids": series_ids, "retention": retention})
            return "rel"

    td = _Asym()
    call = _td_call(td, relative=True, kwargs={"start_valid": None})
    assert call([1], ["medium"]) == "rel"
    assert td.calls[0] == {"series_ids": [1], "retention": ["medium"]}
