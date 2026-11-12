"""Monkeypatch with the proposed minimal fixes (the checkout is not modified). Import before building datasets.

F1  _iter_pytorch must not load the worker's state into the un-sharded tree (iterable_dataset.py:2909 -> 3051-3054).
F2  _starting_state_dict is consumed by the iteration that uses it (never re-applied to later epochs), and
    load_state_dict() makes state_dict() return the loaded state until the iteration starts.
F3  RebatchedArrowExamplesIterable above a shuffle buffer checkpoints the child's live state (after the shuffled
    table was produced) with 0 chunks to skip: the not-yet-yielded rest of that table counts as lost buffer.
"""
import datasets.iterable_dataset as idm
from datasets.iterable_dataset import IterableDataset, RebatchedArrowExamplesIterable, BufferShuffledExamplesIterable, _fast_copy

_orig_prepare = IterableDataset._prepare_ex_iterable_for_iteration
_orig_iter_pytorch = IterableDataset._iter_pytorch


def _prepare(self, batch_size=1, drop_last_batch=False, _load_starting_state=True):
    starting = self._starting_state_dict
    if not _load_starting_state:
        self._starting_state_dict = None
    try:
        ex_iterable = _orig_prepare(self, batch_size=batch_size, drop_last_batch=drop_last_batch)
    finally:
        if not _load_starting_state:
            self._starting_state_dict = starting
    if _load_starting_state and starting is not None and self.epoch == starting["epoch"]:
        self._starting_state_dict = None  # consumed (F2)
    return ex_iterable


def _iter_pytorch(self):
    # same as the original, except that the first prepare does not load/consume the starting state (F1);
    # the original body then loads it into the worker tree (lines 2944-2947), after which we consume it (F2)
    orig_prepare = self._prepare_ex_iterable_for_iteration
    self._prepare_ex_iterable_for_iteration = lambda *a, **k: _prepare(self, *a, _load_starting_state=False, **k)
    try:
        gen = _orig_iter_pytorch(self)
        first = True
        for x in gen:
            if first and self._starting_state_dict is not None and self.epoch == self._starting_state_dict["epoch"]:
                self._starting_state_dict = None
            first = False
            yield x
        if self._starting_state_dict is not None and self.epoch == self._starting_state_dict["epoch"]:
            self._starting_state_dict = None
    finally:
        del self._prepare_ex_iterable_for_iteration


def _load_state_dict(self, state_dict):
    self._starting_state_dict = state_dict
    self._state_dict = _fast_copy(state_dict)  # F2b


def _has_buffer_shuffle(ex):
    if isinstance(ex, BufferShuffledExamplesIterable):
        return True
    kids = []
    if hasattr(ex, "ex_iterable"): kids.append(ex.ex_iterable)
    if hasattr(ex, "ex_iterables"): kids.extend(ex.ex_iterables)
    if getattr(ex, "_interleaved", None) is not None: kids.append(ex._interleaved)
    return any(_has_buffer_shuffle(k) for k in kids)


_orig_rebatch_iter_arrow = RebatchedArrowExamplesIterable._iter_arrow


def _rebatch_iter_arrow(self):
    non_replayable = _has_buffer_shuffle(self.ex_iterable)
    for key, table in _orig_rebatch_iter_arrow(self):
        if non_replayable and self._state_dict:
            self._state_dict["previous_state"] = self.ex_iterable.state_dict()
            self._state_dict["num_chunks_since_previous_state"] = 0
            self._state_dict["cropped_chunk_length"] = 0
        yield key, table


IterableDataset._prepare_ex_iterable_for_iteration = _prepare
IterableDataset._iter_pytorch = _iter_pytorch
IterableDataset.load_state_dict = _load_state_dict
RebatchedArrowExamplesIterable._iter_arrow = _rebatch_iter_arrow


# F3 (cross-rank part): with strategy "examples", StepExamplesIterable._iter_arrow consumes whole shuffled tables, so
# ranks that yielded the same number of examples are inside different shuffled tables and resume from different
# shuffle-buffer positions. The element-wise path (StepExamplesIterable.__iter__) keeps every rank at the same
# position (k * world_size elements consumed). Use it whenever a shuffle buffer is below the step.
from datasets.iterable_dataset import StepExamplesIterable

StepExamplesIterable.iter_arrow = property(
    lambda self: (self._iter_arrow if self.ex_iterable.iter_arrow and not _has_buffer_shuffle(self.ex_iterable) else None)
)
