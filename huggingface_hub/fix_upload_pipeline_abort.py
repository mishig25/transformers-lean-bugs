"""Proposed fix for the global xet abort in _UploadPipeline.run, applied in-process for validation only.

`except BaseException: ... abort_xet_session()` (_upload_pipeline.py:454-457) becomes: abort this pipeline's own
in-flight xet upload-commit, and abort the process-wide session only on KeyboardInterrupt (as every other caller
of abort_xet_session does: file_download.py:601, _commit_api.py:679, hf_api.py:14964/15208).
"""

import inspect
import textwrap


def apply():
    from huggingface_hub import _upload_pipeline as up

    src = textwrap.dedent(inspect.getsource(up._UploadPipeline))
    old_run = """        try:
            self._coordinator_loop()
        except BaseException:
            self.abort_event.set()
            abort_xet_session()
            raise"""
    new_run = """        try:
            self._coordinator_loop()
        except BaseException as e:
            self.abort_event.set()
            if isinstance(e, KeyboardInterrupt):
                abort_xet_session()
            raise"""
    old_loop = """        batch = _Batch()
        for start in range(0, len(self.add_operations), PREUPLOAD_BATCH_SIZE):"""
    new_loop = """        batch = _Batch()
        try:
            self._coordinator_loop_inner(batch)
        except BaseException:
            self._abort_batch(self._current_batch)  # only this pipeline's in-flight xet upload-commit
            raise

    def _coordinator_loop_inner(self, batch) -> None:
        import hf_xet

        self._current_batch = batch
        for start in range(0, len(self.add_operations), PREUPLOAD_BATCH_SIZE):"""
    old_new_batch = """                    self._enqueue(batch)
                    batch = _Batch()"""
    new_new_batch = """                    self._enqueue(batch)
                    batch = _Batch()
                    self._current_batch = batch"""
    for old, new in ((old_run, new_run), (old_loop, new_loop), (old_new_batch, new_new_batch)):
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    ns = dict(vars(up))
    exec(compile(src, up.__file__ + " (fixed)", "exec"), ns)
    up._UploadPipeline = ns["_UploadPipeline"]
