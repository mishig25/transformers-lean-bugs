"""Applies the proposed fixes as runtime monkeypatches (src/ is untouched), then runs a reproducer script.
Usage: python run_with_fixes.py <repro_script.py>

Fix 1 (wedge): a cancelled *active* request frees memory like a finished one, so it also clears
         scheduler.block_new_requests (and the flag is never left set with no active request).
Fix 2 (async KeyError / output after cancel): a cancelled state is marked FINISHED, exactly like the old object of a
         soft-reset request, so update_batch skips it if it is still in flight (async mode).
Fix 3 (hard stop): requests drained in the same iteration as the HARD_STOP are registered as waiting so that
         _fail_all_remaining_requests fails them instead of dropping them.
"""

import runpy
import sys

from transformers.generation.continuous_batching import continuous_api
from transformers.generation.continuous_batching.requests import RequestStatus
from transformers.generation.continuous_batching.scheduler import Scheduler


def clear_cancelled_requests(self):
    cancelled_states = []
    with self._cancellation_lock:
        for request_id in self._requests_to_cancel:
            state_a = self.active_requests.pop(request_id, None)
            state_w = self.waiting_requests.pop(request_id, None)
            state = state_a or state_w
            if state is not None:
                cancelled_states.append(state)
                state._status = RequestStatus.FINISHED  # fix 2: in-flight (async) updates skip it
            if state_a is not None:
                self.block_new_requests = False  # fix 1: its blocks are freed below, like a finished request
            if request_id in self.waiting_requests_order:
                self.waiting_requests_order.remove(request_id)
            self.cache.free_blocks(request_id)
        self._requests_to_cancel = set()
    if not self.active_requests:
        self.block_new_requests = False  # fix 1 (belt and braces): nothing active can ever reset it otherwise
    return cancelled_states


Scheduler.clear_cancelled_requests = clear_cancelled_requests

orig_update = continuous_api.ContinuousBatchProcessor._update_tp_group_state
BTS = continuous_api.BackgroundThreadStatus


def _update_tp_group_state(self):
    if self.input_queue is not None and self.background_thread_status.local_status == BTS.HARD_STOP:
        # fix 3 (simplified for a single rank): keep what the drain would have discarded so it gets failed
        from transformers.generation.continuous_batching.utils import drain_queue

        for state in drain_queue(self.input_queue):
            self.scheduler.add_waiting_request(state)
    return orig_update(self)


continuous_api.ContinuousBatchProcessor._update_tp_group_state = _update_tp_group_state

sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
