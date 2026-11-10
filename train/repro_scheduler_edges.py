"""
Scheduler edge cases found by Schedulers.lean (all low priority).
"""

import warnings

import torch

from transformers import get_scheduler


def lrs(name, W, T, n, **kw):
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    s = get_scheduler(name, opt, num_warmup_steps=W, num_training_steps=T, scheduler_specific_kwargs=kw)
    out = [opt.param_groups[0]["lr"]]
    for _ in range(n):
        opt.step()
        s.step()
        out.append(opt.param_groups[0]["lr"])
    return [round(x, 4) for x in out]


# (a) polynomial with num_warmup_steps == num_training_steps: last scheduler.step() divides by zero
try:
    print("polynomial W=T=5:", lrs("polynomial", 5, 5, 5, lr_end=0.0))
except ZeroDivisionError as e:
    print("polynomial W=T=5: ZeroDivisionError at step 5:", e)

# (b) cosine_warmup_with_min_lr, warmup_lr_rate=0.1: with W=2 the last warmup step uses the peak (1.0);
#     with W=1 the single warmup step uses 0.1 and the peak is never used.
print("cosine_warmup_with_min_lr W=2:", lrs("cosine_warmup_with_min_lr", 2, 20, 4, min_lr_rate=0.0, warmup_lr_rate=0.1))
print("cosine_warmup_with_min_lr W=1:", lrs("cosine_warmup_with_min_lr", 1, 20, 4, min_lr_rate=0.0, warmup_lr_rate=0.1))

# (c) WSD with num_warmup_steps + num_decay_steps > num_training_steps -> negative stable phase,
#     LR jumps from 0.9 to 0.5 when warmup ends; no error/warning.
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    print("wsd W=10 D=10 T=15:", lrs("warmup_stable_decay", 10, 15, 15, num_decay_steps=10, decay_type="linear"))
    print("  warnings:", [str(x.message) for x in w])
