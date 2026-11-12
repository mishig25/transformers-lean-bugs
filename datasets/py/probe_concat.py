import copy, traceback
import torch.utils.data as tud
from datasets import Dataset, concatenate_datasets
import datasets
datasets.logging.set_verbosity_error()
class WI:
    def __init__(self, id, n): self.id, self.num_workers, self.seed, self.dataset = id, n, 0, None
def mk(base, n_shards, per):
    return Dataset.from_dict({"a": [base + s * 100 + j for s in range(n_shards) for j in range(per)]}).to_iterable_dataset(num_shards=n_shards)
ds0 = concatenate_datasets([mk(0, 2, 3), mk(10000, 1, 4)])
print("num_shards", ds0.num_shards)
for wid in range(2):
    tud.get_worker_info = lambda: WI(wid, 2)
    ds = copy.deepcopy(ds0)
    it = iter(ds)
    print("worker", wid, "first:", next(it))
    sd = ds.state_dict(); print("state:", sd)
    ds2 = copy.deepcopy(ds0); ds2.load_state_dict(sd)
    try:
        print("resumed:", [e["a"] for e in ds2])
    except Exception:
        traceback.print_exc(limit=-3)
    for k in [0]:
        ds = copy.deepcopy(ds0); sd = ds.state_dict(); print("k=0 state:", sd)
        ds2 = copy.deepcopy(ds0); ds2.load_state_dict(sd)
        try: print("resumed k=0:", [e["a"] for e in ds2])
        except Exception: traceback.print_exc(limit=-3)
