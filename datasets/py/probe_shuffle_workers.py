import copy, traceback
import torch.utils.data as tud
from datasets import Dataset
import datasets
datasets.logging.set_verbosity_error()
class WI:
    def __init__(self, id, n): self.id, self.num_workers, self.seed, self.dataset = id, n, 0, None
ds0 = Dataset.from_dict({"a": list(range(15))}).to_iterable_dataset(num_shards=3).shuffle(seed=3, buffer_size=4)
tud.get_worker_info = lambda: WI(0, 2)
ds = copy.deepcopy(ds0); it = iter(ds); print(next(it)); sd = ds.state_dict(); print(sd)
ds2 = copy.deepcopy(ds0); ds2.load_state_dict(sd)
try: print([e["a"] for e in ds2])
except Exception: traceback.print_exc()
