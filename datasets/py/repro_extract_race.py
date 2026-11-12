"""Repro (finding 5): concurrent eager extraction of the same archive into the shared HF cache.

Two processes load the same local `data.tar.gz` (CSV members) with load_dataset("csv", ...) into the same cache_dir but with
different builder configs (here: an explicit `sep=","` in one of them), e.g. two jobs of a sweep sharing the
cache, so they do NOT share the builder lock (builder.py:837-840). Both call ExtractManager.extract on the same
archive -> same output path:
(tar archives are always extracted eagerly, file_utils.py:223-247; packaged builders only stream single-file
compressions such as .csv.gz on the fly):
  - P1: _do_extract() (extract.py:42-45) sees nothing -> FileLock -> rmtree -> TarExtractor creates the output
        directory and extracts the members in place (extract.py:132-136, 451-461).
  - P2 (starts a bit later): _do_extract() runs WITHOUT the lock, sees a non-empty output directory -> returns it
        while P1 is still writing -> the CSV builder generates P2's dataset from missing/truncated members.
"""
import gzip, multiprocessing as mp, os, sys, tempfile, time


def load(tag, path, cache_dir, kwargs, q):
    import datasets
    datasets.logging.set_verbosity_error()
    datasets.disable_progress_bars()
    try:
        ds = datasets.load_dataset("csv", data_files=path, cache_dir=cache_dir, split="train", **kwargs)
        q.put((tag, f"{ds.num_rows} rows"))
    except Exception as e:
        q.put((tag, f"{type(e).__name__}: {str(e).splitlines()[0][:150]}"))


if __name__ == "__main__":
    n_rows = int(sys.argv[1]) if len(sys.argv) > 1 else 3_000_000
    delay = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
    tmp = tempfile.mkdtemp()
    import tarfile
    n_members = 4
    for m in range(n_members):
        with open(os.path.join(tmp, f"part-{m}.csv"), "w") as f:
            f.write("id,text\n")
            for i in range(m * n_rows // n_members, (m + 1) * n_rows // n_members):
                f.write(f"{i},row number {i} with some padding text to make the file larger\n")
    path = os.path.join(tmp, "data.tar.gz")
    with tarfile.open(path, "w:gz", compresslevel=1) as tar:
        for m in range(n_members):
            tar.add(os.path.join(tmp, f"part-{m}.csv"), arcname=f"part-{m}.csv")
            os.remove(os.path.join(tmp, f"part-{m}.csv"))
    print(f"expected rows: {n_rows}, archive size: {os.path.getsize(path) / 1e6:.1f} MB")
    cache_dir = os.path.join(tmp, "cache")
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p1 = ctx.Process(target=load, args=("P1 (default config)", path, cache_dir, {}, q))
    same = len(sys.argv) > 3 and sys.argv[3] == "same"   # control: same config -> same builder lock
    p2 = ctx.Process(target=load, args=(("P2 (same config)   " if same else "P2 (sep=',')       "), path, cache_dir,
                                        ({} if same else {"sep": ","}), q))
    p1.start()
    time.sleep(delay)
    p2.start()
    p1.join(); p2.join()
    while not q.empty():
        tag, res = q.get()
        print(f"{tag}: {res}")
