import h5py
from typing import Generator, Sequence, Tuple

def batch_count(data, batch_size):
    """Return number of batch_size batches into which data can be divided."""
    from math import ceil
    return ceil(len(data) / batch_size)

def batch(data: Sequence, batch_size: int) -> Generator[Tuple[int, int, Sequence], None, None]:
    """Iterate over sequence data in batches.
    
    Arguments:
        data: must support len() and __getitem__
        batch_size: how many items to return for each iteration
    Yields:
        tuple of
        * batch number (int)
        * row offset (int)
        * batch_size number of items from data
    """        
    total_row_count = len(data)
    for start in range(0, total_row_count, batch_size):
        i = start//batch_size
        end = min(start + batch_size, total_row_count)
        yield i, start, data[start:end]

def save_visual_outputs(output_path: str, data: dict):
    t0 = time()

    # output_path is full path to labels.json, so replace "json" with "h5"
    viz_output_path = output_path
    if viz_output_path.endswith(".json"):
        viz_output_path = viz_output_path[:-(len(".json"))]
    viz_output_path += ".h5"

    # write file
    with h5py.File(viz_output_path, "a") as f:
        for key, val in data.items():
            val = np.array(val)
            if key in f:
                f[key].resize(f[key].shape[0] + val.shape[0], axis=0)
                f[key][-val.shape[0]:] = val
            else:
                maxshape = (None, *val.shape[1:])
                f.create_dataset(key, data=val, maxshape=maxshape,
                    compression="gzip", compression_opts=9)

    logger.info("  Saved visual outputs [%.1fs]" % (time() - t0))