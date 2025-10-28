import argparse
import os
from typing import List, Optional

import numpy as np
import tensorflow as tf

# Reuse the repo's TFDS loader and transforms
from openx.data.core import load_dataset
from openx.data.transforms import concatenate
from openx.data.datasets.oxe import viola_dataset_transform


def _iterate_episodes(builder_dir: str, split: str):
    """
    Load VIOLA episodes via the repo's TFDS plumbing and yield standardized episodes.
    We do not apply a fixed structure so we can flatten manually using concatenate().
    """
    ds, _ = load_dataset(
        path=builder_dir,
        split=split,
        standardization_transform=viola_dataset_transform,
        structure=None,
        dataset_statistics=None,
        shuffle=False,
        num_parallel_reads=1,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    for ep in ds:
        # Flatten state and action dicts into single arrays
        ep = concatenate(ep)
        yield ep


def _to_npz_arrays(episodes) -> dict:
    """
    Concatenate variable-length episodes into flat arrays with episode_id and timestep.
    Produces keys compatible with DemInf PyTorch pipeline.
    """
    states: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    episode_ids: List[np.ndarray] = []
    timesteps: List[np.ndarray] = []
    languages: List[Optional[List[str]]] = []
    lengths: List[int] = []

    ep_id = 0
    for ep in episodes:
        # Each ep contains tensors; convert to numpy
        s = ep["observation"]["state"].numpy()
        a = ep["action"].numpy()

        T = s.shape[0]
        states.append(s)
        actions.append(a)
        episode_ids.append(np.full((T,), ep_id, dtype=np.int32))
        timesteps.append(np.arange(T, dtype=np.int32))
        lengths.append(T)

        # Optional language support if present (per-step or per-episode)
        lang_series: Optional[List[str]] = None
        if "language_instruction" in ep:
            # language may be a scalar string tensor per-episode; broadcast over time
            lang_tensor = ep["language_instruction"]
            if len(lang_tensor.shape) == 0:
                lang_str = lang_tensor.numpy().decode("utf-8")
                lang_series = [lang_str] * T
            else:
                # per-step strings
                lang_series = [x.decode("utf-8") for x in lang_tensor.numpy().tolist()]
        languages.append(lang_series)

        ep_id += 1

    out = {
        "observation/state": np.concatenate(states, axis=0).astype(np.float32),
        "action": np.concatenate(actions, axis=0).astype(np.float32),
        "episode_id": np.concatenate(episode_ids, axis=0),
        "timestep": np.concatenate(timesteps, axis=0),
    }
    # Only include language if any episode had it
    if any(ls is not None for ls in languages):
        # Fill missing with empty strings
        filled: List[str] = []
        for i, ls in enumerate(languages):
            if ls is None:
                filled.extend([""] * lengths[i])
            else:
                filled.extend(ls)
        out["language"] = np.array(filled, dtype=object)
    return out


def convert_split(builder_dir: str, split: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    episodes = _iterate_episodes(builder_dir, split)
    arrays = _to_npz_arrays(episodes)
    np.savez_compressed(out_path, **arrays)
    print(f"Wrote {out_path} with keys: {list(arrays.keys())}")


def main():
    parser = argparse.ArgumentParser(description="Convert OXE VIOLA TFRecords to DemInf NPZ format")
    parser.add_argument("--builder_dir", type=str, required=True, help="Path to VIOLA TFDS builder dir (0.1.0)")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to write NPZ files")
    parser.add_argument(
        "--splits",
        type=str,
        nargs="*",
        default=["train", "test"],
        help="Splits to convert (default: train test)",
    )

    args = parser.parse_args()

    for split in args.splits:
        out_path = os.path.join(args.output_dir, f"{split}.npz")
        convert_split(args.builder_dir, split, out_path)


if __name__ == "__main__":
    main()


