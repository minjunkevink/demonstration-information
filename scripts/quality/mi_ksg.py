import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch

from deminf_pytorch import DemInfQualityEstimator
from deminf_pytorch.datasets import GenericTrajectoryDataset


def aggregate_by_episode(step_scores: np.ndarray, ep_idx: np.ndarray):
    acc = defaultdict(list)
    for s, e in zip(step_scores.tolist(), ep_idx.tolist()):
        acc[int(e)].append(float(s))
    ep_ids, ep_means, ep_stds, ep_counts = [], [], [], []
    for e in sorted(acc.keys()):
        vals = np.array(acc[e], dtype=np.float32)
        ep_ids.append(e)
        ep_means.append(float(vals.mean()))
        ep_stds.append(float(vals.std()))
        ep_counts.append(int(vals.shape[0]))
    return np.array(ep_ids), np.array(ep_means), np.array(ep_stds), np.array(ep_counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", type=str, required=True)
    parser.add_argument("--models_dir", type=str, required=True, help="Directory with trained state_vae.pth and action_vae.pth")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--keep_fraction", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    dataset = GenericTrajectoryDataset(args.npz_path, use_language=False)

    # Build estimator and load models
    state_dim = dataset.states.shape[1]
    action_dim = dataset.actions.shape[1]
    estimator = DemInfQualityEstimator(
        state_dim=state_dim,
        action_dim=action_dim,
        state_z_dim=12,
        action_z_dim=6,
        hidden_dims=[512, 512],
        beta=0.05,
        k_values=[5, 6, 7],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    estimator.load_models(args.models_dir)

    # Encode in batches to limit memory
    batch_size = 8192
    scores = []
    for start in range(0, len(dataset), batch_size):
        end = min(len(dataset), start + batch_size)
        states = torch.from_numpy(dataset.states[start:end]).to(estimator.device)
        actions = torch.from_numpy(dataset.actions[start:end]).to(estimator.device)
        with torch.no_grad():
            z_s = estimator.state_vae.predict(states)
            z_a = estimator.action_vae.predict(actions)
            mi = estimator.ksg_estimator.estimate_mutual_information(z_s, z_a)
        scores.append(mi.cpu().numpy())
    step_scores = np.concatenate(scores, axis=0)

    ep_ids, ep_means, ep_stds, ep_counts = aggregate_by_episode(step_scores, dataset.ep_idx)

    # Save CSV-like npz
    np.savez(
        os.path.join(args.out_dir, "mi_per_episode.npz"),
        episode_id=ep_ids,
        mi_mean=ep_means,
        mi_std=ep_stds,
        num_steps=ep_counts,
    )

    # Produce a manifest of top-k episodes
    k = int(len(ep_means) * args.keep_fraction)
    keep_indices = np.argsort(ep_means)[-k:]
    kept_ep_ids = ep_ids[keep_indices].tolist()
    with open(os.path.join(args.out_dir, f"keep_{args.keep_fraction:.2f}.json"), "w") as f:
        json.dump({"keep_episode_ids": kept_ep_ids}, f)
    print(f"Saved per-episode MI and keep manifest to {args.out_dir}")


if __name__ == "__main__":
    main()


