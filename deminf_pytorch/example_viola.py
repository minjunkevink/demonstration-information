"""
Example script: DemInf on OXE VIOLA NPZ exports
Run after converting TFRecords to NPZ via scripts/data/oxe_viola_to_deminf.py
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from deminf_pytorch import DemInfQualityEstimator
from deminf_pytorch.datasets import GenericTrajectoryDataset
import wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", type=str, required=True, help="Path to NPZ file (e.g., train.npz)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--save_dir", type=str, default="./deminf_oxe_viola_models")
    parser.add_argument("--project", type=str, default="deminf-viola")
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    wandb.init(project=args.project, name=args.run_name, config=vars(args))

    dataset = GenericTrajectoryDataset(args.npz_path, use_language=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Infer dims
    state_dim = dataset.states.shape[1]
    action_dim = dataset.actions.shape[1]

    deminf = DemInfQualityEstimator(
        state_dim=state_dim,
        action_dim=action_dim,
        state_z_dim=12,
        action_z_dim=6,
        hidden_dims=[512, 512],
        beta=0.05,
        k_values=[5, 6, 7],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print("Training State VAE...")
    for epoch in range(args.epochs):
        deminf.train_vae(loader, epochs=1, vae_type='state')
        wandb.log({"epoch": epoch + 1, "vae_state/epoch": epoch + 1})
    print("Training Action VAE...")
    for epoch in range(args.epochs):
        deminf.train_vae(loader, epochs=1, vae_type='action')
        wandb.log({"epoch": epoch + 1, "vae_action/epoch": epoch + 1})

    print("Estimating MI-based quality scores...")
    scores_dict = deminf.estimate_quality_scores(loader)
    scores = scores_dict['quality_scores']
    print(f"Scores: mean={np.mean(scores):.4f}, std={np.std(scores):.4f}")

    os.makedirs(args.save_dir, exist_ok=True)
    deminf.save_models(args.save_dir)
    np.save(os.path.join(args.save_dir, "quality_scores.npy"), scores)
    print(f"Saved models and scores to {args.save_dir}")

    wandb.log({"scores/mean": float(np.mean(scores)), "scores/std": float(np.std(scores))})
    wandb.finish()


if __name__ == "__main__":
    main()


