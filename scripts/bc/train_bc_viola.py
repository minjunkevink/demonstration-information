import argparse
import json
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import wandb

from deminf_pytorch.datasets import GenericTrajectoryDataset


class PolicyMLP(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: List[int] = [512, 512], dropout: float = 0.5):
        super().__init__()
        layers = []
        dims = [state_dim] + hidden
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(dims[-1], action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_keep_manifest(path: str) -> List[int]:
    with open(path, "r") as f:
        obj = json.load(f)
    return list(map(int, obj["keep_episode_ids"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_npz", type=str, required=True)
    parser.add_argument("--val_npz", type=str, required=True)
    parser.add_argument("--keep_manifest", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--project", type=str, default="deminf-viola")
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    wandb.init(project=args.project, name=args.run_name, config=vars(args))

    train_ds = GenericTrajectoryDataset(args.train_npz, use_language=False)
    val_ds = GenericTrajectoryDataset(args.val_npz, use_language=False)

    # If keep manifest provided, restrict training set to selected episodes
    if args.keep_manifest is not None and os.path.exists(args.keep_manifest):
        keep_ep_ids = set(load_keep_manifest(args.keep_manifest))
        train_indices = [i for i, e in enumerate(train_ds.ep_idx) if int(e) in keep_ep_ids]
        train_ds = Subset(train_ds, train_indices)
        wandb.log({"num_train_steps": len(train_indices)})
    else:
        wandb.log({"num_train_steps": len(train_ds)})

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state_dim = train_ds.dataset.states.shape[1] if isinstance(train_ds, Subset) else train_ds.states.shape[1]
    action_dim = train_ds.dataset.actions.shape[1] if isinstance(train_ds, Subset) else train_ds.actions.shape[1]

    policy = PolicyMLP(state_dim, action_dim).to(device)
    optim_ = optim.Adam(policy.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    def _run_epoch(loader, train: bool):
        total, count = 0.0, 0
        policy.train(mode=train)
        for batch in loader:
            states = batch["observation"]["state"].to(device)
            actions = batch["action"].to(device)
            preds = policy(states)
            loss = loss_fn(preds, actions)
            if train:
                optim_.zero_grad()
                loss.backward()
                optim_.step()
            total += loss.item() * states.shape[0]
            count += states.shape[0]
        return total / max(count, 1)

    for epoch in range(1, args.epochs + 1):
        train_loss = _run_epoch(train_loader, True)
        val_loss = _run_epoch(val_loader, False)
        wandb.log({"epoch": epoch, "train/loss": train_loss, "val/loss": val_loss})
        print(f"Epoch {epoch}: train {train_loss:.4f} | val {val_loss:.4f}")

    wandb.finish()


if __name__ == "__main__":
    main()


