from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class GenericTrajectoryDataset(Dataset):
    """
    Simple NPZ-backed dataset producing dicts compatible with DemInfQualityEstimator:
    {
        'observation': {'state': float32[N, D_state]},
        'action': float32[N, D_action],
        'ep_idx': int,
        'dataset_id': int (always 0),
        'quality_score': float (0.0 placeholder),
        'language': Optional[str]
    }
    """

    def __init__(self, npz_path: str, use_language: bool = False):
        arrays = np.load(npz_path, allow_pickle=True)
        self.states = arrays["observation/state"].astype(np.float32)
        self.actions = arrays["action"].astype(np.float32)
        self.ep_idx = arrays.get("episode_id")
        self.language = arrays.get("language") if use_language and "language" in arrays else None
        assert self.states.shape[0] == self.actions.shape[0]
        self.length = self.states.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx) -> Dict:
        out: Dict = {
            "observation": {"state": torch.from_numpy(self.states[idx])},
            "action": torch.from_numpy(self.actions[idx]),
            "ep_idx": int(self.ep_idx[idx]) if self.ep_idx is not None else 0,
            "dataset_id": 0,
            "quality_score": 0.0,
        }
        if self.language is not None:
            out["language"] = self.language[idx]
        return out


