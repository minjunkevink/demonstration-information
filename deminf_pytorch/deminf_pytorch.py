"""
PyTorch Implementation of DemInf (Demonstration Information Estimation)
Based on "Robot Data Curation with Mutual Information Estimators" paper

This is a direct PyTorch reimplementation of the DemInf method for estimating
demonstration quality using mutual information between states and actions.

Key components:
1. VAE for learning low-dimensional state and action embeddings
2. KSG estimator for mutual information using k-nearest neighbors
3. Quality estimation pipeline for demonstration data curation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from scipy.special import digamma
from typing import Dict, List, Tuple, Optional
import json
import os
from tqdm import tqdm


class MLP(nn.Module):
    """Simple MLP for VAE encoder/decoder"""
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, 
                 activation=nn.ReLU, activate_final: bool = False):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2 or activate_final:
                layers.append(activation())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class VAEEncoder(nn.Module):
    """VAE Encoder for state or action embeddings"""
    def __init__(self, input_dim: int, hidden_dims: List[int], z_dim: int):
        super().__init__()
        self.z_dim = z_dim
        
        # Encoder network
        self.encoder = MLP(input_dim, hidden_dims, hidden_dims[-1], activate_final=True)
        
        # Mean and log variance heads
        self.mean_head = nn.Linear(hidden_dims[-1], z_dim)
        self.logvar_head = nn.Linear(hidden_dims[-1], z_dim)
    
    def forward(self, x):
        h = self.encoder(x)
        mean = self.mean_head(h)
        logvar = self.logvar_head(h)
        return mean, logvar
    
    def encode(self, x):
        """Get latent representation"""
        mean, logvar = self.forward(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z, mean, logvar


class VAEDecoder(nn.Module):
    """VAE Decoder for reconstruction"""
    def __init__(self, z_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        self.decoder = MLP(z_dim, hidden_dims, output_dim)
    
    def forward(self, z):
        return self.decoder(z)


class BetaVAE(nn.Module):
    """Beta-VAE for learning state and action embeddings"""
    def __init__(self, input_dim: int, z_dim: int, hidden_dims: List[int] = [512, 512], 
                 beta: float = 0.05):
        super().__init__()
        self.z_dim = z_dim
        self.beta = beta
        
        self.encoder = VAEEncoder(input_dim, hidden_dims, z_dim)
        self.decoder = VAEDecoder(z_dim, hidden_dims, input_dim)
    
    def forward(self, x):
        z, mean, logvar = self.encoder.encode(x)
        recon = self.decoder(z)
        return recon, mean, logvar, z
    
    def encode(self, x):
        """Get latent representation without sampling"""
        mean, logvar = self.encoder(x)
        return mean
    
    def predict(self, x):
        """Predict latent representation (used in quality estimation)"""
        return self.encode(x)


class KSGEstimator:
    """KSG (Kraskov-Stögbauer-Grassberger) Mutual Information Estimator"""
    
    def __init__(self, k_values: List[int] = [5, 6, 7]):
        self.k_values = k_values
    
    def _l2_distances(self, z: torch.Tensor) -> torch.Tensor:
        """Compute pairwise L2 distances between all points"""
        # z: (batch_size, z_dim)
        # Returns: (batch_size, batch_size)
        return torch.cdist(z, z, p=2)
    
    def _knn_distances(self, z: torch.Tensor, k: int) -> torch.Tensor:
        """Get k-nearest neighbor distances for each point"""
        dists = self._l2_distances(z)
        # Sort distances and get k-th nearest neighbor (excluding self)
        knn_dists, _ = torch.topk(dists, k + 1, dim=1, largest=False)
        return knn_dists[:, 1:]  # Exclude self (distance 0)
    
    def estimate_mutual_information(self, z_obs: torch.Tensor, z_action: torch.Tensor) -> torch.Tensor:
        """
        Estimate mutual information I(S;A) using KSG estimator
        
        Args:
            z_obs: State embeddings (batch_size, z_dim)
            z_action: Action embeddings (batch_size, z_dim)
        
        Returns:
            MI estimates for each sample (batch_size,)
        """
        batch_size = z_obs.shape[0]
        mi_estimates = []
        
        for k in self.k_values:
            # Compute distances
            obs_dists = self._l2_distances(z_obs)
            action_dists = self._l2_distances(z_action)
            
            # Use InfNorm for joint distance (max of obs and action distances)
            joint_dists = torch.maximum(obs_dists, action_dists)
            
            # Get k-th nearest neighbor distances in joint space
            joint_knn_dists, _ = torch.topk(joint_dists, k + 1, dim=1, largest=False)
            joint_knn_dists = joint_knn_dists[:, 1:]  # Exclude self
            
            # Count neighbors within k-th distance for each marginal
            obs_counts = torch.sum(obs_dists[:, :, None] < joint_knn_dists[:, None, :], dim=1)
            action_counts = torch.sum(action_dists[:, :, None] < joint_knn_dists[:, None, :], dim=1)
            
            # KSG estimator: MI = -mean(digamma(n_x) + digamma(n_y))
            # Convert to numpy for digamma function
            obs_counts_np = obs_counts.cpu().numpy()
            action_counts_np = action_counts.cpu().numpy()
            
            mi_k = -np.mean(digamma(obs_counts_np) + digamma(action_counts_np), axis=1)
            mi_estimates.append(torch.from_numpy(mi_k).float())
        
        # Average across different k values
        mi_estimates = torch.stack(mi_estimates, dim=1)
        return torch.mean(mi_estimates, dim=1)


class DemInfQualityEstimator:
    """Main DemInf quality estimation pipeline"""
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 state_z_dim: int = 12,
                 action_z_dim: int = 6,
                 hidden_dims: List[int] = [512, 512],
                 beta: float = 0.05,
                 k_values: List[int] = [5, 6, 7],
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.device = device
        self.k_values = k_values
        
        # Initialize VAEs
        self.state_vae = BetaVAE(state_dim, state_z_dim, hidden_dims, beta).to(device)
        self.action_vae = BetaVAE(action_dim, action_z_dim, hidden_dims, beta).to(device)
        
        # Initialize KSG estimator
        self.ksg_estimator = KSGEstimator(k_values)
        
        # Optimizers
        self.state_optimizer = optim.Adam(self.state_vae.parameters(), lr=1e-4)
        self.action_optimizer = optim.Adam(self.action_vae.parameters(), lr=1e-4)
    
    def train_vae(self, data_loader: DataLoader, epochs: int = 50, vae_type: str = 'state'):
        """Train VAE on demonstration data"""
        vae = self.state_vae if vae_type == 'state' else self.action_vae
        optimizer = self.state_optimizer if vae_type == 'state' else self.action_optimizer
        
        vae.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in tqdm(data_loader, desc=f"Training {vae_type} VAE - Epoch {epoch+1}"):
                if vae_type == 'state':
                    x = batch['observation']['state'].to(self.device)
                else:
                    x = batch['action'].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                recon, mean, logvar, z = vae(x)
                
                # Loss computation
                recon_loss = F.mse_loss(recon, x, reduction='mean')
                kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1).mean()
                total_loss_batch = recon_loss + vae.beta * kl_loss
                
                total_loss_batch.backward()
                optimizer.step()
                
                total_loss += total_loss_batch.item()
            
            avg_loss = total_loss / len(data_loader)
            print(f"{vae_type} VAE Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    def estimate_quality_scores(self, data_loader: DataLoader) -> Dict[str, np.ndarray]:
        """
        Estimate quality scores for demonstrations using DemInf
        
        Args:
            data_loader: DataLoader containing demonstration data
            
        Returns:
            Dictionary with quality scores and metadata
        """
        self.state_vae.eval()
        self.action_vae.eval()
        
        all_scores = []
        all_metadata = {
            'ep_idx': [],
            'dataset_id': [],
            'quality_score': []
        }
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Estimating quality scores"):
                # Get state and action data
                states = batch['observation']['state'].to(self.device)
                actions = batch['action'].to(self.device)
                
                # Get embeddings
                z_states = self.state_vae.predict(states)
                z_actions = self.action_vae.predict(actions)
                
                # Estimate mutual information
                mi_scores = self.ksg_estimator.estimate_mutual_information(z_states, z_actions)
                
                all_scores.append(mi_scores.cpu().numpy())
                
                # Store metadata
                if 'ep_idx' in batch:
                    all_metadata['ep_idx'].extend(batch['ep_idx'].numpy())
                if 'dataset_id' in batch:
                    all_metadata['dataset_id'].extend(batch['dataset_id'].numpy())
                if 'quality_score' in batch:
                    all_metadata['quality_score'].extend(batch['quality_score'].numpy())
        
        # Concatenate all scores
        all_scores = np.concatenate(all_scores, axis=0)
        
        # Normalize scores (clip outliers and standardize)
        all_scores = np.clip(all_scores, 
                           np.percentile(all_scores, 1), 
                           np.percentile(all_scores, 99))
        all_scores = (all_scores - np.mean(all_scores)) / np.std(all_scores)
        
        return {
            'quality_scores': all_scores,
            'metadata': all_metadata
        }
    
    def filter_data_by_quality(self, data_loader: DataLoader, 
                              keep_fraction: float = 0.5) -> List[int]:
        """
        Filter data based on quality scores, keeping top fraction
        
        Args:
            data_loader: DataLoader containing demonstration data
            keep_fraction: Fraction of data to keep (0.5 = keep top 50%)
            
        Returns:
            List of indices for high-quality demonstrations
        """
        quality_results = self.estimate_quality_scores(data_loader)
        scores = quality_results['quality_scores']
        
        # Get indices of top quality demonstrations
        num_keep = int(len(scores) * keep_fraction)
        top_indices = np.argsort(scores)[-num_keep:]
        
        return top_indices.tolist()
    
    def save_models(self, save_dir: str):
        """Save trained VAE models"""
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save(self.state_vae.state_dict(), os.path.join(save_dir, 'state_vae.pth'))
        torch.save(self.action_vae.state_dict(), os.path.join(save_dir, 'action_vae.pth'))
        
        # Save configuration
        config = {
            'state_z_dim': self.state_vae.z_dim,
            'action_z_dim': self.action_vae.z_dim,
            'beta': self.state_vae.beta,
            'k_values': self.k_values
        }
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_models(self, save_dir: str):
        """Load trained VAE models"""
        self.state_vae.load_state_dict(torch.load(os.path.join(save_dir, 'state_vae.pth')))
        self.action_vae.load_state_dict(torch.load(os.path.join(save_dir, 'action_vae.pth')))
        
        with open(os.path.join(save_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        
        print(f"Loaded models with config: {config}")


class RoboMimicDataset(Dataset):
    """Simple dataset wrapper for RoboMimic data"""
    
    def __init__(self, data_dict: Dict):
        self.data = data_dict
        self.length = len(data_dict['observation']['state'])
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return {
            'observation': {
                'state': torch.FloatTensor(self.data['observation']['state'][idx])
            },
            'action': torch.FloatTensor(self.data['action'][idx]),
            'ep_idx': self.data.get('ep_idx', [0] * self.length)[idx],
            'dataset_id': self.data.get('dataset_id', [0] * self.length)[idx],
            'quality_score': self.data.get('quality_score', [0.0] * self.length)[idx]
        }


def demo_usage():
    """Demonstration of how to use the DemInf PyTorch implementation"""
    
    # Example usage with synthetic data
    print("DemInf PyTorch Implementation Demo")
    print("=" * 50)
    
    # Create synthetic demonstration data
    batch_size = 1000
    state_dim = 13  # Typical RoboMimic state dimension
    action_dim = 7  # Typical RoboMimic action dimension
    
    # Generate synthetic data
    states = torch.randn(batch_size, state_dim)
    actions = torch.randn(batch_size, action_dim)
    
    # Create dataset
    data_dict = {
        'observation': {'state': states.numpy()},
        'action': actions.numpy(),
        'ep_idx': list(range(batch_size)),
        'dataset_id': [0] * batch_size,
        'quality_score': [0.0] * batch_size
    }
    
    dataset = RoboMimicDataset(data_dict)
    data_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    # Initialize DemInf estimator
    deminf = DemInfQualityEstimator(
        state_dim=state_dim,
        action_dim=action_dim,
        state_z_dim=12,
        action_z_dim=6,
        hidden_dims=[512, 512],
        beta=0.05,
        k_values=[5, 6, 7]
    )
    
    print("Training VAEs...")
    # Train VAEs
    deminf.train_vae(data_loader, epochs=10, vae_type='state')
    deminf.train_vae(data_loader, epochs=10, vae_type='action')
    
    print("Estimating quality scores...")
    # Estimate quality scores
    quality_results = deminf.estimate_quality_scores(data_loader)
    scores = quality_results['quality_scores']
    
    print(f"Quality scores - Mean: {np.mean(scores):.4f}, Std: {np.std(scores):.4f}")
    print(f"Quality scores - Min: {np.min(scores):.4f}, Max: {np.max(scores):.4f}")
    
    # Filter data by quality
    print("Filtering data by quality...")
    top_indices = deminf.filter_data_by_quality(data_loader, keep_fraction=0.5)
    print(f"Kept {len(top_indices)} out of {batch_size} demonstrations")
    
    # Save models
    print("Saving models...")
    deminf.save_models('./deminf_models')
    
    print("Demo completed successfully!")


if __name__ == "__main__":
    demo_usage()

