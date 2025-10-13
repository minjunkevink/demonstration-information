"""
Example script showing how to use DemInf PyTorch implementation with RoboMimic data
"""

import torch
import numpy as np
import h5py
from torch.utils.data import DataLoader
from deminf_pytorch import DemInfQualityEstimator, RoboMimicDataset
import os


def load_robomimic_data(data_path: str):
    """
    Load RoboMimic HDF5 data and convert to format expected by DemInf
    
    Args:
        data_path: Path to RoboMimic HDF5 file
        
    Returns:
        Dictionary with states, actions, and metadata
    """
    print(f"Loading RoboMimic data from {data_path}")
    
    all_states = []
    all_actions = []
    ep_idx = []
    dataset_id = []
    quality_score = []
    
    with h5py.File(data_path, 'r') as f:
        demo_keys = [k for k in f['data'].keys() if k.startswith('demo_')]
        print(f"Found {len(demo_keys)} demonstrations")
        
        for i, demo_key in enumerate(demo_keys):
            demo = f['data'][demo_key]
            
            # Get states (concatenate all observation components)
            obs_components = []
            for obs_key in demo['obs'].keys():
                obs_components.append(demo['obs'][obs_key][:])
            
            # Concatenate all observation components
            states = np.concatenate(obs_components, axis=1)
            actions = demo['actions'][:]
            
            all_states.append(states)
            all_actions.append(actions)
            
            # Create episode indices for this demo
            ep_idx.extend([i] * len(states))
            dataset_id.extend([0] * len(states))  # All from same dataset
            quality_score.extend([0.0] * len(states))  # Placeholder
    
    # Concatenate all demonstrations
    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    
    print(f"Loaded {len(states)} timesteps across {len(demo_keys)} episodes")
    print(f"State shape: {states.shape}, Action shape: {actions.shape}")
    
    return {
        'observation': {'state': states},
        'action': actions,
        'ep_idx': ep_idx,
        'dataset_id': dataset_id,
        'quality_score': quality_score
    }


def main():
    """Main example function"""
    print("DemInf PyTorch - RoboMimic Example")
    print("=" * 50)
    
    # Path to downloaded RoboMimic data
    robomimic_data_path = "/scr/kimkj/beta/demonstration-information/robomimic/datasets/lift/ph/low_dim.hdf5"
    
    if not os.path.exists(robomimic_data_path):
        print(f"Error: RoboMimic data not found at {robomimic_data_path}")
        print("Please make sure you have downloaded the RoboMimic datasets first.")
        return
    
    # Load data
    data_dict = load_robomimic_data(robomimic_data_path)
    
    # Create dataset and dataloader
    dataset = RoboMimicDataset(data_dict)
    data_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    # Get dimensions from data
    state_dim = data_dict['observation']['state'].shape[1]
    action_dim = data_dict['action'].shape[1]
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Initialize DemInf estimator
    deminf = DemInfQualityEstimator(
        state_dim=state_dim,
        action_dim=action_dim,
        state_z_dim=12,  # From paper config
        action_z_dim=6,  # From paper config
        hidden_dims=[512, 512],  # From paper config
        beta=0.05,  # From paper config
        k_values=[5, 6, 7],  # From paper config
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"Using device: {deminf.device}")
    
    # Train VAEs
    print("\nTraining State VAE...")
    deminf.train_vae(data_loader, epochs=20, vae_type='state')
    
    print("\nTraining Action VAE...")
    deminf.train_vae(data_loader, epochs=20, vae_type='action')
    
    # Estimate quality scores
    print("\nEstimating quality scores...")
    quality_results = deminf.estimate_quality_scores(data_loader)
    scores = quality_results['quality_scores']
    
    print(f"\nQuality Score Statistics:")
    print(f"  Mean: {np.mean(scores):.4f}")
    print(f"  Std:  {np.std(scores):.4f}")
    print(f"  Min:  {np.min(scores):.4f}")
    print(f"  Max:  {np.max(scores):.4f}")
    print(f"  Median: {np.median(scores):.4f}")
    
    # Show distribution
    print(f"\nQuality Score Percentiles:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  {p}th percentile: {np.percentile(scores, p):.4f}")
    
    # Filter data by quality
    print("\nFiltering data by quality...")
    for keep_frac in [0.1, 0.25, 0.5, 0.75, 0.9]:
        top_indices = deminf.filter_data_by_quality(data_loader, keep_fraction=keep_frac)
        print(f"  Keep top {keep_frac*100:.0f}%: {len(top_indices)} demonstrations")
    
    # Save models
    print("\nSaving trained models...")
    save_dir = "./deminf_robomimic_models"
    deminf.save_models(save_dir)
    print(f"Models saved to {save_dir}")
    
    # Save quality scores
    np.save(os.path.join(save_dir, "quality_scores.npy"), scores)
    print(f"Quality scores saved to {save_dir}/quality_scores.npy")
    
    print("\nExample completed successfully!")
    print("\nNext steps:")
    print("1. Use the quality scores to filter your demonstration data")
    print("2. Train imitation learning policies on the filtered high-quality data")
    print("3. Compare performance with policies trained on all data")


if __name__ == "__main__":
    main()
