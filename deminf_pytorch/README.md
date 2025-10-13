# DemInf PyTorch Implementation

This is a direct PyTorch reimplementation of the **DemInf (Demonstration Information Estimation)** method from the paper "Robot Data Curation with Mutual Information Estimators".

## Overview

DemInf estimates the quality of robot demonstrations by measuring the mutual information between states and actions. High-quality demonstrations should have:
- **High action diversity** (high marginal action entropy)
- **Predictable actions given states** (low conditional action entropy)

This leads to high mutual information I(S;A) = H(A) - H(A|S).

## Key Components

### 1. VAE Embeddings
- **State VAE**: Learns low-dimensional representations of robot states
- **Action VAE**: Learns low-dimensional representations of robot actions
- Uses β-VAE with β=0.05 for balanced reconstruction and regularization

### 2. KSG Mutual Information Estimator
- **Kraskov-Stögbauer-Grassberger** estimator using k-nearest neighbors
- Estimates I(S;A) using distances in the joint state-action space
- Uses InfNorm (maximum of state and action distances) for joint distances

### 3. Quality Estimation Pipeline
- Trains VAEs on demonstration data
- Estimates mutual information for each state-action pair
- Aggregates scores per demonstration/episode
- Filters data based on quality scores

## Usage

### Basic Usage

```python
from deminf_pytorch import DemInfQualityEstimator, RoboMimicDataset
from torch.utils.data import DataLoader

# Initialize estimator
deminf = DemInfQualityEstimator(
    state_dim=13,      # Robot state dimension
    action_dim=7,      # Robot action dimension
    state_z_dim=12,    # State embedding dimension
    action_z_dim=6,    # Action embedding dimension
    hidden_dims=[512, 512],
    beta=0.05,
    k_values=[5, 6, 7]
)

# Create dataset and dataloader
dataset = RoboMimicDataset(your_data_dict)
data_loader = DataLoader(dataset, batch_size=256, shuffle=True)

# Train VAEs
deminf.train_vae(data_loader, epochs=50, vae_type='state')
deminf.train_vae(data_loader, epochs=50, vae_type='action')

# Estimate quality scores
quality_results = deminf.estimate_quality_scores(data_loader)
scores = quality_results['quality_scores']

# Filter data by quality (keep top 50%)
top_indices = deminf.filter_data_by_quality(data_loader, keep_fraction=0.5)
```

### With RoboMimic Data

```python
# Run the example script
python example_robomimic.py
```

This will:
1. Load RoboMimic HDF5 data
2. Train state and action VAEs
3. Estimate quality scores for all demonstrations
4. Show quality statistics and filtering results
5. Save trained models and quality scores

## Data Format

The implementation expects data in this format:

```python
data_dict = {
    'observation': {
        'state': np.array,  # Shape: (n_timesteps, state_dim)
    },
    'action': np.array,     # Shape: (n_timesteps, action_dim)
    'ep_idx': list,         # Episode indices for each timestep
    'dataset_id': list,     # Dataset identifiers
    'quality_score': list   # Placeholder quality scores
}
```

## Hyperparameters

Based on the original paper:

- **State embedding dimension**: 12 (for RoboMimic)
- **Action embedding dimension**: 6 (for RoboMimic)
- **VAE hidden dimensions**: [512, 512]
- **β (VAE regularization)**: 0.05
- **k values for KSG**: [5, 6, 7]
- **Learning rate**: 1e-4
- **Batch size**: 256

## Key Differences from Original

1. **Framework**: PyTorch instead of JAX/Flax
2. **Simplified**: Single file implementation, less modular
3. **Direct**: No complex configuration system
4. **Compatible**: Same core algorithms and hyperparameters

## Expected Results

According to the paper, using DemInf to filter demonstration data leads to:
- **5-10% improvement** in RoboMimic benchmark performance
- Better performance on real robot setups (ALOHA, Franka)
- Consistent quality ranking compared to human expert scores

## Files

- `deminf_pytorch.py`: Main implementation
- `example_robomimic.py`: Example usage with RoboMimic data
- `README.md`: This documentation

## Dependencies

```bash
pip install torch numpy scipy h5py tqdm
```

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{hejna2025robot,
  title={Robot Data Curation with Mutual Information Estimators},
  author={Hejna, Joey and Mirchandani, Suvir and Balakrishna, Ashwin and Xie, Annie and Wahid, Ayzaan and Tompson, Jonathan and Sanketi, Pannag and Shah, Dhruv and Devin, Coline and Sadigh, Dorsa},
  journal={arXiv preprint arXiv:2502.08623},
  year={2025}
}
```

