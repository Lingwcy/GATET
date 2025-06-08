# GATET: GAT-Enhanced TabNet for Multi-Disease Risk Prediction

## Overview

This implementation of **GATET (GAT-Enhanced TabNet)** provides a deep learning framework for predicting multi-disease coexistence risk from medical tabular data. The model integrates three key modules from the research paper:

1. **DFE (Dependency Feature Extraction)** - Constructs graph structure using medical prior knowledge
2. **CGsA (Graph Attention Aggregation)** - Performs multi-head graph attention
3. **FWT (Feature Weighting TabNet)** - Computes feature importance weights

The model achieves state-of-the-art performance (91.06% accuracy) by capturing complex relationships in electronic health records.

## Key Components

### 1. Main Classes

#### GATET (`GATET` class)

python

```python
class GATET(nn.Module):
    """
    Main GAT-Enhanced TabNet model integrating:
    - DFE: Dependency Feature Extraction (graph construction)
    - CGsA: Graph Attention Aggregation
    - FWT: Feature Weighting TabNet
    
    Args:
        num_features: Input feature dimension
        num_classes: Output class dimension
        age_groups: Age boundaries for medical grouping
        gat_params: Parameters for Graph Attention Network
        tabnet_params: Parameters for TabNet weighting
    """
```

#### Graph Attention Network (`GAT` class)

python

```python
class GAT(nn.Module):
    """
    Graph Attention Network (CGsA Module) with:
    - Multi-head attention mechanism
    - LeakyReLU activations
    - Dropout regularization
    
    Implements equations:
    a_ij = exp(e_ij) / Σ_k∈N(i) exp(e_ik)
    e_ij = a(Wh_i, Wh_j)
    h_i' = σ(Σ_j∈N(i) a_ij W h_j)
    """
```

#### Feature Weighting Module (`TabNetFeatureWeighter` class)

```python
class TabNetFeatureWeighter(nn.Module):
    """
    Feature Weighting Block (FWT Module) using simplified TabNet:
    - Pre-trains TabNet to obtain feature importance weights
    - Applies feature-wise multiplication for importance weighting
    - Uses optimal step length (8) identified in paper
    
    Implements TabNet's attentive transformer:
    h(X) = (W*X + b) ⊗ σ(V*X + c)
    """
```

### 2. Core Methods

#### Graph Construction (`build_graph`)

```python
def build_graph(self, features, ages, labels):
    """
    DFE Module: Medical prior-guided graph construction
    - Groups patients by age and disease labels
    - Creates fully-connected subgraphs within groups
    - Implements medical prior knowledge (age-based grouping)
    """
```

#### Feature Fusion (in `forward`)

python

```python
# CGsA: Graph attention aggregation
gat_out = self.gat(x, edge_index)  # [num_nodes, out_channels]

# FWT: Feature weighting
weights = self.tabnet(x)  # [num_nodes, num_features]
weighted_features = x * weights  # [num_nodes, num_features]

# Feature fusion: Element-wise multiplication
fused = gat_out * weighted_features  # [num_nodes, out_channels]
```

#### Training Pipeline (`train_gatet`)

python

```python
def train_gatet(model, data, epochs=100, lr=0.005):
    """
    End-to-end training pipeline:
    1. Builds graph structure using DFE
    2. Pre-trains TabNet (FWT)
    3. Trains GAT (CGsA)
    4. Performs feature fusion
    5. Outputs predictions
    """
```

## Configuration Parameters

python

```python
CONFIG = {
    'data_path': 'medical_data.xlsx',
    'age_groups': [0, 40, 60, 75, 100],  # Medical age grouping
    'gat_params': {                      # Graph Attention Network
        'hidden_channels': 8,
        'out_channels': 32,  # Should match input feature dim
        'heads': 8,          # Multi-head attention
        'dropout': 0.6       # Regularization
    },
    'tabnet_params': {        # Feature Weighting
        'n_d': 8,             # Prediction layer dimension
        'n_a': 8,             # Attention layer dimension
        'n_steps': 8,         # Optimal per paper results
        'gamma': 1.3          # Sparsity regularization
    },
    'training': {             # Optimization
        'epochs': 100,        # Training iterations
        'lr': 0.005           # Learning rate
    }
}
```

## Usage Example

python

```python
# Load and preprocess medical data
data = load_and_preprocess_data('patient_records.xlsx', [0, 40, 60, 75, 100])

# Initialize GATET model
model = GATET(
    num_features=41,  # 41 medical features
    num_classes=2,    # Binary classification: at-risk/not-at-risk
    age_groups=[0, 40, 60, 75, 100],
    gat_params={'hidden_channels': 8, 'out_channels': 32, 'heads': 8, 'dropout': 0.6},
    tabnet_params={'n_d': 8, 'n_a': 8, 'n_steps': 8, 'gamma': 1.3}
)

# Pre-train TabNet for feature weighting
model.tabnet.fit(data['X_train'].numpy(), data['y_train'].numpy())

# Train end-to-end model
train_gatet(model, data, epochs=100, lr=0.005)

# Example inference
test_graph = model.build_graph(data['X_test'], data['age_test'], data['y_test'])
logits = model(data['X_test'], test_graph)
predictions = torch.softmax(logits, dim=1)
```

## References

1. Arik & Pfister. "TabNet: Attentive Interpretable Tabular Learning" (2020)
2. Velickovic et al. "Graph Attention Networks" (2018)