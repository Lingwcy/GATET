import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pytorch_tabnet.tab_model import TabNetClassifier


class GATET(nn.Module):
    """
    GAT-Enhanced TabNet (GATET) for multi-disease coexistence risk prediction
    Architecture components:
    1. DFE (Dependency Feature Extraction) - Builds graph structure using medical prior knowledge
    2. CGsA (Graph Attention Aggregation) - Performs multi-head graph attention aggregation
    3. FWT (Feature Weighting TabNet) - Computes feature importance weights
    """

    def __init__(self, num_features, num_classes, age_groups, gat_params, tabnet_params):
        """
        Initialize GATET model

        Args:
            num_features: Number of input features
            num_classes: Number of output classes
            age_groups: Age group boundaries [0, 40, 60, 75, 100]
            gat_params: Dictionary of GAT parameters
            tabnet_params: Dictionary of TabNet parameters
        """
        super(GATET, self).__init__()
        self.age_groups = age_groups
        self.num_groups = len(age_groups) - 1

        # DFE Module: Graph structure will be built externally
        # CGsA Module: Graph Attention Network
        self.gat = GAT(
            in_channels=num_features,
            hidden_channels=gat_params['hidden_channels'],
            out_channels=gat_params['out_channels'],
            heads=gat_params['heads'],
            dropout=gat_params['dropout']
        )

        # FWT Module: TabNet-based feature weighting
        self.tabnet = TabNetFeatureWeighter(
            input_dim=num_features,
        ​ ** tabnet_params
        )

        # Fusion Classifier
        self.classifier = nn.Sequential(
            nn.Linear(gat_params['out_channels'], 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def build_graph(self, features, ages, labels):
        """
        DFE Module: Build graph structure based on medical prior knowledge (age groups)

        Args:
            features: Input features tensor [num_nodes, num_features]
            ages: Age values tensor [num_nodes]
            labels: Target labels tensor [num_nodes]

        Returns:
            edge_index: Graph connectivity [2, num_edges]
        """
        edge_indices = []

        # Group nodes by age and label
        for age_group in range(self.num_groups):
            lower, upper = self.age_groups[age_group], self.age_groups[age_group + 1]
            age_mask = (ages >= lower) & (ages < upper)

            for label in torch.unique(labels):
                label_mask = (labels == label)
                group_mask = age_mask & label_mask
                group_indices = torch.where(group_mask)[0]

                # Create fully connected subgraph
                for i in range(len(group_indices)):
                    for j in range(i + 1, len(group_indices)):
                        # Undirected edges (both directions)
                        edge_indices.append([group_indices[i], group_indices[j]])
                        edge_indices.append([group_indices[j], group_indices[i]])

        if not edge_indices:
            return torch.empty((2, 0), dtype=torch.long)

        return torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    def forward(self, x, edge_index):
        """
        Forward pass through GATET model

        Args:
            x: Input features [num_nodes, num_features]
            edge_index: Graph connectivity [2, num_edges]

        Returns:
            logits: Output predictions [num_nodes, num_classes]
        """
        # CGsA Module: Graph attention aggregation
        gat_out = self.gat(x, edge_index)  # [num_nodes, out_channels]

        # FWT Module: Feature weighting
        weights = self.tabnet(x)  # [num_nodes, num_features]
        weighted_features = x * weights  # [num_nodes, num_features]

        # Feature fusion: Element-wise multiplication
        fused = gat_out * weighted_features  # [num_nodes, out_channels]

        # Classification
        return self.classifier(fused)


class GAT(nn.Module):
    """Graph Attention Network (CGsA Module)"""

    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.6):
        """
        Initialize GAT model

        Args:
            in_channels: Input feature dimension
            hidden_channels: Hidden layer dimension
            out_channels: Output feature dimension
            heads: Number of attention heads
            dropout: Dropout probability
        """
        super(GAT, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat2 = GATConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            concat=False,
            dropout=dropout
        )
        self.dropout = dropout

    def forward(self, x, edge_index):
        """Forward pass through GAT layers"""
        x = F.leaky_relu(self.gat1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.gat2(x, edge_index)


class TabNetFeatureWeighter(nn.Module):
    """Feature Weighting Block (FWT Module)"""

    def __init__(self, input_dim, n_d=8, n_a=8, n_steps=3, gamma=1.3):
        """
        Initialize simplified TabNet feature weighting module

        Args:
            input_dim: Number of input features
            n_d: Dimension of prediction layer
            n_a: Dimension of attention layer
            n_steps: Number of sequential steps
            gamma: Sparsity regularization coefficient
        """
        super(TabNetFeatureWeighter, self).__init__()
        self.n_steps = n_steps
        self.tabnet = TabNetClassifier(
            input_dim=input_dim,
            output_dim=input_dim,  # Same as input for weighting
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma
        )
        self.feature_importances = None

    def fit(self, X, y):
        """Pre-train TabNet to obtain feature weights"""
        self.tabnet.fit(X, y, max_epochs=100, patience=20)
        self.feature_importances = torch.tensor(
            self.tabnet.feature_importances_,
            dtype=torch.float32
        )

    def forward(self, x):
        """Apply feature weights"""
        if self.feature_importances is None:
            raise RuntimeError("TabNet weights not initialized. Call fit() first.")
        return x * self.feature_importances.to(x.device)


def load_and_preprocess_data(file_path, age_groups):
    """
    Load and preprocess medical data

    Args:
        file_path: Path to Excel data file
        age_groups: Age group boundaries for graph construction

    Returns:
        Dictionary containing processed data tensors
    """
    # Load data from Excel
    df = pd.read_excel(file_path)

    # Extract features, labels, and ages
    features = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values
    ages = df.iloc[:, -7].values  # Age is in the 7th last column

    # Standardize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Split data
    X_train, X_test, y_train, y_test, age_train, age_test = train_test_split(
        features, labels, ages, test_size=0.2, random_state=42
    )

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    age_train = torch.tensor(age_train, dtype=torch.long)
    age_test = torch.tensor(age_test, dtype=torch.long)

    return {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'age_train': age_train, 'age_test': age_test,
        'feature_names': df.columns[:-1].tolist()
    }


def train_gatet(model, data, epochs=100, lr=0.005):
    """
    Train GATET model

    Args:
        model: Initialized GATET model
        data: Dictionary containing training data
        epochs: Number of training epochs
        lr: Learning rate
    """
    # Build graph structure
    edge_index = model.build_graph(
        data['X_train'],
        data['age_train'],
        data['y_train']
    )

    # Set up optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        logits = model(data['X_train'], edge_index)
        loss = criterion(logits, data['y_train'])

        # Backward pass
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(data['X_test'], model.build_graph(
                data['X_test'], data['age_test'], data['y_test']))
            val_loss = criterion(val_logits, data['y_test'])
            _, preds = torch.max(val_logits, 1)
            acc = (preds == data['y_test']).float().mean()

        # Print metrics
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} | "
                  f"Train Loss: {loss.item():.4f} | "
                  f"Val Loss: {val_loss.item():.4f} | "
                  f"Val Acc: {acc:.4f}")


# Configuration parameters (as in the paper)
CONFIG = {
    'data_path': 'medical_data.xlsx',
    'age_groups': [0, 40, 60, 75, 100],
    'gat_params': {
        'hidden_channels': 8,
        'out_channels': 32,  # Should match input feature dim
        'heads': 8,
        'dropout': 0.6
    },
    'tabnet_params': {
        'n_d': 8,
        'n_a': 8,
        'n_steps': 8,  # Optimal per paper results
        'gamma': 1.3
    },
    'training': {
        'epochs': 100,
        'lr': 0.005
    }
}

if __name__ == "__main__":
    # Load and preprocess data
    data = load_and_preprocess_data(CONFIG['data_path'], CONFIG['age_groups'])
    num_features = data['X_train'].shape[1]
    num_classes = len(torch.unique(data['y_train']))

    # Initialize model
    model = GATET(
        num_features=num_features,
        num_classes=num_classes,
        age_groups=CONFIG['age_groups'],
        gat_params=CONFIG['gat_params'],
        tabnet_params=CONFIG['tabnet_params']
    )

    # Pre-train TabNet (FWT Module)
    model.tabnet.fit(
        data['X_train'].numpy(),
        data['y_train'].numpy()
    )

    # Train GATET model
    train_gatet(
        model,
        data,
        epochs=CONFIG['training']['epochs'],
        lr=CONFIG['training']['lr']
    )