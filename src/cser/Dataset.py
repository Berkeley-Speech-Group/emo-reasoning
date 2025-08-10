import torch
from torch.utils.data import Dataset
import pandas as pd
import os

class MSPFeatureDataset(Dataset):
    """
    A PyTorch Dataset for loading pre-extracted audio features and their
    corresponding continuous emotion labels (Valence, Arousal, Dominance).
    """
    def __init__(self, feature_dir: str, label_dir: str):
        """
        Args:
            feature_dir (str): Path to the directory containing feature files (.pt).
            label_dir (str): Path to the directory containing label files (.csv).
        """
        self.feature_dir = feature_dir
        self.label_dir = label_dir

        # Get all feature file names (assuming they end with '_features.pt')
        self.feature_files = [
            f for f in os.listdir(feature_dir) if f.endswith('_features.pt')
        ]

    def __len__(self) -> int:
        return len(self.feature_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a single sample (features and labels) from the dataset.
        """
        feature_file = self.feature_files[idx]
        # Assumes label filename corresponds to the feature filename
        label_file = feature_file.replace('_features.pt', '.csv')
        
        # Load features
        feature_path = os.path.join(self.feature_dir, feature_file)
        features = torch.load(feature_path)

        # Load labels
        label_path = os.path.join(self.label_dir, label_file)
        labels_df = pd.read_csv(label_path)
        
        # Normalize labels from [0, 100] to [0, 1.0] for training
        labels_df[['arousal', 'valence', 'dominance']] = labels_df[['arousal', 'valence', 'dominance']] / 100.0
        
        # Extract VAD values and convert to a tensor
        labels = labels_df[['arousal', 'valence', 'dominance']].values
        labels = torch.tensor(labels, dtype=torch.float32)
        
        return features, labels