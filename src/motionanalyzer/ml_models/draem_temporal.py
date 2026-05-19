"""
Temporal anomaly detection using LSTM/GRU autoencoder for FPCB crack detection.

Models temporal dependencies in time-series sequences to detect anomalies
by reconstruction error. Trained on normal sequences only.

References:
- LSTM-Autoencoder for time-series anomaly detection (autocorrelation-based windowing)
- Window length: optimal around 10 frames (performance degrades at 20+)
- Reconstruction error scoring: threshold-based detection

Design:
- Input: Sequences of per-frame features (T frames × D features)
- Architecture: LSTM/GRU encoder-decoder
- Scoring: Reconstruction error per sequence → frame-level scores (max/mean aggregation)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd


class TemporalAnomalyDetector:
    """
    Temporal anomaly detector using LSTM/GRU autoencoder.
    
    Trains on normal sequences only, detects anomalies via reconstruction error.
    """

    def __init__(
        self,
        feature_dim: int,
        sequence_length: int = 10,
        hidden_dim: int = 64,
        num_layers: int = 2,
        cell_type: Literal["LSTM", "GRU"] = "LSTM",
        learning_rate: float = 1e-3,
        batch_size: int = 32,
    ) -> None:
        """
        Initialize temporal anomaly detector.
        
        Args:
            feature_dim: Feature dimension (D) per frame
            sequence_length: Sequence length (T) - number of frames per window
            hidden_dim: Hidden dimension for LSTM/GRU cells
            num_layers: Number of LSTM/GRU layers
            cell_type: "LSTM" or "GRU"
            learning_rate: Training learning rate
            batch_size: Batch size for training
        """
        self.feature_dim = feature_dim
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model: Optional[Any] = None
        self.is_trained = False
        self.reconstruction_threshold: Optional[float] = None
        self._torch: Any = None
        self._device: Any = None

    def _ensure_torch(self) -> None:
        """Ensure PyTorch is available."""
        try:
            import torch
            self._torch = torch
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError as e:
            raise ImportError(
                "Temporal model requires PyTorch. Install with: pip install -e '.[ml]'"
            ) from e

    def _build_model(self) -> None:
        """Build LSTM/GRU autoencoder model."""
        if self._torch is None:
            self._ensure_torch()

        torch = self._torch
        device = self._device

        class TemporalAutoencoder(torch.nn.Module):
            def __init__(
                self,
                input_dim: int,
                hidden_dim: int,
                num_layers: int,
                cell_type: str,
            ):
                super().__init__()
                self.input_dim = input_dim
                self.hidden_dim = hidden_dim
                self.num_layers = num_layers
                self.cell_type = cell_type

                # Encoder
                if cell_type == "LSTM":
                    self.encoder = torch.nn.LSTM(
                        input_dim, hidden_dim, num_layers, batch_first=True
                    )
                    self.decoder = torch.nn.LSTM(
                        hidden_dim, hidden_dim, num_layers, batch_first=True
                    )
                else:  # GRU
                    self.encoder = torch.nn.GRU(
                        input_dim, hidden_dim, num_layers, batch_first=True
                    )
                    self.decoder = torch.nn.GRU(
                        hidden_dim, hidden_dim, num_layers, batch_first=True
                    )

                # Output projection
                self.output_proj = torch.nn.Linear(hidden_dim, input_dim)

            def forward(self, x: Any) -> tuple[Any, Any]:
                """
                Forward pass: encode sequence, decode, reconstruct.
                
                Args:
                    x: Input sequence (batch_size, seq_len, input_dim)
                
                Returns:
                    (reconstructed, hidden_state)
                """
                # Encode
                encoded, hidden = self.encoder(x)

                # Decode (use last hidden state as initial state)
                # For decoder, we need to provide input sequence
                # Use encoded sequence as decoder input
                decoded, _ = self.decoder(encoded, hidden)

                # Project to input dimension
                reconstructed = self.output_proj(decoded)

                return reconstructed, hidden

        self.model = TemporalAutoencoder(
            input_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            cell_type=self.cell_type,
        ).to(device)

    def _build_sequences(
        self,
        features_df: pd.DataFrame,
        feature_cols: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build sequences from per-frame features.
        
        Groups by dataset_path, sorts by frame, creates sliding windows.
        
        Args:
            features_df: DataFrame with 'dataset_path', 'frame', and feature columns
            feature_cols: List of feature column names
        
        Returns:
            (sequences, frame_indices) where:
            - sequences: (n_sequences, seq_len, feature_dim) array
            - frame_indices: (n_sequences, seq_len) array of (dataset_path_idx, frame) tuples
        """
        sequences = []
        frame_indices = []

        # Group by dataset_path
        for dataset_path, group in features_df.groupby("dataset_path"):
            group_sorted = group.sort_values("frame").reset_index(drop=True)
            frames = group_sorted["frame"].values
            features = group_sorted[feature_cols].fillna(0.0).values.astype(np.float32)

            # Create sliding windows
            for i in range(len(features) - self.sequence_length + 1):
                seq = features[i : i + self.sequence_length]
                seq_frames = frames[i : i + self.sequence_length]
                sequences.append(seq)
                frame_indices.append(seq_frames)

        if not sequences:
            return np.empty((0, self.sequence_length, self.feature_dim)), np.empty((0, self.sequence_length))

        sequences_array = np.array(sequences, dtype=np.float32)
        frame_indices_array = np.array(frame_indices, dtype=int)

        return sequences_array, frame_indices_array

    def fit(
        self,
        normal_data: pd.DataFrame,
        feature_cols: list[str],
        epochs: int = 50,
    ) -> None:
        """
        Train on normal sequences only.
        
        Args:
            normal_data: Normal training data (must have 'dataset_path', 'frame', and feature columns)
            feature_cols: List of feature column names
            epochs: Training epochs
        """
        if self._torch is None:
            self._ensure_torch()

        if self.model is None:
            self._build_model()

        torch = self._torch
        device = self._device

        # Build sequences
        sequences, _ = self._build_sequences(normal_data, feature_cols)
        if len(sequences) == 0:
            raise ValueError("No sequences built from normal data")

        # Convert to tensor
        train_data = torch.from_numpy(sequences).to(device)

        self.model.train()
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        n_samples = len(train_data)
        n_batches = max(1, (n_samples + self.batch_size - 1) // self.batch_size)

        for epoch in range(epochs):
            total_loss = 0.0
            indices = torch.randperm(n_samples, device=device)
            for i in range(n_batches):
                start_idx = i * self.batch_size
                end_idx = min(start_idx + self.batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                x_batch = train_data[batch_indices]

                optimizer.zero_grad()
                reconstructed, _ = self.model(x_batch)
                loss = criterion(reconstructed, x_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / n_batches
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

        self.is_trained = True

    def predict(
        self,
        data: pd.DataFrame,
        feature_cols: list[str],
        aggregation: Literal["max", "mean"] = "max",
    ) -> pd.DataFrame:
        """
        Predict anomaly scores per frame.
        
        Args:
            data: Test data (must have 'dataset_path', 'frame', and feature columns)
            feature_cols: List of feature column names
            aggregation: How to aggregate sequence scores to frame scores ("max" or "mean")
        
        Returns:
            DataFrame with columns: ['dataset_path', 'frame', 'anomaly_score']
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")

        if self._torch is None:
            self._ensure_torch()

        torch = self._torch
        device = self._device

        # Build sequences
        sequences, frame_indices = self._build_sequences(data, feature_cols)
        if len(sequences) == 0:
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=["dataset_path", "frame", "anomaly_score"])

        # Convert to tensor
        test_data = torch.from_numpy(sequences).to(device)

        self.model.eval()
        criterion = torch.nn.MSELoss(reduction="none")

        # Compute reconstruction error per sequence
        with torch.no_grad():
            reconstructed, _ = self.model(test_data)
            # Reconstruction error per timestep: (batch, seq_len, feature_dim)
            errors = criterion(reconstructed, test_data)
            # Sum over feature dimension: (batch, seq_len)
            errors_per_timestep = errors.sum(dim=2)
            # Mean over sequence length: (batch,)
            sequence_scores = errors_per_timestep.mean(dim=1).cpu().numpy()

        # Aggregate sequence scores to frame scores
        # Each frame appears in multiple sequences (sliding window)
        frame_scores_dict: dict[tuple[str, int], list[float]] = {}
        dataset_paths = data["dataset_path"].unique()

        for seq_idx, (seq_score, seq_frames) in enumerate(zip(sequence_scores, frame_indices)):
            # Find dataset_path for this sequence
            # Use first frame's dataset_path (all frames in sequence are from same dataset)
            first_frame_idx = seq_frames[0]
            matching_rows = data[(data["frame"] == first_frame_idx)]
            if len(matching_rows) > 0:
                dataset_path = matching_rows.iloc[0]["dataset_path"]
            else:
                continue

            # Assign score to each frame in sequence
            for frame_idx in seq_frames:
                key = (dataset_path, int(frame_idx))
                if key not in frame_scores_dict:
                    frame_scores_dict[key] = []
                frame_scores_dict[key].append(float(seq_score))

        # Aggregate scores per frame
        frame_scores_list = []
        for (dataset_path, frame), scores in frame_scores_dict.items():
            if aggregation == "max":
                score = max(scores)
            else:  # mean
                score = np.mean(scores)
            frame_scores_list.append({
                "dataset_path": dataset_path,
                "frame": frame,
                "anomaly_score": score,
            })

        result_df = pd.DataFrame(frame_scores_list)
        return result_df

    def predict_binary(
        self,
        data: pd.DataFrame,
        feature_cols: list[str],
        threshold: Optional[float] = None,
        aggregation: Literal["max", "mean"] = "max",
    ) -> pd.DataFrame:
        """
        Predict binary labels (0=normal, 1=anomaly) per frame.
        
        Args:
            data: Test data
            feature_cols: List of feature column names
            threshold: Anomaly threshold (if None, use self.reconstruction_threshold)
            aggregation: How to aggregate sequence scores to frame scores
        
        Returns:
            DataFrame with columns: ['dataset_path', 'frame', 'is_anomaly']
        """
        scores_df = self.predict(data, feature_cols, aggregation=aggregation)
        thresh = threshold if threshold is not None else self.reconstruction_threshold
        if thresh is None:
            raise ValueError("Threshold not set. Call set_threshold_from_normal() first.")

        scores_df["is_anomaly"] = (scores_df["anomaly_score"] > thresh).astype(int)
        return scores_df[["dataset_path", "frame", "is_anomaly"]]

    def set_threshold_from_normal(
        self,
        normal_data: pd.DataFrame,
        feature_cols: list[str],
        percentile: float = 95.0,
    ) -> None:
        """
        Set anomaly threshold from normal data reconstruction errors.
        
        Args:
            normal_data: Normal data (validation set)
            feature_cols: List of feature column names
            percentile: Percentile for threshold (e.g., 95 = p95)
        """
        scores_df = self.predict(normal_data, feature_cols)
        if len(scores_df) == 0:
            raise ValueError("No scores computed from normal data")
        self.reconstruction_threshold = float(np.percentile(scores_df["anomaly_score"], percentile))

    def save(self, path: Path) -> None:
        """Save model weights and config."""
        if self._torch is None:
            self._ensure_torch()

        if self.model is None:
            raise ValueError("Model not built. Call fit() first.")

        torch = self._torch
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "feature_dim": self.feature_dim,
            "sequence_length": self.sequence_length,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "cell_type": self.cell_type,
            "reconstruction_threshold": self.reconstruction_threshold,
        }
        torch.save(save_dict, path)

    def load(self, path: Path) -> None:
        """Load model weights and config."""
        if self._torch is None:
            self._ensure_torch()

        torch = self._torch
        device = self._device

        checkpoint = torch.load(path, map_location=device)
        self.feature_dim = checkpoint["feature_dim"]
        self.sequence_length = checkpoint["sequence_length"]
        self.hidden_dim = checkpoint["hidden_dim"]
        self.num_layers = checkpoint["num_layers"]
        self.cell_type = checkpoint["cell_type"]
        self.reconstruction_threshold = checkpoint.get("reconstruction_threshold")

        self._build_model()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.is_trained = True
