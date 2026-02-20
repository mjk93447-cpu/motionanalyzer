"""
DREAM (DRAEM strategy) for FPCB bending copper-wire crack detection.

DREAM follows the strategy of:
  Zavrtanik, V., Kristan, M., & Skočaj, D. (2021). DRAEM - A Discriminatively
  Trained Reconstruction Embedding for Surface Anomaly Detection. ICCV 2021, 8330-8339.
  arXiv:2108.07610. Code: https://github.com/VitjanZ/DRAEM

Strategy: train on normal data only; use synthetic anomalies (e.g. normal + noise)
to train (1) a reconstructive subnetwork that maps input toward normal, and
(2) a discriminative subnetwork on (input, reconstruction) to separate normal vs anomaly.
Inference: anomaly score from discriminator and/or reconstruction error.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


class DREAMAnomalyDetector(ABC):
    """
    Deep Reconstruction Error-based Anomaly Model for FPCB crack detection.

    Trains on normal data only, detects anomalies via reconstruction error.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [64, 32, 16],
        latent_dim: int = 8,
        learning_rate: float = 1e-3,
    ) -> None:
        """
        Initialize DREAM model.

        Args:
            input_dim: Input feature dimension (e.g., from vectors.csv columns)
            hidden_dims: Encoder/decoder hidden layer dimensions
            latent_dim: Latent space dimension
            learning_rate: Training learning rate
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.is_trained = False
        self.reconstruction_error_threshold: Optional[float] = None

    @abstractmethod
    def fit(self, normal_data: pd.DataFrame | np.ndarray, epochs: int = 100) -> None:
        """
        Train on normal data only.

        Args:
            normal_data: Normal FPCB bending data (vectors.csv-like or feature array)
            epochs: Training epochs
        """
        raise NotImplementedError("Subclass must implement fit()")

    @abstractmethod
    def predict(self, data: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores (reconstruction error).

        Args:
            data: Test data (same format as fit)

        Returns:
            Anomaly scores (higher = more anomalous). Shape: (n_samples,)
        """
        raise NotImplementedError("Subclass must implement predict()")

    @abstractmethod
    def predict_binary(self, data: pd.DataFrame | np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """
        Predict binary labels (0=normal, 1=anomaly).

        Args:
            data: Test data
            threshold: Anomaly threshold (if None, use self.reconstruction_error_threshold)

        Returns:
            Binary labels (0 or 1). Shape: (n_samples,)
        """
        raise NotImplementedError("Subclass must implement predict_binary()")

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model weights and config."""
        raise NotImplementedError("Subclass must implement save()")

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load model weights and config."""
        raise NotImplementedError("Subclass must implement load()")

    def set_threshold_from_normal(self, normal_data: pd.DataFrame | np.ndarray, percentile: float = 95.0) -> None:
        """
        Set anomaly threshold from normal data reconstruction errors.

        Args:
            normal_data: Normal data (validation set)
            percentile: Percentile for threshold (e.g., 95 = p95)
        """
        scores = self.predict(normal_data)
        self.reconstruction_error_threshold = float(np.percentile(scores, percentile))

    def optimize_threshold_for_precision_recall(
        self,
        normal_data: pd.DataFrame | np.ndarray,
        anomaly_data: pd.DataFrame | np.ndarray,
        target_metric: str = "f1",
    ) -> tuple[float, dict[str, float]]:
        """
        Optimize threshold to maximize F1 or balance precision-recall.

        Args:
            normal_data: Normal validation data
            anomaly_data: Anomaly validation data
            target_metric: "f1", "precision", "recall", or "balanced" (F1 with recall >= 0.7)

        Returns:
            (optimal_threshold, metrics_dict) where metrics_dict has precision, recall, f1, accuracy
        """
        try:
            from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        except ImportError:
            raise ImportError("Threshold optimization requires scikit-learn")

        normal_scores = self.predict(normal_data)
        anomaly_scores = self.predict(anomaly_data)
        all_scores = np.concatenate([normal_scores, anomaly_scores])
        all_labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(anomaly_scores))])

        # Try thresholds from p50 to p99.9 of normal scores
        thresholds = np.percentile(normal_scores, np.linspace(50, 99.9, 100))
        best_metric = -1.0
        best_threshold = float(np.percentile(normal_scores, 95.0))
        best_metrics = {}

        for thresh in thresholds:
            pred = (all_scores > thresh).astype(int)
            prec = precision_score(all_labels, pred, zero_division=0)
            rec = recall_score(all_labels, pred, zero_division=0)
            f1 = f1_score(all_labels, pred, zero_division=0)
            acc = accuracy_score(all_labels, pred)

            if target_metric == "f1":
                metric_val = f1
            elif target_metric == "precision":
                metric_val = prec
            elif target_metric == "recall":
                metric_val = rec
            elif target_metric == "balanced":
                # F1 but require recall >= 0.7
                metric_val = f1 if rec >= 0.7 else f1 * 0.5
            else:
                metric_val = f1

            if metric_val > best_metric:
                best_metric = metric_val
                best_threshold = float(thresh)
                best_metrics = {"precision": prec, "recall": rec, "f1": f1, "accuracy": acc}

        self.reconstruction_error_threshold = best_threshold
        return best_threshold, best_metrics


class DREAMPyTorch(DREAMAnomalyDetector):
    """
    DREAM with DRAEM strategy: reconstructive AE + discriminative head.

    - Reconstructor: input_dim -> hidden_dims -> latent_dim -> hidden_dims -> input_dim.
    - Discriminative (optional): concat(input, reconstruction) -> MLP -> P(anomaly).
    - Training: normal only; synthetic anomalies = normal + Gaussian noise. Loss = L_recon + λ * L_disc.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [64, 32, 16],
        latent_dim: int = 8,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        device: str | None = None,
        use_discriminative: bool = True,
        synthetic_noise_std: float = 0.3,
        discriminator_weight: float = 0.5,
        weight_decay: float = 1e-5,
    ) -> None:
        super().__init__(input_dim, hidden_dims, latent_dim, learning_rate)
        self._check_pytorch()
        if device is None:
            device = "cuda" if self.torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.device = device
        self.use_discriminative = use_discriminative
        self.synthetic_noise_std = synthetic_noise_std
        self.discriminator_weight = discriminator_weight
        self.weight_decay = weight_decay
        self.model: Optional[Any] = None
        self.discriminator: Optional[Any] = None
        self.optimizer: Optional[Any] = None

    def _check_pytorch(self) -> None:
        """Check if PyTorch is available."""
        try:
            import torch
            self.torch = torch
        except ImportError:
            raise ImportError(
                "PyTorch not installed. Install with: pip install torch\n"
                "Or install ML dependencies: pip install -e '.[ml]'"
            )

    def _build_model(self) -> None:
        """Build AE and optional discriminative head."""
        import torch.nn as nn

        class AutoEncoder(nn.Module):
            def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int) -> None:
                super().__init__()
                encoder_layers = []
                prev_dim = input_dim
                for hidden_dim in hidden_dims:
                    encoder_layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                    ])
                    prev_dim = hidden_dim
                encoder_layers.append(nn.Linear(prev_dim, latent_dim))
                self.encoder = nn.Sequential(*encoder_layers)
                decoder_layers = []
                prev_dim = latent_dim
                for hidden_dim in reversed(hidden_dims):
                    decoder_layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                    ])
                    prev_dim = hidden_dim
                decoder_layers.append(nn.Linear(prev_dim, input_dim))
                self.decoder = nn.Sequential(*decoder_layers)

            def forward(self, x: Any) -> tuple[Any, Any]:
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded, encoded

        class DiscriminatorMLP(nn.Module):
            """(input, reconstruction) -> P(anomaly). Input dim = 2 * input_dim."""

            def __init__(self, joint_dim: int, hidden: int = 64) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(joint_dim, hidden),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden, 1),
                    nn.Sigmoid(),
                )

            def forward(self, x: Any) -> Any:
                return self.net(x).squeeze(-1)

        self.model = AutoEncoder(self.input_dim, self.hidden_dims, self.latent_dim).to(self.device)
        if self.use_discriminative:
            self.discriminator = DiscriminatorMLP(2 * self.input_dim, hidden=min(64, 2 * self.input_dim)).to(self.device)
            self.optimizer = self.torch.optim.Adam(
                list(self.model.parameters()) + list(self.discriminator.parameters()),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            self.discriminator = None
            self.optimizer = self.torch.optim.Adam(
                self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )

    def _prepare_data(self, data: pd.DataFrame | np.ndarray) -> Any:
        """Convert data to PyTorch tensor."""
        if isinstance(data, pd.DataFrame):
            # Extract numeric columns (exclude label, dataset_path, etc.)
            exclude = ["label", "dataset_path", "frame", "index", "x", "y"]
            # Use select_dtypes to get all numeric types (includes float32, float64, int32, int64, etc.)
            numeric_df = data.select_dtypes(include=[np.number])
            numeric_cols = [c for c in numeric_df.columns if c not in exclude]
            if not numeric_cols:
                # Fallback: use all columns except exclude
                numeric_cols = [c for c in data.columns if c not in exclude]
            array = data[numeric_cols].to_numpy(dtype=np.float32)
        else:
            array = np.asarray(data, dtype=np.float32)

        if array.size == 0:
            raise ValueError(f"Empty array after preparation. DataFrame columns: {data.columns.tolist() if isinstance(data, pd.DataFrame) else 'N/A'}")

        tensor = self.torch.from_numpy(array).to(self.device)
        return tensor

    def _generate_crack_like_anomaly(
        self,
        x_normal: Any,
        feature_names: list[str] | None = None,
    ) -> Any:
        """
        Generate crack-like synthetic anomaly from normal features.

        Strategy: simulate micro-trajectory changes, velocity/acceleration spikes,
        and shockwave patterns that occur when a crack forms during bending.
        These patterns are detectable in time-series analysis even if the crack
        itself is not visible in side-view camera images.

        Modifications:
        - Acceleration spikes (shockwave): increase acceleration_max, acceleration_std
        - Trajectory deviation: increase curvature_concentration, curvature_like_max
        - Strain concentration: increase strain_surrogate_max, strain_surrogate_std
        - Impact/shock: increase impact_surrogate_max
        - Velocity change: increase speed_std (micro-vibration)
        """
        x_anom = x_normal.clone()
        batch_size, n_features = x_normal.shape

        if feature_names is None:
            # Heuristic: try to identify feature columns by common patterns
            # This is approximate; ideally feature_names should be passed from prepare_training_data
            feature_names = [f"f{i}" for i in range(n_features)]

        # Find indices for acceleration, curvature, strain, impact, speed features
        # More flexible matching: any feature containing these keywords (not just max/std)
        accel_indices = [i for i, name in enumerate(feature_names) if "acceleration" in name.lower()]
        curv_indices = [i for i, name in enumerate(feature_names) if "curvature" in name.lower()]
        strain_indices = [i for i, name in enumerate(feature_names) if "strain" in name.lower()]
        impact_indices = [i for i, name in enumerate(feature_names) if "impact" in name.lower()]
        speed_indices = [i for i, name in enumerate(feature_names) if "speed" in name.lower()]
        
        # Prioritize max/std/concentration features if available, but use any match if not
        accel_priority = [i for i in accel_indices if any(kw in feature_names[i].lower() for kw in ["max", "std", "mean"])]
        if accel_priority:
            accel_indices = accel_priority
        curv_priority = [i for i in curv_indices if any(kw in feature_names[i].lower() for kw in ["max", "concentration", "mean"])]
        if curv_priority:
            curv_indices = curv_priority
        strain_priority = [i for i in strain_indices if any(kw in feature_names[i].lower() for kw in ["max", "std", "mean"])]
        if strain_priority:
            strain_indices = strain_priority
        impact_priority = [i for i in impact_indices if any(kw in feature_names[i].lower() for kw in ["max", "mean"])]
        if impact_priority:
            impact_indices = impact_priority
        speed_priority = [i for i in speed_indices if any(kw in feature_names[i].lower() for kw in ["std", "mean"])]
        if speed_priority:
            speed_indices = speed_priority

        # Apply crack-like modifications
        rng = self.torch.rand(batch_size, device=x_normal.device)
        anomaly_mask = rng > 0.5  # 50% of batch becomes anomaly

        for idx in accel_indices:
            if idx < n_features:
                # Shockwave: acceleration spike (1.2-1.8x increase)
                spike_factor = 1.2 + 0.6 * self.torch.rand(batch_size, device=x_normal.device)
                x_anom[:, idx] = self.torch.where(anomaly_mask, x_normal[:, idx] * spike_factor, x_anom[:, idx])

        for idx in curv_indices:
            if idx < n_features:
                # Trajectory deviation: curvature concentration (1.15-1.5x)
                curv_factor = 1.15 + 0.35 * self.torch.rand(batch_size, device=x_normal.device)
                x_anom[:, idx] = self.torch.where(anomaly_mask, x_normal[:, idx] * curv_factor, x_anom[:, idx])

        for idx in strain_indices:
            if idx < n_features:
                # Strain concentration at crack location (1.1-1.4x)
                strain_factor = 1.1 + 0.3 * self.torch.rand(batch_size, device=x_normal.device)
                x_anom[:, idx] = self.torch.where(anomaly_mask, x_normal[:, idx] * strain_factor, x_anom[:, idx])

        for idx in impact_indices:
            if idx < n_features:
                # Impact/shock from crack formation (1.3-2.0x)
                impact_factor = 1.3 + 0.7 * self.torch.rand(batch_size, device=x_normal.device)
                x_anom[:, idx] = self.torch.where(anomaly_mask, x_normal[:, idx] * impact_factor, x_anom[:, idx])

        for idx in speed_indices:
            if idx < n_features:
                # Micro-vibration: speed variation (1.1-1.3x std)
                vib_factor = 1.1 + 0.2 * self.torch.rand(batch_size, device=x_normal.device)
                x_anom[:, idx] = self.torch.where(anomaly_mask, x_normal[:, idx] * vib_factor, x_anom[:, idx])

        # Fallback: if no feature names match, apply physics-informed transformations
        # Simulate crack effects: increase some features (dynamics-related) more than others
        if not (accel_indices or curv_indices or strain_indices or impact_indices or speed_indices):
            # Apply crack-like pattern: some features increase more (simulating shock/strain)
            # Assume first half might be base features, second half might be derived stats
            for idx in range(n_features):
                if anomaly_mask.any():
                    # Varying impact: some features increase 1.1-1.4x (strain-like), others 1.05-1.2x
                    if idx < n_features // 2:
                        factor = 1.1 + 0.3 * self.torch.rand(batch_size, device=x_normal.device)
                    else:
                        factor = 1.05 + 0.15 * self.torch.rand(batch_size, device=x_normal.device)
                    x_anom[:, idx] = self.torch.where(anomaly_mask, x_normal[:, idx] * factor, x_anom[:, idx])

        return x_anom

    def fit(
        self,
        normal_data: pd.DataFrame | np.ndarray,
        epochs: int = 100,
        feature_names: list[str] | None = None,
    ) -> None:
        """
        Train on normal data only. If use_discriminative, add crack-like synthetic anomalies.

        Args:
            normal_data: Normal training data
            epochs: Training epochs
            feature_names: Optional list of feature column names (for crack-like anomaly generation)
        """
        if self.model is None:
            self._build_model()

        # Extract feature names if DataFrame
        if isinstance(normal_data, pd.DataFrame):
            exclude = ["label", "dataset_path", "frame", "index", "x", "y"]
            if feature_names is None:
                feature_names = [c for c in normal_data.columns if c not in exclude and normal_data[c].dtype in [np.float64, np.int64, float, int]]

        train_data = self._prepare_data(normal_data)
        self.model.train()
        if self.discriminator is not None:
            self.discriminator.train()
        criterion_mse = self.torch.nn.MSELoss()
        criterion_bce = self.torch.nn.BCELoss()

        n_samples = len(train_data)
        n_batches = max(1, (n_samples + self.batch_size - 1) // self.batch_size)

        for epoch in range(epochs):
            total_loss = 0.0
            indices = self.torch.randperm(n_samples)
            for i in range(n_batches):
                start_idx = i * self.batch_size
                end_idx = min(start_idx + self.batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                x_clean = train_data[batch_indices]

                batch_size_cur = x_clean.size(0)
                if self.use_discriminative and self.discriminator is not None:
                    # Generate crack-like synthetic anomalies (physics-informed) instead of pure noise
                    x_input = self._generate_crack_like_anomaly(x_clean, feature_names)
                    # Label: 1 if modified (anomaly), 0 if unchanged (normal)
                    # Check per-sample if any feature was modified
                    diff = self.torch.abs(x_input - x_clean)
                    is_anomaly = (diff > 1e-6).any(dim=1)  # Per-sample: True if any feature changed
                    disc_labels = is_anomaly.float()
                else:
                    x_input = x_clean
                    disc_labels = self.torch.zeros(batch_size_cur, device=x_clean.device)

                self.optimizer.zero_grad()
                reconstructed, _ = self.model(x_input)
                # DRAEM: Autoencoder reconstructs to CLEAN normal data, not the input
                # This ensures anomalies have high reconstruction error
                l_recon = criterion_mse(reconstructed, x_clean)
                loss = l_recon

                if self.use_discriminative and self.discriminator is not None:
                    # Discriminator learns to distinguish (input, reconstruction) pairs
                    # For anomalies: (anomaly_input, normal_reconstruction) -> should predict anomaly
                    joint = self.torch.cat([x_input, reconstructed], dim=1)
                    disc_out = self.discriminator(joint)
                    l_disc = criterion_bce(disc_out, disc_labels)
                    loss = loss + self.discriminator_weight * l_disc

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / n_batches
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

        self.is_trained = True

    def predict(self, data: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores. 
        
        DRAEM strategy: Uses reconstruction error as primary signal.
        - Autoencoder is trained to reconstruct normal data
        - Anomalies (crack) will have high reconstruction error
        - Discriminator provides additional signal: P(anomaly) from (input, reconstruction) pair
        - Combined score: reconstruction_error * (1 + discriminator_weight * P(anomaly))
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")

        test_data = self._prepare_data(data)
        self.model.eval()
        if self.discriminator is not None:
            self.discriminator.eval()
        criterion = self.torch.nn.MSELoss(reduction="none")

        scores = []
        with self.torch.no_grad():
            n_samples = len(test_data)
            for i in range(0, n_samples, self.batch_size):
                batch = test_data[i:i + self.batch_size]
                reconstructed, _ = self.model(batch)
                
                # Reconstruction error: how well the model reconstructs the input
                # For normal data: low error (model learned normal patterns)
                # For anomalies: high error (model tries to reconstruct as normal, but input is anomalous)
                recon_error = criterion(reconstructed, batch).mean(dim=1)
                
                if self.discriminator is not None:
                    # Discriminator outputs P(anomaly) from (input, reconstruction) pair
                    joint = self.torch.cat([batch, reconstructed], dim=1)
                    disc_out = self.discriminator(joint)  # P(anomaly) in [0, 1]
                    # Combine: reconstruction error weighted by discriminator confidence
                    # Higher P(anomaly) amplifies the reconstruction error
                    batch_scores = recon_error * (1.0 + self.discriminator_weight * disc_out)
                    batch_scores = batch_scores.cpu().numpy()
                else:
                    batch_scores = recon_error.cpu().numpy()
                scores.extend(batch_scores)

        return np.array(scores)

    def predict_binary(self, data: pd.DataFrame | np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Predict binary labels (0=normal, 1=anomaly)."""
        scores = self.predict(data)
        if threshold is None:
            threshold = self.reconstruction_error_threshold
        if threshold is None:
            raise ValueError("Threshold not set. Call set_threshold_from_normal() or provide threshold.")
        return (scores > threshold).astype(int)

    def save(self, path: Path) -> None:
        """Save model weights and config."""
        if not self.is_trained:
            raise ValueError("Model not trained. Cannot save untrained model.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "latent_dim": self.latent_dim,
            "learning_rate": self.learning_rate,
            "reconstruction_error_threshold": self.reconstruction_error_threshold,
            "is_trained": self.is_trained,
            "use_discriminative": self.use_discriminative,
            "synthetic_noise_std": self.synthetic_noise_std,
            "discriminator_weight": self.discriminator_weight,
        }
        if self.discriminator is not None:
            save_dict["discriminator_state_dict"] = self.discriminator.state_dict()
        self.torch.save(save_dict, path)

    def load(self, path: Path) -> None:
        """Load model weights and config."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        checkpoint = self.torch.load(path, map_location=self.device)
        self.input_dim = checkpoint["input_dim"]
        self.hidden_dims = checkpoint["hidden_dims"]
        self.latent_dim = checkpoint["latent_dim"]
        self.learning_rate = checkpoint["learning_rate"]
        self.reconstruction_error_threshold = checkpoint.get("reconstruction_error_threshold")
        self.is_trained = checkpoint.get("is_trained", False)
        self.use_discriminative = checkpoint.get("use_discriminative", False)
        self.synthetic_noise_std = checkpoint.get("synthetic_noise_std", 0.3)
        self.discriminator_weight = checkpoint.get("discriminator_weight", 0.5)

        self._build_model()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if self.discriminator is not None and "discriminator_state_dict" in checkpoint:
            self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])


# Default implementation
DREAMAnomalyDetector = DREAMPyTorch
