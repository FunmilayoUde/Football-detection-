import random
import numpy as np
import supervision as sv
import torch
import cv2
from transformers import AutoProcessor, SiglipVisionModel
from more_itertools import chunked
import umap
from sklearn.cluster import KMeans


RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)
torch.set_num_threads(4)  


class TeamClassifier:
    """
    Team classification via SigLIP embeddings -> UMAP -> KMeans.
    Now CPU-friendly, deterministic, and with stable cluster->team mapping.

    Improvements included:
      - Force CPU usage (no CUDA probes)
      - Fixed random seeds (repeatable results)
      - Smaller batch size for CPU
      - Stable mapping of clusters to Team 0/1 using jersey hue signatures
      - Safer GK team resolution when one team temporarily missing
    """

    def __init__(self, device: str = "cpu"):
        self.device = "cpu"
        self.model_path = "google/siglip-base-patch16-224"
        self.model = SiglipVisionModel.from_pretrained(self.model_path).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.reducer = umap.UMAP(
            n_components=3,
            n_neighbors=15,
            min_dist=0.1,
            random_state=RNG_SEED
        )
        self.clustering_model = KMeans(
            n_clusters=2,
            n_init=10,
            random_state=RNG_SEED
        )

        self.is_fitted = False
        self.team_centroids = None
        self.cluster_hues = None
        self.cluster_remap = None  

    def _extract_embeddings(self, crops, batch_size: int = 8) -> np.ndarray:
        """Convert image crops (BGR np arrays) into SigLIP embeddings (N, D)."""
        if len(crops) == 0:
            return np.empty((0, 512))

        crops_pil = [sv.cv2_to_pillow(c) for c in crops]
        out = []
        with torch.no_grad():
            for batch in chunked(crops_pil, batch_size):
                inputs = self.processor(images=list(batch), return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                emb = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                out.append(emb)
        return np.concatenate(out) if out else np.empty((0, 512))

    def _hue_signature(self, crops) -> float:
        """
        Crude jersey hue signature per cluster:
        mean Hue over pixels not too dark (V > 40).
        Used only to stabilize which cluster -> Team 0 or 1.
        """
        hues = []
        for c in crops:
            hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
            mask = hsv[..., 2] > 40 
            if mask.any():
                hues.append(float(hsv[..., 0][mask].mean()))
        return float(np.mean(hues)) if len(hues) else 0.0

    def fit(self, player_crops):
        """
        One-time calibration on a batch of player crops.
        Learns UMAP + KMeans + stable cluster->team mapping (via jersey hue).
        """
        print("🟡 Calibrating TeamClassifier on initial frames...")
        embeddings = self._extract_embeddings(player_crops)
        if embeddings.shape[0] == 0:
            print("⚠️ No crops available for fitting.")
            return

        projections = self.reducer.fit_transform(embeddings)
        clusters = self.clustering_model.fit_predict(projections)

        centroids = []
        for c in range(2):
            centroids.append(projections[clusters == c].mean(axis=0))
        self.team_centroids = np.array(centroids)

        cluster_hues = []
        for c in range(2):
            idx = np.where(clusters == c)[0]
            cluster_crops = [player_crops[i] for i in idx]
            cluster_hues.append(self._hue_signature(cluster_crops))
        self.cluster_hues = cluster_hues

        order = np.argsort(self.cluster_hues)  
        self.cluster_remap = {int(order[0]): 0, int(order[1]): 1}

        self.is_fitted = True
        print("TeamClassifier calibrated successfully.")

    def predict(self, player_crops):
        """
        Predict stable team IDs (0/1). Auto-fit on-demand if not calibrated.
        """
        embeddings = self._extract_embeddings(player_crops)
        if embeddings.shape[0] == 0:
            return np.array([])

        if not self.is_fitted:
            self.fit(player_crops)

        projections = self.reducer.transform(embeddings)
        raw_clusters = self.clustering_model.predict(projections)

        if self.cluster_remap is not None:
            mapped = np.array([self.cluster_remap[int(c)] for c in raw_clusters], dtype=int)
            return mapped
        else:
            return raw_clusters.astype(int)

    def resolve_goalkeepers_team_id(self, players: sv.Detections, goalkeepers: sv.Detections) -> np.ndarray:
        """
        Assign goalkeeper team by nearest team centroid of players (image bottom-center).
        Safer when a team is momentarily missing.
        """
        if len(goalkeepers) == 0 or len(players) == 0:
            return np.array([])

        players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        gk_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

        mask0 = (players.class_id == 0)
        mask1 = (players.class_id == 1)

        if (not mask0.any()) or (not mask1.any()):
            majority = 0 if mask0.sum() >= mask1.sum() else 1
            return np.full(len(gk_xy), majority, dtype=int)

        team_0_centroid = players_xy[mask0].mean(axis=0)
        team_1_centroid = players_xy[mask1].mean(axis=0)

        out = []
        for p in gk_xy:
            d0 = np.linalg.norm(p - team_0_centroid)
            d1 = np.linalg.norm(p - team_1_centroid)
            out.append(0 if d0 < d1 else 1)
        return np.array(out, dtype=int)

