import numpy as np
import supervision as sv
import torch
from transformers import AutoProcessor, SiglipVisionModel
from more_itertools import chunked
import umap
from sklearn.cluster import KMeans


class TeamClassifier:
    """
    Handles player team classification using SigLIP embeddings + UMAP + KMeans.
    Includes a calibration phase for stable team colors.
    """

    def __init__(self, device="cuda"):
        # === Load SigLIP model and processor ===
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_path = "google/siglip-base-patch16-224"
        self.model = SiglipVisionModel.from_pretrained(self.model_path).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_path)

        # === UMAP + KMeans ===
        self.reducer = umap.UMAP(n_components=3)
        self.clustering_model = KMeans(n_clusters=2)

        # === Calibration state ===
        self.is_fitted = False
        self.team_centroids = None

    def _extract_embeddings(self, crops, batch_size=32):
        """Convert image crops into embeddings using SigLIP."""
        if len(crops) == 0:
            return np.empty((0, 512))

        crops = [sv.cv2_to_pillow(crop) for crop in crops]
        data = []

        with torch.no_grad():
            for batch in chunked(crops, batch_size):
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)

        return np.concatenate(data)

    def fit(self, player_crops):
        """
        Perform one-time fitting using a batch of player crops.
        Should be called once at the start of the video.
        """
        print("🟡 Calibrating TeamClassifier on initial frames...")
        embeddings = self._extract_embeddings(player_crops)
        if embeddings.shape[0] == 0:
            print("⚠️ No crops available for fitting.")
            return

        projections = self.reducer.fit_transform(embeddings)
        clusters = self.clustering_model.fit_predict(projections)

        # Compute centroids for later use
        centroids = []
        for c in range(2):
            centroids.append(projections[clusters == c].mean(axis=0))
        self.team_centroids = np.array(centroids)

        self.is_fitted = True
        print("✅ TeamClassifier calibrated successfully.")

    def predict(self, player_crops):
        """
        Predict team IDs using fitted models.
        If not yet fitted, auto-fit on current crops.
        """
        embeddings = self._extract_embeddings(player_crops)
        if embeddings.shape[0] == 0:
            return np.array([])

        if not self.is_fitted:
            self.fit(player_crops)

        projections = self.reducer.transform(embeddings)
        clusters = self.clustering_model.predict(projections)
        return clusters

    def resolve_goalkeepers_team_id(self, players: sv.Detections, goalkeepers: sv.Detections) -> np.ndarray:
        """
        Resolves which goalkeeper belongs to which team based on proximity to cluster centroids.
        """
        if len(goalkeepers) == 0 or len(players) == 0:
            return np.array([])

        goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

        team_0_centroid = players_xy[players.class_id == 0].mean(axis=0)
        team_1_centroid = players_xy[players.class_id == 1].mean(axis=0)

        goalkeepers_team_id = []
        for goalkeeper_xy in goalkeepers_xy:
            dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
            dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
            goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)

        return np.array(goalkeepers_team_id)
