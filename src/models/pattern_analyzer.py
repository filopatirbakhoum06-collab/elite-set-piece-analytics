"""
Pattern Analyzer for Elite Set-Piece Analytics.
محلل الأنماط

This module implements pattern recognition and clustering for
identifying and analyzing set-piece tactical patterns.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import logging
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

from ..utils.config import Config


class PatternAnalyzer:
    """
    Analyzer for set-piece tactical patterns.
    محلل أنماط الكرات الثابتة التكتيكية
    
    This class provides functionality to:
    - Cluster set-pieces by tactical similarity
    - Find similar set-pieces to a given example
    - Identify common patterns and templates
    
    Example:
        >>> analyzer = PatternAnalyzer()
        >>> clusters = analyzer.cluster_set_pieces(features_df)
        >>> similar = analyzer.find_similar(query_features, features_df, n=5)
    """
    
    def __init__(
        self,
        n_clusters: int = 10,
        random_state: int = Config.RANDOM_SEED
    ):
        """
        Initialize the pattern analyzer.
        
        Args:
            n_clusters: Number of clusters for K-means
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.config = Config()
        
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% variance
        self.kmeans = KMeans(
            n_clusters=n_clusters, 
            random_state=random_state,
            n_init=10
        )
        
        self.is_fitted = False
        self.feature_names: List[str] = []
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def fit(
        self,
        X: pd.DataFrame,
        use_pca: bool = True
    ) -> 'PatternAnalyzer':
        """
        Fit the pattern analyzer on set-piece features.
        
        Args:
            X: Feature DataFrame
            use_pca: Whether to apply PCA dimensionality reduction
            
        Returns:
            self
        """
        self.logger.info("Fitting PatternAnalyzer...")
        
        self.feature_names = list(X.columns)
        X_array = X.values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_array)
        
        # Apply PCA if requested and data has enough features
        self.use_pca = use_pca and X_scaled.shape[1] > 10
        if self.use_pca:
            self.X_transformed = self.pca.fit_transform(X_scaled)
            self.logger.info(f"PCA reduced dimensions from {X_scaled.shape[1]} to {self.X_transformed.shape[1]}")
        else:
            self.X_transformed = X_scaled
            self.pca = None  # Disable PCA if not used
        
        # Fit K-means
        self.kmeans.fit(self.X_transformed)
        
        self.is_fitted = True
        self.logger.info(f"PatternAnalyzer fitted with {self.n_clusters} clusters")
        
        return self
    
    def cluster_set_pieces(
        self,
        X: pd.DataFrame
    ) -> np.ndarray:
        """
        Assign set-pieces to clusters.
        تصنيف الكرات الثابتة إلى مجموعات
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of cluster assignments
        """
        self._check_fitted()
        
        X_array = X.values
        X_scaled = self.scaler.transform(X_array)
        
        if self.pca is not None and hasattr(self.pca, 'components_'):
            X_transformed = self.pca.transform(X_scaled)
        else:
            X_transformed = X_scaled
        
        return self.kmeans.predict(X_transformed)
    
    def find_similar(
        self,
        query: pd.DataFrame,
        database: pd.DataFrame,
        n: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Find most similar set-pieces to a query.
        البحث عن كرات ثابتة مشابهة
        
        Args:
            query: Query features (single row DataFrame)
            database: Database of set-piece features
            n: Number of similar examples to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        self._check_fitted()
        
        # Transform query
        query_array = query.values.reshape(1, -1)
        query_scaled = self.scaler.transform(query_array)
        
        if self.pca is not None and hasattr(self.pca, 'components_'):
            query_transformed = self.pca.transform(query_scaled)
        else:
            query_transformed = query_scaled
        
        # Transform database
        db_array = database.values
        db_scaled = self.scaler.transform(db_array)
        
        if self.pca is not None and hasattr(self.pca, 'components_'):
            db_transformed = self.pca.transform(db_scaled)
        else:
            db_transformed = db_scaled
        
        # Calculate distances
        distances = cdist(query_transformed, db_transformed, metric='euclidean')[0]
        
        # Get top-n closest
        top_indices = np.argsort(distances)[:n]
        
        return [(int(idx), float(distances[idx])) for idx in top_indices]
    
    def get_cluster_centers(self) -> pd.DataFrame:
        """
        Get cluster center coordinates.
        
        Returns:
            DataFrame with cluster centers
        """
        self._check_fitted()
        
        centers = self.kmeans.cluster_centers_
        
        # Inverse transform if PCA was used
        if hasattr(self, 'pca') and self.pca is not None:
            centers = self.pca.inverse_transform(centers)
        
        centers = self.scaler.inverse_transform(centers)
        
        return pd.DataFrame(centers, columns=self.feature_names)
    
    def get_cluster_statistics(
        self,
        X: pd.DataFrame,
        labels: Optional[np.ndarray] = None
    ) -> Dict[int, Dict[str, Any]]:
        """
        Calculate statistics for each cluster.
        
        Args:
            X: Feature DataFrame
            labels: Cluster labels (computed if not provided)
            
        Returns:
            Dictionary of statistics per cluster
        """
        if labels is None:
            labels = self.cluster_set_pieces(X)
        
        stats = {}
        for cluster_id in range(self.n_clusters):
            mask = labels == cluster_id
            cluster_data = X[mask]
            
            if len(cluster_data) == 0:
                continue
            
            stats[cluster_id] = {
                'count': int(mask.sum()),
                'percentage': float(mask.sum() / len(X) * 100),
                'mean_features': cluster_data.mean().to_dict(),
                'std_features': cluster_data.std().to_dict(),
            }
        
        return stats
    
    def identify_pattern_type(
        self,
        cluster_id: int,
        cluster_stats: Dict[int, Dict[str, Any]]
    ) -> str:
        """
        Identify the type of pattern for a cluster.
        
        Args:
            cluster_id: Cluster identifier
            cluster_stats: Statistics from get_cluster_statistics()
            
        Returns:
            Pattern type description
        """
        if cluster_id not in cluster_stats:
            return "Unknown"
        
        stats = cluster_stats[cluster_id]
        means = stats.get('mean_features', {})
        
        # Simple pattern classification based on features
        # This would be customized based on actual features
        patterns = []
        
        if means.get('distance_to_goal', 50) < 20:
            patterns.append("Near-post")
        elif means.get('distance_to_goal', 50) > 40:
            patterns.append("Long-ball")
        
        if means.get('player_density_5m', 0) > 3:
            patterns.append("Crowded-box")
        else:
            patterns.append("Open-play")
        
        return " ".join(patterns) if patterns else "Standard"
    
    def _check_fitted(self) -> None:
        """Check if analyzer is fitted."""
        if not self.is_fitted:
            raise RuntimeError("PatternAnalyzer has not been fitted. Call fit() first.")


def extract_pattern_features(
    events: pd.DataFrame,
    positions: pd.DataFrame
) -> pd.DataFrame:
    """
    Extract features for pattern analysis.
    
    Args:
        events: Set-piece event data
        positions: Player position data
        
    Returns:
        DataFrame with pattern features
    """
    features = []
    
    # This would extract meaningful pattern features
    # Placeholder implementation
    for idx in range(len(events)):
        feat = {
            'n_attackers': 5,  # placeholder
            'n_defenders': 5,  # placeholder
            'compactness': 0.5,  # placeholder
            'spread': 20.0,  # placeholder
        }
        features.append(feat)
    
    return pd.DataFrame(features)
