"""
Baseline: Ratio-Based Player Similarity

This module contains the original ratio-based similarity method from the
MLSE Player Similarity project, refactored into a clean API for baseline
comparison against the new GNN-based embeddings.

Key features:
1. Ratio comparison: sim(a,b) = 1 - |a-b| / max(a,b)
2. Sigmoid neutralization: handles rare actions without inflating scores
3. Activity thresholds: both players must have meaningful activity

Reference: Original implementation in Player_Similarity_Model_Armen.ipynb
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from collections import defaultdict


@dataclass
class RatioSimilarityConfig:
    """Configuration for ratio-based similarity."""
    
    # Grid size for spatial heatmaps
    grid_size: Tuple[int, int] = (12, 8)
    
    # Ratio comparison parameters
    ratio_epsilon: float = 0.01
    sigmoid_midpoint: float = 0.1
    sigmoid_steepness: float = 10.0
    
    # Feature weights
    feature_weights: Dict[str, float] = field(default_factory=lambda: {
        "spatial": 0.20,
        "action_chains": 0.15,
        "value_added": 0.20,
        "pressing": 0.15,
        "passing": 0.15,
        "receiving": 0.10,
        "transition": 0.05,
    })
    
    # Embedding parameters
    embedding_dim: int = 32
    n_roles: int = 10


class RatioBasedSimilarity:
    """
    Custom similarity metric using ratio-based comparison.
    
    Key innovations:
    1. Ratio comparison: sim(a,b) = 1 - |a-b| / max(a,b)
    2. Sigmoid neutralization: handles rare actions without inflating scores
    3. Activity thresholds: both players must have meaningful activity
    
    This addresses the "two players who rarely do X" problem:
    - If both players have 0.01 aerial win rate, raw ratio gives high similarity
    - Sigmoid neutralization downweights this contribution based on activity level
    """
    
    def __init__(self, config: Optional[RatioSimilarityConfig] = None):
        self.config = config or RatioSimilarityConfig()
        self.epsilon = self.config.ratio_epsilon
        self.sigmoid_midpoint = self.config.sigmoid_midpoint
        self.sigmoid_steepness = self.config.sigmoid_steepness
    
    def sigmoid_weight(self, activity_level: float) -> float:
        """
        Compute sigmoid weight based on activity level.
        
        Low activity → weight near 0 (don't trust comparison)
        High activity → weight near 1 (trust comparison)
        """
        return 1 / (1 + np.exp(-self.sigmoid_steepness * 
                               (activity_level - self.sigmoid_midpoint)))
    
    def ratio_similarity(
        self, 
        a: float, 
        b: float,
        apply_sigmoid: bool = True,
    ) -> float:
        """
        Compute ratio-based similarity between two values.
        
        Formula: sim = 1 - |a - b| / max(a, b, epsilon)
        
        With sigmoid neutralization:
        final_sim = base_sim * sigmoid_weight(min(a, b))
        """
        max_val = max(a, b, self.epsilon)
        base_sim = 1 - abs(a - b) / max_val
        
        if apply_sigmoid:
            min_activity = min(a, b)
            weight = self.sigmoid_weight(min_activity)
            return base_sim * weight
        
        return base_sim
    
    def vector_similarity(
        self,
        vec_a: np.ndarray,
        vec_b: np.ndarray,
        weights: Optional[np.ndarray] = None,
        apply_sigmoid: bool = True,
    ) -> float:
        """Compute ratio-based similarity between two feature vectors."""
        if len(vec_a) != len(vec_b):
            raise ValueError("Vectors must have same length")
            
        if weights is None:
            weights = np.ones(len(vec_a))
        
        similarities = []
        total_weight = 0
        
        for i, (a, b) in enumerate(zip(vec_a, vec_b)):
            sim = self.ratio_similarity(a, b, apply_sigmoid)
            similarities.append(sim * weights[i])
            
            if apply_sigmoid:
                activity = min(a, b)
                total_weight += weights[i] * self.sigmoid_weight(activity)
            else:
                total_weight += weights[i]
        
        if total_weight < self.epsilon:
            return 0.0
            
        return sum(similarities) / total_weight
    
    def heatmap_similarity(
        self,
        heat_a: np.ndarray,
        heat_b: np.ndarray,
    ) -> float:
        """
        Compute similarity between two spatial heatmaps.
        
        Uses:
        1. Histogram intersection (overlapping areas)
        2. Location-weighted ratio comparison
        """
        # Normalize heatmaps
        heat_a = heat_a / (heat_a.sum() + self.epsilon)
        heat_b = heat_b / (heat_b.sum() + self.epsilon)
        
        # Histogram intersection
        intersection = np.minimum(heat_a, heat_b).sum()
        
        # Cell-by-cell ratio similarity
        flat_a = heat_a.flatten()
        flat_b = heat_b.flatten()
        
        cell_sims = []
        for a, b in zip(flat_a, flat_b):
            if a > 0.001 or b > 0.001:
                cell_sim = self.ratio_similarity(a, b, apply_sigmoid=True)
                cell_sims.append(cell_sim)
        
        cell_avg = np.mean(cell_sims) if cell_sims else 0.0
        
        return 0.6 * intersection + 0.4 * cell_avg
    
    def dict_similarity(
        self,
        dict_a: Dict[str, float],
        dict_b: Dict[str, float],
    ) -> float:
        """Compute similarity between two dictionaries of features."""
        all_keys = set(dict_a.keys()) | set(dict_b.keys())
        
        if not all_keys:
            return 1.0
        
        similarities = []
        for key in all_keys:
            a = dict_a.get(key, 0.0)
            b = dict_b.get(key, 0.0)
            sim = self.ratio_similarity(a, b, apply_sigmoid=True)
            similarities.append(sim)
        
        return np.mean(similarities)
    
    def compute_player_similarity(
        self,
        profile_a: Dict[str, Any],
        profile_b: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Compute comprehensive similarity between two player profiles.
        
        Returns similarity scores for each feature block and overall score.
        """
        weights = self.config.feature_weights
        scores = {}
        
        # 1. Spatial similarity
        spatial_sims = []
        for hmap_type in ['all', 'pass', 'receive', 'pressure']:
            if (hmap_type in profile_a.get('spatial', {}) and 
                hmap_type in profile_b.get('spatial', {})):
                sim = self.heatmap_similarity(
                    profile_a['spatial'][hmap_type],
                    profile_b['spatial'][hmap_type]
                )
                spatial_sims.append(sim)
        scores['spatial'] = np.mean(spatial_sims) if spatial_sims else 0.0
        
        # 2. Action chains
        if 'chains' in profile_a and 'chains' in profile_b:
            scores['action_chains'] = self.vector_similarity(
                profile_a['chains'], profile_b['chains']
            )
        else:
            scores['action_chains'] = 0.0
        
        # 3. Value added
        if 'value_added' in profile_a and 'value_added' in profile_b:
            scores['value_added'] = self.dict_similarity(
                profile_a['value_added'], profile_b['value_added']
            )
        else:
            scores['value_added'] = 0.0
        
        # 4. Pressing
        if 'pressing' in profile_a and 'pressing' in profile_b:
            scores['pressing'] = self.dict_similarity(
                profile_a['pressing'], profile_b['pressing']
            )
        else:
            scores['pressing'] = 0.0
        
        # 5. Passing
        if 'passing' in profile_a and 'passing' in profile_b:
            scores['passing'] = self.dict_similarity(
                profile_a['passing'], profile_b['passing']
            )
        else:
            scores['passing'] = 0.0
        
        # 6. Receiving
        if 'receiving' in profile_a and 'receiving' in profile_b:
            scores['receiving'] = self.dict_similarity(
                profile_a['receiving'], profile_b['receiving']
            )
        else:
            scores['receiving'] = 0.0
        
        # 7. Transition
        if 'transition' in profile_a and 'transition' in profile_b:
            scores['transition'] = self.dict_similarity(
                profile_a['transition'], profile_b['transition']
            )
        else:
            scores['transition'] = 0.0
        
        # Overall weighted score
        overall = sum(
            scores.get(feat, 0) * w
            for feat, w in weights.items()
        )
        scores['overall'] = overall
        
        return scores


class RatioSimilarityBaseline:
    """
    Complete baseline model with fit/topk interface.
    
    Matches the API expected by the evaluation framework.
    """
    
    def __init__(self, config: Optional[RatioSimilarityConfig] = None):
        self.config = config or RatioSimilarityConfig()
        self.similarity_calc = RatioBasedSimilarity(self.config)
        
        # Storage
        self.player_profiles: Dict[str, Dict] = {}
        self.player_ids: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.player_roles: Dict[str, Dict] = {}
        
        # Embedding model
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.config.embedding_dim)
    
    def fit(
        self,
        player_profiles: Dict[str, Dict[str, Any]],
        player_metadata: Optional[Dict[str, Dict]] = None,
    ):
        """
        Fit the baseline model on player profiles.
        
        Args:
            player_profiles: Dict mapping player_id -> behavioral profile
            player_metadata: Optional metadata for each player
        """
        self.player_profiles = player_profiles
        self.player_ids = list(player_profiles.keys())
        
        # Convert profiles to embedding vectors
        self.embeddings = self._profiles_to_embeddings(player_profiles)
        
        print(f"RatioSimilarityBaseline: Fitted on {len(self.player_ids)} players")
        print(f"  Embedding dimension: {self.embeddings.shape[1]}")
    
    def _profiles_to_embeddings(
        self,
        profiles: Dict[str, Dict],
    ) -> np.ndarray:
        """Convert player profiles to embedding vectors."""
        feature_vectors = []
        
        for pid in self.player_ids:
            profile = profiles[pid]
            vec = self._flatten_profile(profile)
            feature_vectors.append(vec)
        
        features = np.array(feature_vectors)
        features = np.nan_to_num(features, nan=0.0)
        
        # Scale and reduce
        features_scaled = self.scaler.fit_transform(features)
        
        n_components = min(
            self.config.embedding_dim,
            features_scaled.shape[0] - 1,
            features_scaled.shape[1]
        )
        self.pca = PCA(n_components=n_components)
        embeddings = self.pca.fit_transform(features_scaled)
        
        return embeddings
    
    def _flatten_profile(self, profile: Dict) -> List[float]:
        """Flatten a profile dict to a feature vector."""
        vec = []
        
        # Spatial features (flatten heatmaps)
        for hmap_type in ['all', 'pass', 'receive', 'pressure']:
            if hmap_type in profile.get('spatial', {}):
                vec.extend(profile['spatial'][hmap_type].flatten().tolist())
        
        # Chain features
        if 'chains' in profile:
            vec.extend(profile['chains'].tolist() if hasattr(profile['chains'], 'tolist') else list(profile['chains']))
        
        # Dict features
        for feat_name in ['value_added', 'pressing', 'passing', 'receiving', 'transition']:
            if feat_name in profile:
                vec.extend(list(profile[feat_name].values()))
        
        return vec
    
    def topk(
        self,
        query_player: str,
        k: int = 10,
        same_role_only: bool = False,
        exclude_players: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Find top-k most similar players.
        
        Args:
            query_player: Query player ID
            k: Number of results
            same_role_only: Only return same-role players
            exclude_players: Players to exclude from results
            
        Returns:
            List of dicts with player info and similarity scores
        """
        if query_player not in self.player_profiles:
            raise ValueError(f"Player {query_player} not found")
        
        query_profile = self.player_profiles[query_player]
        query_role = self.player_roles.get(query_player, {}).get('primary_role')
        
        exclude_set = set(exclude_players or [])
        exclude_set.add(query_player)
        
        results = []
        
        for other_player in self.player_ids:
            if other_player in exclude_set:
                continue
            
            other_profile = self.player_profiles[other_player]
            other_role = self.player_roles.get(other_player, {}).get('primary_role')
            
            if same_role_only and other_role != query_role:
                continue
            
            # Compute similarity
            scores = self.similarity_calc.compute_player_similarity(
                query_profile, other_profile
            )
            
            results.append({
                'player_id': other_player,
                'similarity': scores['overall'],
                'role': other_role,
                'role_match': other_role == query_role,
                'detailed_scores': scores,
            })
        
        # Sort by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return results[:k]
    
    def get_embedding(self, player_id: str) -> Optional[np.ndarray]:
        """Get embedding for a player."""
        if player_id not in self.player_ids:
            return None
        
        idx = self.player_ids.index(player_id)
        return self.embeddings[idx]
    
    def compute_similarity_matrix(
        self,
        player_subset: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Compute full similarity matrix."""
        if player_subset is None:
            player_subset = self.player_ids[:50]
        
        n = len(player_subset)
        sim_matrix = np.zeros((n, n))
        
        for i, p1 in enumerate(player_subset):
            for j, p2 in enumerate(player_subset):
                if i == j:
                    sim_matrix[i, j] = 1.0
                elif j > i:
                    scores = self.similarity_calc.compute_player_similarity(
                        self.player_profiles[p1],
                        self.player_profiles[p2]
                    )
                    sim_matrix[i, j] = scores['overall']
                    sim_matrix[j, i] = scores['overall']
        
        return pd.DataFrame(sim_matrix, index=player_subset, columns=player_subset)
    
    def save(self, path: str):
        """Save model to file."""
        import pickle
        
        state = {
            'config': self.config,
            'player_ids': self.player_ids,
            'player_profiles': self.player_profiles,
            'embeddings': self.embeddings,
            'player_roles': self.player_roles,
            'scaler': self.scaler,
            'pca': self.pca,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    @classmethod
    def load(cls, path: str) -> "RatioSimilarityBaseline":
        """Load model from file."""
        import pickle
        
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        model = cls(config=state['config'])
        model.player_ids = state['player_ids']
        model.player_profiles = state['player_profiles']
        model.embeddings = state['embeddings']
        model.player_roles = state['player_roles']
        model.scaler = state['scaler']
        model.pca = state['pca']
        
        return model


class RoleDiscovery:
    """
    Discover player roles using clustering.
    
    Uses Gaussian Mixture Model for soft clustering, allowing
    players to have probabilities across multiple roles.
    """
    
    # Predefined role archetypes with spatial centers
    ROLE_ARCHETYPES = {
        "goalkeeper": {"spatial_center": (6, 40), "description": "Shot-stopper and sweeper"},
        "ball_playing_cb": {"spatial_center": (20, 40), "description": "Build-up from the back"},
        "stopper_cb": {"spatial_center": (18, 40), "description": "Aggressive defending CB"},
        "fullback_traditional": {"spatial_center": (35, 10), "description": "Defensive wide player"},
        "fullback_attacking": {"spatial_center": (55, 10), "description": "Overlapping wide player"},
        "inverted_fullback": {"spatial_center": (45, 30), "description": "Tucks inside, build-up role"},
        "anchor_dm": {"spatial_center": (35, 40), "description": "Shields defense, recycles"},
        "box_to_box": {"spatial_center": (55, 40), "description": "All-around midfielder"},
        "deep_playmaker": {"spatial_center": (40, 40), "description": "Dictates tempo from deep"},
        "advanced_playmaker": {"spatial_center": (70, 40), "description": "Creative in final third"},
        "wide_midfielder": {"spatial_center": (60, 15), "description": "Width and crossing"},
        "inside_forward": {"spatial_center": (85, 25), "description": "Cuts inside, goal threat"},
        "traditional_winger": {"spatial_center": (80, 10), "description": "Stays wide, beats fullback"},
        "target_striker": {"spatial_center": (95, 40), "description": "Hold-up play, aerial threat"},
        "poacher": {"spatial_center": (105, 40), "description": "Box presence, finishing"},
        "false_nine": {"spatial_center": (80, 40), "description": "Drops deep, links play"},
    }
    
    def __init__(self, config: Optional[RatioSimilarityConfig] = None):
        self.config = config or RatioSimilarityConfig()
        self.n_roles = self.config.n_roles
        
        self.scaler = StandardScaler()
        self.cluster_model: Optional[GaussianMixture] = None
        self.cluster_centers: Optional[np.ndarray] = None
        self.role_labels: Dict[int, str] = {}
    
    def extract_clustering_features(self, profile: Dict) -> np.ndarray:
        """Extract features for clustering."""
        features = []
        
        # 1. Spatial center (from heatmap)
        if 'spatial' in profile and 'all' in profile['spatial']:
            heatmap = profile['spatial']['all']
            if heatmap.sum() > 0:
                norm_heat = heatmap / heatmap.sum()
                rows, cols = np.mgrid[0:heatmap.shape[0], 0:heatmap.shape[1]]
                x_center = (cols * norm_heat).sum() / norm_heat.sum() * (120 / heatmap.shape[1])
                y_center = (rows * norm_heat).sum() / norm_heat.sum() * (80 / heatmap.shape[0])
                features.extend([x_center, y_center])
            else:
                features.extend([60, 40])
        else:
            features.extend([60, 40])
        
        # 2. Value added
        if 'value_added' in profile:
            va = profile['value_added']
            features.extend([
                va.get('xT_gained', 0),
                va.get('dangerous_pass_share', 0),
                va.get('progressive_share', 0)
            ])
        else:
            features.extend([0, 0, 0])
        
        # 3. Pressing
        if 'pressing' in profile:
            pr = profile['pressing']
            features.extend([
                pr.get('pressure_rate', 0),
                pr.get('high_press_rate', 0)
            ])
        else:
            features.extend([0, 0])
        
        # 4. Passing
        if 'passing' in profile:
            pa = profile['passing']
            features.extend([
                pa.get('forward_pass_rate', 0),
                pa.get('progressive_pass_rate', 0),
                pa.get('pass_length_mean', 0) / 50,
                pa.get('verticality', 0)
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        return np.array(features)
    
    def fit(
        self,
        player_profiles: Dict[str, Dict],
    ) -> Dict[str, Dict]:
        """
        Fit clustering model on player profiles.
        
        Returns mapping of player_id -> role info
        """
        player_ids = list(player_profiles.keys())
        features = []
        
        for pid in player_ids:
            feat = self.extract_clustering_features(player_profiles[pid])
            features.append(feat)
        
        features = np.array(features)
        features = np.nan_to_num(features, nan=0.0)
        
        # Scale
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit GMM
        self.cluster_model = GaussianMixture(
            n_components=self.n_roles,
            covariance_type='full',
            random_state=42,
            n_init=5
        )
        self.cluster_model.fit(features_scaled)
        
        # Get assignments
        cluster_labels = self.cluster_model.predict(features_scaled)
        cluster_probs = self.cluster_model.predict_proba(features_scaled)
        
        # Store centers
        self.cluster_centers = self.scaler.inverse_transform(
            self.cluster_model.means_
        )
        
        # Assign labels
        self._assign_role_labels()
        
        # Create player-role mapping
        player_roles = {}
        for i, pid in enumerate(player_ids):
            player_roles[pid] = {
                'primary_role': self.role_labels.get(cluster_labels[i], f"Role_{cluster_labels[i]}"),
                'role_probabilities': {
                    self.role_labels.get(j, f"Role_{j}"): float(cluster_probs[i, j])
                    for j in range(self.n_roles)
                }
            }
        
        return player_roles
    
    def _assign_role_labels(self):
        """Assign interpretable labels to clusters."""
        if self.cluster_centers is None:
            return
        
        for cluster_idx in range(self.n_roles):
            center = self.cluster_centers[cluster_idx]
            x_center = center[0]
            y_center = center[1]
            
            best_match = None
            best_dist = float('inf')
            
            for role_name, archetype in self.ROLE_ARCHETYPES.items():
                arch_x, arch_y = archetype['spatial_center']
                dist = np.sqrt((x_center - arch_x)**2 + (y_center - arch_y)**2)
                
                if dist < best_dist:
                    best_dist = dist
                    best_match = role_name
            
            self.role_labels[cluster_idx] = best_match if best_dist < 30 else f"custom_role_{cluster_idx}"
