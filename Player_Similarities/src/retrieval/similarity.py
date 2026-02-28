"""
Similarity Functions

Implements various similarity measures for player embeddings:
1. Cosine similarity (standard)
2. Reliability-weighted similarity
3. Role-aware similarity
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class SimilarityResult:
    """Container for similarity search results."""
    
    player_ids: List[int]
    scores: List[float]
    ranks: List[int]
    metadata: Optional[List[Dict]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {
            "player_ids": self.player_ids,
            "scores": self.scores,
            "ranks": self.ranks,
        }
        if self.metadata:
            result["metadata"] = self.metadata
        return result


class CosineSimilarity:
    """
    Standard cosine similarity between embeddings.
    
    sim(a, b) = (a · b) / (||a|| ||b||)
    """
    
    def __init__(self, normalize: bool = True):
        self.normalize = normalize
    
    def __call__(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cosine similarity between query and candidates.
        
        Args:
            query: [D] or [1, D] query embedding
            candidates: [N, D] candidate embeddings
            
        Returns:
            [N] similarity scores
        """
        if query.dim() == 1:
            query = query.unsqueeze(0)
        
        if self.normalize:
            query = F.normalize(query, p=2, dim=-1)
            candidates = F.normalize(candidates, p=2, dim=-1)
        
        # [1, D] @ [D, N] -> [1, N] -> [N]
        similarities = torch.matmul(query, candidates.t()).squeeze(0)
        
        return similarities
    
    def pairwise(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute all pairwise similarities.
        
        Args:
            embeddings: [N, D] embeddings
            
        Returns:
            [N, N] similarity matrix
        """
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return torch.matmul(embeddings, embeddings.t())


class ReliabilityWeightedSimilarity:
    """
    Reliability-weighted similarity.
    
    Adjusts similarity scores based on the reliability of both
    query and candidate embeddings.
    
    weighted_sim(a, b) = base_sim(a, b) * sqrt(r_a * r_b)
    
    This downweights matches involving low-data players.
    """
    
    def __init__(
        self,
        base_similarity: Optional[CosineSimilarity] = None,
        min_reliability: float = 0.1,
        weight_power: float = 0.5,
    ):
        self.base_sim = base_similarity or CosineSimilarity()
        self.min_reliability = min_reliability
        self.weight_power = weight_power
    
    def __call__(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
        query_reliability: float,
        candidate_reliabilities: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute reliability-weighted similarity.
        
        Args:
            query: [D] query embedding
            candidates: [N, D] candidate embeddings
            query_reliability: Reliability score of query (0-1)
            candidate_reliabilities: [N] reliability scores
            
        Returns:
            [N] weighted similarity scores
        """
        # Base similarity
        base_scores = self.base_sim(query, candidates)
        
        # Clamp reliabilities
        query_r = max(query_reliability, self.min_reliability)
        cand_r = candidate_reliabilities.clamp(min=self.min_reliability)
        
        # Combined reliability weight
        weights = (query_r * cand_r).pow(self.weight_power)
        
        # Weighted scores
        return base_scores * weights


class RoleAwareSimilarity:
    """
    Role-aware similarity that considers player positions.
    
    Options:
    1. Hard filter: Only compare within same role
    2. Soft weighting: Downweight cross-role matches
    3. Role distance: Use role embedding distance
    """
    
    def __init__(
        self,
        base_similarity: Optional[CosineSimilarity] = None,
        mode: str = "soft",  # 'hard', 'soft', 'distance'
        cross_role_penalty: float = 0.5,
        role_embeddings: Optional[torch.Tensor] = None,
    ):
        self.base_sim = base_similarity or CosineSimilarity()
        self.mode = mode
        self.cross_role_penalty = cross_role_penalty
        self.role_embeddings = role_embeddings
    
    def __call__(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
        query_role: int,
        candidate_roles: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute role-aware similarity.
        
        Args:
            query: [D] query embedding
            candidates: [N, D] candidate embeddings
            query_role: Role index of query
            candidate_roles: [N] role indices of candidates
            
        Returns:
            [N] similarity scores (or -inf for filtered out)
        """
        base_scores = self.base_sim(query, candidates)
        
        same_role = candidate_roles == query_role
        
        if self.mode == "hard":
            # Only same-role matches
            scores = base_scores.clone()
            scores[~same_role] = float('-inf')
            return scores
        
        elif self.mode == "soft":
            # Penalize cross-role matches
            weights = torch.ones_like(base_scores)
            weights[~same_role] = self.cross_role_penalty
            return base_scores * weights
        
        elif self.mode == "distance" and self.role_embeddings is not None:
            # Use role embedding distance
            query_role_emb = self.role_embeddings[query_role]
            cand_role_embs = self.role_embeddings[candidate_roles]
            
            role_sim = F.cosine_similarity(
                query_role_emb.unsqueeze(0),
                cand_role_embs,
                dim=-1,
            )
            
            # Combine: player_sim * (0.5 + 0.5 * role_sim)
            role_weight = 0.5 + 0.5 * role_sim
            return base_scores * role_weight
        
        return base_scores


class SimilaritySearch:
    """
    Complete similarity search with filtering and ranking.
    """
    
    def __init__(
        self,
        embeddings: torch.Tensor,
        player_ids: torch.Tensor,
        reliabilities: Optional[torch.Tensor] = None,
        roles: Optional[torch.Tensor] = None,
        metadata: Optional[Dict[int, Dict]] = None,
        similarity_fn: Optional[CosineSimilarity] = None,
    ):
        """
        Args:
            embeddings: [N, D] player embeddings
            player_ids: [N] player IDs
            reliabilities: [N] reliability scores
            roles: [N] role indices
            metadata: Dict mapping player_id -> metadata dict
            similarity_fn: Similarity function to use
        """
        self.embeddings = embeddings
        self.player_ids = player_ids
        self.reliabilities = reliabilities
        self.roles = roles
        self.metadata = metadata or {}
        
        self.similarity_fn = similarity_fn or CosineSimilarity()
        
        # Build ID to index mapping
        self.id_to_idx = {
            pid.item(): i for i, pid in enumerate(player_ids)
        }
    
    def search(
        self,
        query_player_id: int,
        k: int = 10,
        exclude_self: bool = True,
        role_filter: Optional[int] = None,
        min_reliability: Optional[float] = None,
        reliability_weighted: bool = False,
    ) -> SimilarityResult:
        """
        Find k most similar players.
        
        Args:
            query_player_id: ID of query player
            k: Number of results to return
            exclude_self: Whether to exclude query from results
            role_filter: Only return players with this role
            min_reliability: Only return players above this reliability
            reliability_weighted: Whether to weight by reliability
            
        Returns:
            SimilarityResult with top-k matches
        """
        # Get query embedding
        if query_player_id not in self.id_to_idx:
            raise ValueError(f"Player {query_player_id} not found in index")
        
        query_idx = self.id_to_idx[query_player_id]
        query_emb = self.embeddings[query_idx]
        
        # Compute similarities
        if reliability_weighted and self.reliabilities is not None:
            query_rel = self.reliabilities[query_idx].item()
            sim_fn = ReliabilityWeightedSimilarity(self.similarity_fn)
            scores = sim_fn(query_emb, self.embeddings, query_rel, self.reliabilities)
        else:
            scores = self.similarity_fn(query_emb, self.embeddings)
        
        # Apply filters
        mask = torch.ones(len(self.player_ids), dtype=torch.bool)
        
        if exclude_self:
            mask[query_idx] = False
        
        if role_filter is not None and self.roles is not None:
            mask &= (self.roles == role_filter)
        
        if min_reliability is not None and self.reliabilities is not None:
            mask &= (self.reliabilities >= min_reliability)
        
        # Filter scores
        filtered_scores = scores.clone()
        filtered_scores[~mask] = float('-inf')
        
        # Get top-k
        k = min(k, mask.sum().item())
        top_scores, top_indices = torch.topk(filtered_scores, k)
        
        # Build result
        result_ids = self.player_ids[top_indices].tolist()
        result_scores = top_scores.tolist()
        result_ranks = list(range(1, k + 1))
        
        # Add metadata
        result_metadata = None
        if self.metadata:
            result_metadata = [
                self.metadata.get(pid, {}) for pid in result_ids
            ]
        
        return SimilarityResult(
            player_ids=result_ids,
            scores=result_scores,
            ranks=result_ranks,
            metadata=result_metadata,
        )
    
    def search_by_embedding(
        self,
        query_embedding: torch.Tensor,
        k: int = 10,
        **kwargs,
    ) -> SimilarityResult:
        """
        Search using a raw embedding (not necessarily in index).
        
        Args:
            query_embedding: [D] embedding vector
            k: Number of results
            **kwargs: Additional filters
            
        Returns:
            SimilarityResult
        """
        scores = self.similarity_fn(query_embedding, self.embeddings)
        
        # Apply filters
        mask = torch.ones(len(self.player_ids), dtype=torch.bool)
        
        role_filter = kwargs.get("role_filter")
        if role_filter is not None and self.roles is not None:
            mask &= (self.roles == role_filter)
        
        min_reliability = kwargs.get("min_reliability")
        if min_reliability is not None and self.reliabilities is not None:
            mask &= (self.reliabilities >= min_reliability)
        
        filtered_scores = scores.clone()
        filtered_scores[~mask] = float('-inf')
        
        k = min(k, mask.sum().item())
        top_scores, top_indices = torch.topk(filtered_scores, k)
        
        result_ids = self.player_ids[top_indices].tolist()
        result_scores = top_scores.tolist()
        
        return SimilarityResult(
            player_ids=result_ids,
            scores=result_scores,
            ranks=list(range(1, k + 1)),
            metadata=[self.metadata.get(pid, {}) for pid in result_ids] if self.metadata else None,
        )
    
    def find_clusters(
        self,
        threshold: float = 0.8,
        min_cluster_size: int = 2,
    ) -> List[List[int]]:
        """
        Find clusters of similar players using threshold.
        
        Simple agglomerative approach.
        
        Returns:
            List of player ID clusters
        """
        # Compute full similarity matrix
        sim_matrix = self.similarity_fn.pairwise(self.embeddings)
        
        # Greedy clustering
        N = len(self.player_ids)
        assigned = set()
        clusters = []
        
        for i in range(N):
            if i in assigned:
                continue
            
            # Find all similar players
            similar_mask = sim_matrix[i] >= threshold
            similar_indices = torch.where(similar_mask)[0].tolist()
            
            # Filter already assigned
            cluster = [j for j in similar_indices if j not in assigned]
            
            if len(cluster) >= min_cluster_size:
                clusters.append([self.player_ids[j].item() for j in cluster])
                assigned.update(cluster)
        
        return clusters
