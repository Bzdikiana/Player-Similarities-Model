"""
Evaluation Metrics

Metrics for evaluating player embeddings:
- Recall@K
- NDCG (Normalized Discounted Cumulative Gain)
- Embedding space diagnostics
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def compute_similarity_matrix(
    embeddings: torch.Tensor,
    metric: str = "cosine",
) -> torch.Tensor:
    """
    Compute pairwise similarity matrix.
    
    Args:
        embeddings: [N, D] embeddings
        metric: 'cosine' or 'dot'
        
    Returns:
        [N, N] similarity matrix
    """
    if metric == "cosine":
        # L2 normalize
        embeddings = F.normalize(embeddings, p=2, dim=-1)
    
    # Dot product similarity
    return torch.matmul(embeddings, embeddings.t())


def compute_recall_at_k(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    k_values: List[int] = [1, 5, 10],
    metric: str = "cosine",
) -> Dict[str, float]:
    """
    Compute Recall@K for retrieval evaluation.
    
    For each query, check if any of the top-K retrieved items
    have the same label (same player from different context).
    
    Args:
        embeddings: [N, D] embeddings
        labels: [N] player IDs (same ID = same player)
        k_values: List of K values to compute
        metric: Similarity metric
        
    Returns:
        Dict mapping 'recall_at_{k}' -> score
    """
    N = embeddings.shape[0]
    
    # Compute similarity
    sim_matrix = compute_similarity_matrix(embeddings, metric)
    
    # For each sample, find if positives are in top-K
    results = {}
    
    for k in k_values:
        recalls = []
        
        for i in range(N):
            # Get similarities for this query (excluding self)
            sims = sim_matrix[i].clone()
            sims[i] = float('-inf')  # Exclude self
            
            # Get top-K indices
            _, top_k_indices = torch.topk(sims, min(k, N - 1))
            
            # Check if any top-K have same label
            query_label = labels[i]
            top_k_labels = labels[top_k_indices]
            
            hit = (top_k_labels == query_label).any().float()
            recalls.append(hit.item())
        
        results[f"recall_at_{k}"] = np.mean(recalls)
    
    return results


def compute_ndcg(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    k: int = 10,
    metric: str = "cosine",
) -> float:
    """
    Compute NDCG@K (Normalized Discounted Cumulative Gain).
    
    Considers the position of relevant items in the ranking.
    
    Args:
        embeddings: [N, D] embeddings
        labels: [N] player IDs
        k: Cutoff for DCG computation
        metric: Similarity metric
        
    Returns:
        NDCG score
    """
    N = embeddings.shape[0]
    
    sim_matrix = compute_similarity_matrix(embeddings, metric)
    
    ndcgs = []
    
    for i in range(N):
        sims = sim_matrix[i].clone()
        sims[i] = float('-inf')
        
        # Get ranking
        _, ranking = torch.sort(sims, descending=True)
        ranking = ranking[:k]
        
        # Relevance: 1 if same label, 0 otherwise
        query_label = labels[i]
        relevances = (labels[ranking] == query_label).float()
        
        # DCG
        positions = torch.arange(1, len(relevances) + 1, dtype=torch.float32)
        dcg = (relevances / torch.log2(positions + 1)).sum().item()
        
        # Ideal DCG (all relevant items at top)
        n_relevant = (labels == query_label).sum().item() - 1  # Exclude self
        ideal_relevances = torch.zeros(k)
        ideal_relevances[:min(n_relevant, k)] = 1.0
        idcg = (ideal_relevances / torch.log2(positions + 1)).sum().item()
        
        if idcg > 0:
            ndcgs.append(dcg / idcg)
        else:
            ndcgs.append(0.0)
    
    return np.mean(ndcgs)


def compute_embedding_statistics(
    embeddings: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute embedding space statistics/diagnostics.
    
    These help identify issues like:
    - Collapsed embeddings (all similar)
    - Highly varied norms
    - Dimensional collapse
    
    Args:
        embeddings: [N, D] embeddings
        
    Returns:
        Dict with statistics
    """
    N, D = embeddings.shape
    
    # Norms
    norms = torch.norm(embeddings, p=2, dim=-1)
    
    # Pairwise similarities (on normalized)
    normalized = F.normalize(embeddings, p=2, dim=-1)
    sim_matrix = torch.matmul(normalized, normalized.t())
    
    # Get off-diagonal similarities
    mask = ~torch.eye(N, dtype=torch.bool, device=embeddings.device)
    off_diag_sims = sim_matrix[mask]
    
    # Compute statistics
    stats = {
        # Norm statistics
        "norm_mean": norms.mean().item(),
        "norm_std": norms.std().item(),
        "norm_min": norms.min().item(),
        "norm_max": norms.max().item(),
        
        # Similarity statistics
        "sim_mean": off_diag_sims.mean().item(),
        "sim_std": off_diag_sims.std().item(),
        "sim_min": off_diag_sims.min().item(),
        "sim_max": off_diag_sims.max().item(),
        
        # Collapse indicators
        "alignment": off_diag_sims.mean().item(),  # High = collapsed
        "uniformity": torch.pdist(normalized).pow(2).mul(-2).exp().mean().log().item(),
    }
    
    # Per-dimension variance (check for dimensional collapse)
    dim_vars = embeddings.var(dim=0)
    stats["dim_var_mean"] = dim_vars.mean().item()
    stats["dim_var_std"] = dim_vars.std().item()
    stats["n_dead_dims"] = (dim_vars < 1e-6).sum().item()
    
    return stats


class EmbeddingMetrics:
    """
    Comprehensive embedding quality metrics.
    """
    
    def __init__(
        self,
        k_values: List[int] = [1, 5, 10, 20],
        metric: str = "cosine",
    ):
        self.k_values = k_values
        self.metric = metric
    
    def compute_all(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Args:
            embeddings: [N, D] embeddings
            labels: [N] player IDs
            
        Returns:
            Dict with all metrics
        """
        metrics = {}
        
        # Recall@K
        recall_metrics = compute_recall_at_k(
            embeddings, labels, self.k_values, self.metric
        )
        metrics.update(recall_metrics)
        
        # NDCG
        for k in [5, 10]:
            metrics[f"ndcg_at_{k}"] = compute_ndcg(
                embeddings, labels, k, self.metric
            )
        
        # Embedding statistics
        stats = compute_embedding_statistics(embeddings)
        metrics.update(stats)
        
        return metrics
    
    def compute_per_player(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        player_metadata: Optional[Dict[int, Dict]] = None,
    ) -> Dict[int, Dict[str, float]]:
        """
        Compute per-player retrieval quality.
        
        Useful for identifying which players have poor embeddings.
        """
        N = embeddings.shape[0]
        sim_matrix = compute_similarity_matrix(embeddings, self.metric)
        
        per_player = defaultdict(lambda: {"hits_at_10": 0, "count": 0})
        
        for i in range(N):
            player_id = labels[i].item()
            
            sims = sim_matrix[i].clone()
            sims[i] = float('-inf')
            
            _, top_k = torch.topk(sims, min(10, N - 1))
            top_labels = labels[top_k]
            
            hit = (top_labels == labels[i]).any().float().item()
            
            per_player[player_id]["hits_at_10"] += hit
            per_player[player_id]["count"] += 1
        
        # Compute averages
        results = {}
        for pid, data in per_player.items():
            results[pid] = {
                "recall_at_10": data["hits_at_10"] / max(data["count"], 1),
                "n_samples": data["count"],
            }
            
            if player_metadata and pid in player_metadata:
                results[pid].update(player_metadata[pid])
        
        return results


def evaluate_cold_start(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    event_counts: torch.Tensor,
    thresholds: List[int] = [10, 50, 100, 500],
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate embedding quality by data availability.
    
    Checks if cold-start regularization is working by comparing
    retrieval quality for low-data vs high-data players.
    
    Args:
        embeddings: [N, D] embeddings
        labels: [N] player IDs
        event_counts: [N] number of events per sample
        thresholds: Boundaries for binning by event count
        
    Returns:
        Dict mapping bin name -> metrics
    """
    N = embeddings.shape[0]
    
    results = {}
    
    # Bin by event count
    bins = [(0, thresholds[0])] + [
        (thresholds[i], thresholds[i + 1])
        for i in range(len(thresholds) - 1)
    ] + [(thresholds[-1], float('inf'))]
    
    for low, high in bins:
        mask = (event_counts >= low) & (event_counts < high)
        
        if mask.sum() < 2:
            continue
        
        bin_embeddings = embeddings[mask]
        bin_labels = labels[mask]
        
        bin_name = f"events_{low}-{int(high) if high != float('inf') else 'inf'}"
        
        recall = compute_recall_at_k(bin_embeddings, bin_labels, [1, 5, 10])
        recall["n_samples"] = mask.sum().item()
        
        results[bin_name] = recall
    
    return results
