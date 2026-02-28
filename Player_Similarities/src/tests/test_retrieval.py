"""
Tests for retrieval modules.
"""

import pytest
import torch
import numpy as np

import sys
sys.path.append('../..')

from src.retrieval.similarity import (
    CosineSimilarity,
    ReliabilityWeightedSimilarity,
    RoleAwareSimilarity,
    SimilaritySearch,
    SimilarityResult,
)
from src.retrieval.index import EmbeddingIndex, EmbeddingStore
from src.retrieval.api import PlayerSimilarityAPI, PlayerFilter


class TestCosineSimilarity:
    """Tests for cosine similarity."""
    
    def test_identical_vectors(self):
        """Test similarity of identical vectors is 1."""
        sim_fn = CosineSimilarity()
        
        query = torch.randn(128)
        candidates = query.unsqueeze(0)
        
        sim = sim_fn(query, candidates)
        
        assert torch.isclose(sim, torch.tensor([1.0]), atol=1e-5)
    
    def test_orthogonal_vectors(self):
        """Test similarity of orthogonal vectors is 0."""
        sim_fn = CosineSimilarity()
        
        query = torch.zeros(128)
        query[0] = 1.0
        
        candidate = torch.zeros(128)
        candidate[1] = 1.0
        
        sim = sim_fn(query, candidate.unsqueeze(0))
        
        assert torch.isclose(sim, torch.tensor([0.0]), atol=1e-5)
    
    def test_batch_similarity(self):
        """Test similarity with multiple candidates."""
        sim_fn = CosineSimilarity()
        
        query = torch.randn(128)
        candidates = torch.randn(10, 128)
        
        sim = sim_fn(query, candidates)
        
        assert sim.shape == (10,)
        assert (sim >= -1).all() and (sim <= 1).all()
    
    def test_pairwise_similarity(self):
        """Test pairwise similarity matrix."""
        sim_fn = CosineSimilarity()
        
        embeddings = torch.randn(20, 128)
        
        sim_matrix = sim_fn.pairwise(embeddings)
        
        assert sim_matrix.shape == (20, 20)
        # Diagonal should be 1 (self-similarity)
        assert torch.allclose(sim_matrix.diag(), torch.ones(20), atol=1e-5)
        # Matrix should be symmetric
        assert torch.allclose(sim_matrix, sim_matrix.t(), atol=1e-5)


class TestReliabilityWeightedSimilarity:
    """Tests for reliability-weighted similarity."""
    
    def test_weighting(self):
        """Test that reliability affects the score."""
        sim_fn = ReliabilityWeightedSimilarity()
        
        query = torch.randn(128)
        candidates = torch.randn(10, 128)
        
        # High reliability
        high_rel = torch.ones(10)
        sim_high = sim_fn(query, candidates, 1.0, high_rel)
        
        # Low reliability
        low_rel = torch.ones(10) * 0.1
        sim_low = sim_fn(query, candidates, 1.0, low_rel)
        
        # Higher reliability should give higher absolute scores
        assert sim_high.abs().mean() > sim_low.abs().mean()


class TestSimilaritySearch:
    """Tests for similarity search."""
    
    @pytest.fixture
    def search_index(self):
        """Create a search index for testing."""
        n_players = 100
        embeddings = torch.randn(n_players, 128)
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        
        player_ids = torch.arange(n_players)
        reliabilities = torch.rand(n_players)
        roles = torch.randint(0, 5, (n_players,))
        
        metadata = {
            i: {"name": f"Player {i}", "role": roles[i].item()}
            for i in range(n_players)
        }
        
        return SimilaritySearch(
            embeddings=embeddings,
            player_ids=player_ids,
            reliabilities=reliabilities,
            roles=roles,
            metadata=metadata,
        )
    
    def test_basic_search(self, search_index):
        """Test basic similarity search."""
        result = search_index.search(query_player_id=0, k=10)
        
        assert isinstance(result, SimilarityResult)
        assert len(result.player_ids) == 10
        assert len(result.scores) == 10
        assert 0 not in result.player_ids  # Should exclude self
    
    def test_search_with_role_filter(self, search_index):
        """Test search with role filter."""
        query_role = search_index.roles[0].item()
        result = search_index.search(
            query_player_id=0,
            k=10,
            role_filter=query_role,
        )
        
        # All results should have the same role
        for pid in result.player_ids:
            assert search_index.metadata[pid]["role"] == query_role
    
    def test_search_with_reliability_filter(self, search_index):
        """Test search with minimum reliability filter."""
        result = search_index.search(
            query_player_id=0,
            k=10,
            min_reliability=0.5,
        )
        
        for pid in result.player_ids:
            idx = search_index.id_to_idx[pid]
            assert search_index.reliabilities[idx] >= 0.5
    
    def test_search_by_embedding(self, search_index):
        """Test search using raw embedding."""
        query_emb = torch.randn(128)
        
        result = search_index.search_by_embedding(query_emb, k=10)
        
        assert len(result.player_ids) == 10


class TestEmbeddingIndex:
    """Tests for embedding index."""
    
    def test_build_index(self):
        """Test building an index."""
        embeddings = {i: torch.randn(128) for i in range(100)}
        
        index = EmbeddingIndex(embeddings)
        
        assert len(index) == 100
    
    def test_query_index(self):
        """Test querying the index."""
        embeddings = {i: torch.randn(128) for i in range(100)}
        index = EmbeddingIndex(embeddings)
        
        query = torch.randn(128)
        result_ids, result_scores = index.query(query, k=10)
        
        assert len(result_ids) == 10
        assert len(result_scores) == 10
    
    def test_query_by_id(self):
        """Test querying by player ID."""
        embeddings = {i: torch.randn(128) for i in range(100)}
        index = EmbeddingIndex(embeddings)
        
        result_ids, result_scores = index.query_by_id(0, k=10)
        
        assert len(result_ids) == 10
        assert 0 not in result_ids  # Self excluded
    
    def test_add_remove(self):
        """Test adding and removing from index."""
        embeddings = {i: torch.randn(128) for i in range(10)}
        index = EmbeddingIndex(embeddings)
        
        assert len(index) == 10
        
        # Add
        index.add(100, torch.randn(128))
        assert len(index) == 11
        assert 100 in index
        
        # Remove
        index.remove(100)
        assert len(index) == 10
        assert 100 not in index
    
    def test_get_embedding(self):
        """Test retrieving embedding by ID."""
        embeddings = {i: torch.randn(128) for i in range(10)}
        index = EmbeddingIndex(embeddings)
        
        emb = index.get_embedding(0)
        
        assert emb is not None
        assert emb.shape == (128,)
        
        # Non-existent player
        assert index.get_embedding(999) is None


class TestPlayerFilter:
    """Tests for player filters."""
    
    def test_create_filter(self):
        """Test filter creation."""
        filter = PlayerFilter(
            same_role=True,
            min_reliability=0.5,
            leagues=["La Liga", "Premier League"],
        )
        
        assert filter.same_role is True
        assert filter.min_reliability == 0.5
        assert "La Liga" in filter.leagues
    
    def test_filter_defaults(self):
        """Test filter defaults."""
        filter = PlayerFilter()
        
        assert filter.same_role is False
        assert filter.min_reliability is None
        assert filter.exclude_player_ids == []


class TestSimilarityResult:
    """Tests for similarity result container."""
    
    def test_create_result(self):
        """Test result creation."""
        result = SimilarityResult(
            player_ids=[1, 2, 3],
            scores=[0.9, 0.8, 0.7],
            ranks=[1, 2, 3],
        )
        
        assert len(result.player_ids) == 3
        assert result.scores[0] == 0.9
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = SimilarityResult(
            player_ids=[1, 2, 3],
            scores=[0.9, 0.8, 0.7],
            ranks=[1, 2, 3],
            metadata=[{"name": "A"}, {"name": "B"}, {"name": "C"}],
        )
        
        d = result.to_dict()
        
        assert "player_ids" in d
        assert "scores" in d
        assert "metadata" in d


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
