"""
Tests for loss functions.
"""

import pytest
import torch
import torch.nn.functional as F

import sys
sys.path.append('../..')

from src.training.losses import (
    InfoNCELoss,
    SupConLoss,
    ReliabilityWeightedLoss,
    AuxiliaryLosses,
    TripletLoss,
    compute_reliability_weights,
)


class TestInfoNCELoss:
    """Tests for InfoNCE contrastive loss."""
    
    def test_create_loss(self):
        """Test loss creation."""
        loss_fn = InfoNCELoss(temperature=0.07)
        assert loss_fn is not None
    
    def test_forward_pass(self):
        """Test forward pass."""
        loss_fn = InfoNCELoss(temperature=0.07)
        
        # Batch with some repeated labels (positives)
        embeddings = F.normalize(torch.randn(8, 128), dim=-1)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])  # Pairs
        
        loss = loss_fn(embeddings, labels)
        
        assert loss.shape == ()
        assert loss >= 0
    
    def test_loss_decreases_for_similar(self):
        """Test that loss is lower when positives are more similar."""
        loss_fn = InfoNCELoss(temperature=0.07)
        
        # Create embeddings where pairs are very similar
        base = torch.randn(4, 128)
        similar_pairs = torch.cat([base, base + 0.01 * torch.randn(4, 128)], dim=0)
        similar_pairs = F.normalize(similar_pairs, dim=-1)
        
        # Create embeddings where pairs are random
        random_pairs = F.normalize(torch.randn(8, 128), dim=-1)
        
        labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        
        loss_similar = loss_fn(similar_pairs, labels)
        loss_random = loss_fn(random_pairs, labels)
        
        # Similar pairs should have lower loss
        assert loss_similar < loss_random
    
    def test_no_positives_handling(self):
        """Test handling of samples without positives."""
        loss_fn = InfoNCELoss(temperature=0.07)
        
        # All unique labels (no positives)
        embeddings = F.normalize(torch.randn(8, 128), dim=-1)
        labels = torch.arange(8)  # All unique
        
        loss = loss_fn(embeddings, labels)
        
        # Should return 0 or handle gracefully
        assert not torch.isnan(loss)
    
    def test_temperature_effect(self):
        """Test that temperature affects loss scale."""
        embeddings = F.normalize(torch.randn(8, 128), dim=-1)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        
        loss_low_temp = InfoNCELoss(temperature=0.01)(embeddings, labels)
        loss_high_temp = InfoNCELoss(temperature=1.0)(embeddings, labels)
        
        # Different temperatures should give different losses
        assert not torch.isclose(loss_low_temp, loss_high_temp)


class TestSupConLoss:
    """Tests for Supervised Contrastive Loss."""
    
    def test_forward_pass(self):
        """Test forward pass."""
        loss_fn = SupConLoss(temperature=0.07)
        
        embeddings = F.normalize(torch.randn(8, 128), dim=-1)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        
        loss = loss_fn(embeddings, labels)
        
        assert loss.shape == ()
        assert loss >= 0


class TestReliabilityWeightedLoss:
    """Tests for reliability-weighted loss wrapper."""
    
    def test_weighted_loss(self):
        """Test reliability weighting."""
        base_loss = InfoNCELoss(temperature=0.07, reduction="none")
        weighted_loss = ReliabilityWeightedLoss(base_loss, min_weight=0.1)
        
        embeddings = F.normalize(torch.randn(8, 128), dim=-1)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        reliability = torch.tensor([1.0, 1.0, 0.5, 0.5, 0.1, 0.1, 0.8, 0.8])
        
        loss = weighted_loss(embeddings, labels, reliability)
        
        assert loss.shape == ()
        assert not torch.isnan(loss)


class TestAuxiliaryLosses:
    """Tests for auxiliary losses."""
    
    def test_role_classification_loss(self):
        """Test role classification loss."""
        aux = AuxiliaryLosses(n_roles=10)
        
        role_logits = torch.randn(8, 10)
        role_labels = torch.randint(0, 10, (8,))
        
        loss = aux.role_classification_loss(role_logits, role_labels)
        
        assert loss.shape == ()
        assert loss >= 0
    
    def test_consistency_loss(self):
        """Test consistency loss between data and global embeddings."""
        aux = AuxiliaryLosses()
        
        data_emb = F.normalize(torch.randn(8, 128), dim=-1)
        global_emb = F.normalize(torch.randn(8, 128), dim=-1)
        reliability = torch.rand(8)
        
        loss = aux.consistency_loss(data_emb, global_emb, reliability)
        
        assert loss.shape == ()
        assert loss >= 0
    
    def test_combined_auxiliary(self):
        """Test combined auxiliary losses."""
        aux = AuxiliaryLosses(n_roles=10)
        
        losses = aux(
            role_logits=torch.randn(8, 10),
            role_labels=torch.randint(0, 10, (8,)),
            data_embeddings=F.normalize(torch.randn(8, 128), dim=-1),
            global_embeddings=F.normalize(torch.randn(8, 128), dim=-1),
            reliability=torch.rand(8),
        )
        
        assert 'aux_total' in losses
        assert losses['aux_total'] >= 0


class TestTripletLoss:
    """Tests for triplet loss."""
    
    def test_forward_pass(self):
        """Test triplet loss forward pass."""
        loss_fn = TripletLoss(margin=0.3)
        
        anchor = F.normalize(torch.randn(8, 128), dim=-1)
        positive = F.normalize(anchor + 0.1 * torch.randn(8, 128), dim=-1)
        negative = F.normalize(torch.randn(8, 128), dim=-1)
        
        loss = loss_fn(anchor, positive, negative)
        
        assert loss.shape == ()
        assert loss >= 0
    
    def test_margin_effect(self):
        """Test that margin affects the loss."""
        anchor = F.normalize(torch.randn(8, 128), dim=-1)
        positive = anchor.clone()  # Identical
        negative = F.normalize(torch.randn(8, 128), dim=-1)
        
        loss_small_margin = TripletLoss(margin=0.1)(anchor, positive, negative)
        loss_large_margin = TripletLoss(margin=1.0)(anchor, positive, negative)
        
        # Larger margin should give higher loss
        assert loss_large_margin > loss_small_margin


class TestReliabilityWeights:
    """Tests for reliability weight computation."""
    
    def test_compute_weights(self):
        """Test reliability weight computation."""
        n_events = torch.tensor([10, 50, 100, 500, 1000])
        weights = compute_reliability_weights(n_events, k=50.0)
        
        assert weights.shape == n_events.shape
        assert (weights >= 0).all()
        assert (weights <= 1).all()
    
    def test_weight_increases_with_events(self):
        """Test that reliability increases with more events."""
        n_events = torch.tensor([10, 100, 1000])
        weights = compute_reliability_weights(n_events, k=50.0)
        
        assert weights[1] > weights[0]
        assert weights[2] > weights[1]
    
    def test_minimum_weight(self):
        """Test minimum weight clamping."""
        n_events = torch.tensor([0, 1, 5])
        weights = compute_reliability_weights(n_events, k=50.0, min_weight=0.1)
        
        assert (weights >= 0.1).all()


class TestGradientFlow:
    """Tests for gradient flow through losses."""
    
    def test_infonce_gradient(self):
        """Test gradients flow through InfoNCE."""
        loss_fn = InfoNCELoss(temperature=0.07)
        
        embeddings = torch.randn(8, 128, requires_grad=True)
        embeddings_norm = F.normalize(embeddings, dim=-1)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        
        loss = loss_fn(embeddings_norm, labels)
        loss.backward()
        
        assert embeddings.grad is not None
        assert not torch.isnan(embeddings.grad).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
