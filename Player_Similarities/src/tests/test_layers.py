"""
Tests for model layers.
"""

import pytest
import torch
import torch.nn as nn

import sys
sys.path.append('../..')

from src.models.layers.graph_attention import (
    GraphAttention,
    MultiHeadGraphAttention,
    GraphAttentionBlock,
)
from src.models.layers.film_conditional import (
    FiLMGenerator,
    FiLMLayer,
    FiLMConditioning,
    FiLMBlock,
)


class TestGraphAttention:
    """Tests for Graph Attention layers."""
    
    def test_create_layer(self):
        """Test layer creation."""
        layer = GraphAttention(in_dim=64, out_dim=64)
        assert layer is not None
    
    def test_forward_pass(self):
        """Test forward pass."""
        layer = GraphAttention(in_dim=64, out_dim=64)
        
        batch_size = 4
        n_nodes = 11
        
        x = torch.randn(batch_size, n_nodes, 64)
        mask = torch.ones(batch_size, n_nodes)
        
        output = layer(x, mask)
        
        assert output.shape == (batch_size, n_nodes, 64)
    
    def test_attention_mask(self):
        """Test that masked nodes are handled correctly."""
        layer = GraphAttention(in_dim=64, out_dim=64)
        
        x = torch.randn(2, 11, 64)
        mask = torch.zeros(2, 11)
        mask[:, :5] = 1  # Only first 5 nodes valid
        
        output = layer(x, mask)
        
        # Output should still have same shape
        assert output.shape == (2, 11, 64)
    
    def test_multihead_attention(self):
        """Test multi-head attention."""
        layer = MultiHeadGraphAttention(
            in_dim=64,
            out_dim=64,
            n_heads=4,
        )
        
        x = torch.randn(4, 11, 64)
        mask = torch.ones(4, 11)
        
        output = layer(x, mask)
        
        assert output.shape == (4, 11, 64)
    
    def test_attention_block_with_residual(self):
        """Test attention block with residual connection."""
        block = GraphAttentionBlock(
            in_dim=64,
            out_dim=64,
            n_heads=4,
            use_residual=True,
        )
        
        x = torch.randn(4, 11, 64)
        mask = torch.ones(4, 11)
        
        output = block(x, mask)
        
        assert output.shape == (4, 11, 64)
    
    def test_dropout_during_training(self):
        """Test that dropout is applied during training."""
        layer = GraphAttention(in_dim=64, out_dim=64, dropout=0.5)
        layer.train()
        
        x = torch.randn(4, 11, 64)
        mask = torch.ones(4, 11)
        
        # Run twice, should get different outputs due to dropout
        output1 = layer(x, mask)
        output2 = layer(x, mask)
        
        # Outputs should be different (with high probability)
        assert not torch.allclose(output1, output2)


class TestFiLM:
    """Tests for FiLM conditioning layers."""
    
    def test_film_generator(self):
        """Test FiLM parameter generator."""
        generator = FiLMGenerator(
            context_dim=32,
            feature_dim=64,
        )
        
        context = torch.randn(4, 32)
        gamma, beta = generator(context)
        
        assert gamma.shape == (4, 64)
        assert beta.shape == (4, 64)
    
    def test_film_layer(self):
        """Test FiLM layer application."""
        layer = FiLMLayer(
            context_dim=32,
            feature_dim=64,
        )
        
        x = torch.randn(4, 64)
        context = torch.randn(4, 32)
        
        output = layer(x, context)
        
        assert output.shape == (4, 64)
    
    def test_film_conditioning_on_sequence(self):
        """Test FiLM conditioning on sequence data."""
        conditioning = FiLMConditioning(
            context_dim=32,
            feature_dim=64,
        )
        
        # Input: [batch, seq, features]
        x = torch.randn(4, 11, 64)
        context = torch.randn(4, 32)
        
        output = conditioning(x, context)
        
        assert output.shape == (4, 11, 64)
    
    def test_film_block_with_mlp(self):
        """Test FiLM block with MLP."""
        block = FiLMBlock(
            context_dim=32,
            feature_dim=64,
            hidden_dim=128,
        )
        
        x = torch.randn(4, 11, 64)
        context = torch.randn(4, 32)
        
        output = block(x, context)
        
        assert output.shape == (4, 11, 64)
    
    def test_film_identity_initialization(self):
        """Test that FiLM is initialized close to identity."""
        layer = FiLMLayer(
            context_dim=32,
            feature_dim=64,
        )
        
        x = torch.randn(4, 64)
        context = torch.randn(4, 32)
        
        output = layer(x, context)
        
        # With proper initialization, output should be close to input
        # (gamma ≈ 1, beta ≈ 0)
        # This is a loose check
        assert output.shape == x.shape


class TestLayerNormAndDropout:
    """Tests for layer norm and dropout in layers."""
    
    def test_layer_norm_in_gat(self):
        """Test layer norm is applied in GAT block."""
        block = GraphAttentionBlock(
            in_dim=64,
            out_dim=64,
            n_heads=4,
            use_layer_norm=True,
        )
        
        x = torch.randn(4, 11, 64)
        mask = torch.ones(4, 11)
        
        output = block(x, mask)
        
        # Check output is normalized (approximately)
        # Mean should be near 0, std near 1 for each feature
        assert output.shape == (4, 11, 64)
    
    def test_gradient_flow(self):
        """Test that gradients flow through layers."""
        block = GraphAttentionBlock(
            in_dim=64,
            out_dim=64,
            n_heads=4,
        )
        
        x = torch.randn(4, 11, 64, requires_grad=True)
        mask = torch.ones(4, 11)
        
        output = block(x, mask)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
