"""
Tests for data builders.
"""

import pytest
import torch
import numpy as np

import sys
sys.path.append('../..')

from src.datasets.builders.event_graph_builder import (
    EventGraphBuilder,
    EventGraphConfig,
    EventGraph,
    EventGraphBatch,
)
from src.datasets.builders.temporal_sequence_builder import (
    TemporalSequenceBuilder,
    TemporalSequenceConfig,
    PlayerSequence,
)
from src.datasets.schema_contracts import (
    EventRecord,
    PlayerRef,
    MatchContext,
    Coordinates,
)


def create_sample_event(event_id="e1", event_type="Pass", n_players=5):
    """Create a sample event for testing."""
    freeze_frame = [
        PlayerRef(
            player_id=i,
            player_name=f"Player {i}",
            position="Midfielder",
            team_id=1 if i < 3 else 2,
            location=Coordinates(50.0 + i * 5, 40.0 + i * 2),
            is_teammate=(i < 3),
            is_actor=(i == 0),
        )
        for i in range(n_players)
    ]
    
    return EventRecord(
        event_id=event_id,
        event_type=event_type,
        actor=freeze_frame[0],
        location=Coordinates(50.0, 40.0),
        match_context=MatchContext(
            match_id=1,
            period=1,
            minute=30,
            second=0,
            home_team="Team A",
            away_team="Team B",
            home_score=1,
            away_score=0,
        ),
        freeze_frame=freeze_frame,
    )


class TestEventGraphBuilder:
    """Tests for EventGraphBuilder."""
    
    def test_create_builder(self):
        """Test builder creation."""
        config = EventGraphConfig()
        builder = EventGraphBuilder(config)
        assert builder is not None
    
    def test_build_single_graph(self):
        """Test building a single event graph."""
        config = EventGraphConfig(max_players=11)
        builder = EventGraphBuilder(config)
        
        event = create_sample_event(n_players=5)
        graph = builder.build(event)
        
        assert isinstance(graph, EventGraph)
        assert graph.node_features is not None
        assert graph.context_features is not None
        assert graph.attention_mask is not None
    
    def test_graph_node_features_shape(self):
        """Test node features have correct shape."""
        config = EventGraphConfig(
            max_players=11,
            position_embedding_dim=30,
        )
        builder = EventGraphBuilder(config)
        
        event = create_sample_event(n_players=5)
        graph = builder.build(event)
        
        # Shape should be [max_players, feature_dim]
        assert graph.node_features.shape[0] == config.max_players
    
    def test_graph_context_features(self):
        """Test context features extraction."""
        config = EventGraphConfig()
        builder = EventGraphBuilder(config)
        
        event = create_sample_event()
        graph = builder.build(event)
        
        # Context should include period, minute, score_diff, etc.
        assert len(graph.context_features) > 0
    
    def test_graph_attention_mask(self):
        """Test attention mask for valid nodes."""
        config = EventGraphConfig(max_players=11)
        builder = EventGraphBuilder(config)
        
        event = create_sample_event(n_players=5)
        graph = builder.build(event)
        
        # First 5 positions should be valid (1), rest should be masked (0)
        assert graph.attention_mask[:5].sum() == 5
        assert graph.attention_mask[5:].sum() == 0
    
    def test_build_batch(self):
        """Test building a batch of graphs."""
        config = EventGraphConfig(max_players=11)
        builder = EventGraphBuilder(config)
        
        events = [create_sample_event(event_id=f"e{i}") for i in range(4)]
        batch = builder.build_batch(events)
        
        assert isinstance(batch, EventGraphBatch)
        assert batch.node_features.shape[0] == 4  # Batch size
        assert batch.context_features.shape[0] == 4
    
    def test_empty_freeze_frame(self):
        """Test handling event without freeze frame."""
        config = EventGraphConfig(max_players=11)
        builder = EventGraphBuilder(config)
        
        # Event without freeze frame
        event = EventRecord(
            event_id="e1",
            event_type="Pass",
            actor=PlayerRef(
                player_id=1,
                player_name="Player 1",
                position="Midfielder",
                team_id=1,
            ),
            location=Coordinates(50.0, 40.0),
            match_context=MatchContext(
                match_id=1, period=1, minute=0, second=0,
                home_team="A", away_team="B", home_score=0, away_score=0,
            ),
            freeze_frame=[],  # Empty
        )
        
        graph = builder.build(event)
        
        # Should still work, with only actor as node
        assert graph.node_features is not None


class TestTemporalSequenceBuilder:
    """Tests for TemporalSequenceBuilder."""
    
    def test_create_builder(self):
        """Test builder creation."""
        config = TemporalSequenceConfig()
        builder = TemporalSequenceBuilder(config)
        assert builder is not None
    
    def test_build_sequence(self):
        """Test building a player sequence."""
        config = TemporalSequenceConfig(max_sequence_length=50)
        builder = TemporalSequenceBuilder(config)
        
        # Create fake event embeddings
        event_embeddings = [torch.randn(64) for _ in range(30)]
        time_positions = list(range(30))
        
        sequence = builder.build(
            player_id=1,
            event_embeddings=event_embeddings,
            time_positions=time_positions,
        )
        
        assert isinstance(sequence, PlayerSequence)
        assert sequence.event_embeddings.shape[0] == config.max_sequence_length
        assert len(sequence.time_positions) == config.max_sequence_length
    
    def test_sequence_padding(self):
        """Test sequence padding for short sequences."""
        config = TemporalSequenceConfig(max_sequence_length=50)
        builder = TemporalSequenceBuilder(config)
        
        # Create short sequence
        event_embeddings = [torch.randn(64) for _ in range(10)]
        time_positions = list(range(10))
        
        sequence = builder.build(
            player_id=1,
            event_embeddings=event_embeddings,
            time_positions=time_positions,
        )
        
        # Should be padded to max_length
        assert sequence.event_embeddings.shape[0] == 50
        
        # Attention mask should have 10 valid positions
        assert sequence.attention_mask[:10].sum() == 10
        assert sequence.attention_mask[10:].sum() == 0
    
    def test_sequence_truncation(self):
        """Test sequence truncation for long sequences."""
        config = TemporalSequenceConfig(max_sequence_length=50)
        builder = TemporalSequenceBuilder(config)
        
        # Create long sequence
        event_embeddings = [torch.randn(64) for _ in range(100)]
        time_positions = list(range(100))
        
        sequence = builder.build(
            player_id=1,
            event_embeddings=event_embeddings,
            time_positions=time_positions,
        )
        
        # Should be truncated to max_length
        assert sequence.event_embeddings.shape[0] == 50
    
    def test_build_batch(self):
        """Test building a batch of sequences."""
        config = TemporalSequenceConfig(max_sequence_length=50)
        builder = TemporalSequenceBuilder(config)
        
        sequences = []
        for i in range(4):
            n_events = 20 + i * 5
            event_embeddings = [torch.randn(64) for _ in range(n_events)]
            time_positions = list(range(n_events))
            
            seq = builder.build(
                player_id=i,
                event_embeddings=event_embeddings,
                time_positions=time_positions,
            )
            sequences.append(seq)
        
        batch = builder.batch(sequences)
        
        assert batch.event_embeddings.shape[0] == 4
        assert batch.event_embeddings.shape[1] == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
