"""
Temporal Sequence Builder

Builds per-player chronological sequences of event embeddings
for the temporal Transformer encoder.
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict

from ..schema_contracts import EventRecord, PlayerRef


@dataclass
class TemporalSequenceConfig:
    """Configuration for temporal sequence construction."""
    # Sequence settings
    max_sequence_length: int = 512       # Max events per player sequence
    min_sequence_length: int = 10        # Minimum events required
    
    # Time encoding
    use_time_encoding: bool = True       # Add positional time encodings
    time_encoding_dim: int = 32          # Dimension of time encodings
    
    # Event embedding settings
    event_embedding_dim: int = 128       # Expected dim from event GNN
    
    # Padding
    pad_sequences: bool = True           # Pad shorter sequences
    pad_value: float = 0.0               # Value for padding


@dataclass
class PlayerSequence:
    """
    Temporal sequence for a single player.
    
    Attributes:
        player_id: Unique player identifier
        event_ids: List of event IDs in chronological order
        event_embeddings: [L, D] tensor of event embeddings
        time_positions: [L] tensor of normalized time positions
        attention_mask: [L] mask for valid positions
        match_ids: List of match IDs for each event
        sequence_length: Number of actual events (before padding)
    """
    player_id: int
    event_ids: List[str]
    event_embeddings: torch.Tensor
    time_positions: torch.Tensor
    attention_mask: torch.Tensor
    match_ids: List[int]
    sequence_length: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to(self, device: torch.device) -> "PlayerSequence":
        """Move tensors to device."""
        return PlayerSequence(
            player_id=self.player_id,
            event_ids=self.event_ids,
            event_embeddings=self.event_embeddings.to(device),
            time_positions=self.time_positions.to(device),
            attention_mask=self.attention_mask.to(device),
            match_ids=self.match_ids,
            sequence_length=self.sequence_length,
            metadata=self.metadata,
        )


@dataclass
class PlayerSequenceBatch:
    """
    Batched player sequences for efficient processing.
    
    Attributes:
        player_ids: [B] tensor of player IDs
        event_embeddings: [B, L, D] batched event embeddings
        time_positions: [B, L] batched time positions
        attention_mask: [B, L] batched attention masks
        sequence_lengths: [B] actual sequence lengths
    """
    player_ids: torch.Tensor
    event_embeddings: torch.Tensor
    time_positions: torch.Tensor
    attention_mask: torch.Tensor
    sequence_lengths: torch.Tensor
    
    def to(self, device: torch.device) -> "PlayerSequenceBatch":
        """Move tensors to device."""
        return PlayerSequenceBatch(
            player_ids=self.player_ids.to(device),
            event_embeddings=self.event_embeddings.to(device),
            time_positions=self.time_positions.to(device),
            attention_mask=self.attention_mask.to(device),
            sequence_lengths=self.sequence_lengths.to(device),
        )
    
    @property
    def batch_size(self) -> int:
        return self.player_ids.shape[0]
    
    @property
    def max_length(self) -> int:
        return self.event_embeddings.shape[1]


class TemporalSequenceBuilder:
    """
    Builds temporal sequences of events per player.
    
    This builder takes event embeddings (from the event GNN) and organizes them
    into chronological sequences per player for the temporal Transformer.
    
    The temporal order captures how a player's involvement evolves over time,
    which is crucial for learning dynamic playing styles.
    
    Usage:
        builder = TemporalSequenceBuilder(config)
        
        # Group events by player
        player_events = builder.group_events_by_player(events)
        
        # Build sequences (after getting event embeddings from GNN)
        sequences = builder.build_sequences(player_events, event_embeddings)
        
        # Batch for model input
        batch = builder.batch_sequences(sequences)
    """
    
    def __init__(self, config: Optional[TemporalSequenceConfig] = None):
        """Initialize the builder with configuration."""
        self.config = config or TemporalSequenceConfig()
    
    def group_events_by_player(
        self,
        events: List[EventRecord],
        actor_only: bool = True,
    ) -> Dict[int, List[Tuple[EventRecord, int]]]:
        """
        Group events by player ID with their indices.
        
        Args:
            events: List of EventRecord objects (should be sorted by time)
            actor_only: If True, only include events where player is the actor
            
        Returns:
            Dict mapping player_id -> List[(event, original_index)]
        """
        player_events = defaultdict(list)
        
        for idx, event in enumerate(events):
            if actor_only:
                # Only include actor
                if event.actor:
                    player_events[event.actor.player_id].append((event, idx))
            else:
                # Include all players in the event
                for player in event.players:
                    player_events[player.player_id].append((event, idx))
        
        return dict(player_events)
    
    def _compute_time_position(
        self,
        event: EventRecord,
        reference_time: float = 0.0,
    ) -> float:
        """
        Compute normalized time position for an event.
        
        Args:
            event: EventRecord object
            reference_time: Reference time for normalization
            
        Returns:
            Normalized time position in [0, 1] range
        """
        # Compute absolute time in seconds
        period_offset = {1: 0, 2: 45 * 60, 3: 90 * 60, 4: 105 * 60, 5: 120 * 60}
        offset = period_offset.get(event.context.period, 0)
        
        event_time = offset + (event.context.minute * 60) + event.context.second
        
        # Normalize to [0, 1] based on 120 minutes max
        max_time = 120 * 60  # 120 minutes in seconds
        return min(event_time / max_time, 1.0)
    
    def _generate_time_encoding(
        self,
        time_positions: np.ndarray,
        dim: int,
    ) -> np.ndarray:
        """
        Generate sinusoidal time encodings.
        
        Args:
            time_positions: [L] array of normalized time positions
            dim: Encoding dimension
            
        Returns:
            [L, dim] array of time encodings
        """
        L = len(time_positions)
        encodings = np.zeros((L, dim), dtype=np.float32)
        
        # Sinusoidal encoding
        for i in range(dim // 2):
            freq = 1.0 / (10000 ** (2 * i / dim))
            encodings[:, 2 * i] = np.sin(time_positions * freq * 1000)
            encodings[:, 2 * i + 1] = np.cos(time_positions * freq * 1000)
        
        return encodings
    
    def build_sequence(
        self,
        player_id: int,
        player_events: List[Tuple[EventRecord, int]],
        event_embeddings: torch.Tensor,
    ) -> Optional[PlayerSequence]:
        """
        Build a temporal sequence for a single player.
        
        Args:
            player_id: Player identifier
            player_events: List of (event, index) tuples for this player
            event_embeddings: [N, D] tensor of all event embeddings
            
        Returns:
            PlayerSequence or None if insufficient events
        """
        # Check minimum length
        if len(player_events) < self.config.min_sequence_length:
            return None
        
        # Sort by time (should already be sorted, but ensure)
        player_events = sorted(
            player_events,
            key=lambda x: (x[0].context.period, x[0].context.minute, x[0].context.second)
        )
        
        # Truncate to max length
        if len(player_events) > self.config.max_sequence_length:
            # Keep most recent events
            player_events = player_events[-self.config.max_sequence_length:]
        
        seq_length = len(player_events)
        
        # Extract event embeddings and metadata
        event_ids = []
        match_ids = []
        time_positions = []
        embedding_indices = []
        
        for event, idx in player_events:
            event_ids.append(event.event_id)
            match_ids.append(event.match_id)
            time_positions.append(self._compute_time_position(event))
            embedding_indices.append(idx)
        
        # Get embeddings for this player's events
        embeddings = event_embeddings[embedding_indices]  # [L, D]
        
        # Pad if needed
        if self.config.pad_sequences and seq_length < self.config.max_sequence_length:
            pad_length = self.config.max_sequence_length - seq_length
            pad_tensor = torch.full(
                (pad_length, embeddings.shape[1]),
                self.config.pad_value,
                dtype=embeddings.dtype,
            )
            embeddings = torch.cat([embeddings, pad_tensor], dim=0)
            
            # Pad other lists
            event_ids.extend([""] * pad_length)
            match_ids.extend([-1] * pad_length)
            time_positions.extend([0.0] * pad_length)
        
        # Convert to tensors
        time_positions_tensor = torch.tensor(time_positions, dtype=torch.float32)
        
        # Build attention mask (True for valid positions)
        attention_mask = torch.zeros(len(time_positions), dtype=torch.bool)
        attention_mask[:seq_length] = True
        
        # Add time encoding to embeddings if configured
        if self.config.use_time_encoding:
            time_encoding = self._generate_time_encoding(
                np.array(time_positions[:seq_length]),
                self.config.time_encoding_dim,
            )
            # Note: time encoding is added in the Transformer, not here
            # This just stores the time positions for the encoder
        
        # Collect metadata
        unique_matches = set(m for m in match_ids if m >= 0)
        metadata = {
            "n_events": seq_length,
            "n_matches": len(unique_matches),
            "match_ids": list(unique_matches),
        }
        
        return PlayerSequence(
            player_id=player_id,
            event_ids=event_ids,
            event_embeddings=embeddings,
            time_positions=time_positions_tensor,
            attention_mask=attention_mask,
            match_ids=match_ids,
            sequence_length=seq_length,
            metadata=metadata,
        )
    
    def build_sequences(
        self,
        events: List[EventRecord],
        event_embeddings: torch.Tensor,
        actor_only: bool = True,
        min_events: Optional[int] = None,
    ) -> Dict[int, PlayerSequence]:
        """
        Build temporal sequences for all players.
        
        Args:
            events: List of EventRecord objects (sorted by time)
            event_embeddings: [N, D] tensor of event embeddings from GNN
            actor_only: Only include events where player is actor
            min_events: Minimum events for inclusion (overrides config)
            
        Returns:
            Dict mapping player_id -> PlayerSequence
        """
        if min_events is not None:
            original_min = self.config.min_sequence_length
            self.config.min_sequence_length = min_events
        
        # Group events by player
        player_events = self.group_events_by_player(events, actor_only)
        
        # Build sequences
        sequences = {}
        for player_id, events_with_idx in player_events.items():
            sequence = self.build_sequence(player_id, events_with_idx, event_embeddings)
            if sequence is not None:
                sequences[player_id] = sequence
        
        if min_events is not None:
            self.config.min_sequence_length = original_min
        
        return sequences
    
    def batch_sequences(
        self,
        sequences: List[PlayerSequence],
    ) -> PlayerSequenceBatch:
        """
        Batch multiple player sequences for efficient processing.
        
        Args:
            sequences: List of PlayerSequence objects
            
        Returns:
            PlayerSequenceBatch object
        """
        batch_size = len(sequences)
        
        # Find max length in batch (should be same if padded)
        max_len = max(s.event_embeddings.shape[0] for s in sequences)
        embed_dim = sequences[0].event_embeddings.shape[1]
        
        # Initialize batch tensors
        player_ids = torch.tensor([s.player_id for s in sequences], dtype=torch.long)
        event_embeddings = torch.zeros(batch_size, max_len, embed_dim)
        time_positions = torch.zeros(batch_size, max_len)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        sequence_lengths = torch.tensor([s.sequence_length for s in sequences], dtype=torch.long)
        
        # Fill batch
        for i, seq in enumerate(sequences):
            seq_len = seq.event_embeddings.shape[0]
            event_embeddings[i, :seq_len] = seq.event_embeddings
            time_positions[i, :seq_len] = seq.time_positions
            attention_mask[i, :seq_len] = seq.attention_mask
        
        return PlayerSequenceBatch(
            player_ids=player_ids,
            event_embeddings=event_embeddings,
            time_positions=time_positions,
            attention_mask=attention_mask,
            sequence_lengths=sequence_lengths,
        )
    
    def get_player_statistics(
        self,
        events: List[EventRecord],
        actor_only: bool = True,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Get statistics about player event counts.
        
        Useful for determining which players have enough data.
        
        Args:
            events: List of EventRecord objects
            actor_only: Only count events where player is actor
            
        Returns:
            Dict mapping player_id -> statistics dict
        """
        player_events = self.group_events_by_player(events, actor_only)
        
        stats = {}
        for player_id, events_list in player_events.items():
            match_ids = set(e[0].match_id for e in events_list)
            stats[player_id] = {
                "n_events": len(events_list),
                "n_matches": len(match_ids),
                "meets_minimum": len(events_list) >= self.config.min_sequence_length,
            }
        
        return stats


def create_temporal_sequence_builder(
    max_length: int = 512,
    min_length: int = 10,
    event_dim: int = 128,
) -> TemporalSequenceBuilder:
    """
    Factory function to create a TemporalSequenceBuilder with common settings.
    
    Args:
        max_length: Maximum sequence length
        min_length: Minimum sequence length
        event_dim: Event embedding dimension
        
    Returns:
        Configured TemporalSequenceBuilder
    """
    config = TemporalSequenceConfig(
        max_sequence_length=max_length,
        min_sequence_length=min_length,
        event_embedding_dim=event_dim,
        use_time_encoding=True,
        pad_sequences=True,
    )
    return TemporalSequenceBuilder(config)
