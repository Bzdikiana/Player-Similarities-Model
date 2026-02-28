"""
Event Graph Builder

Constructs per-event graphs for the event-centric GNN:
- Nodes: players involved in the event
- Edges: fully connected (attention learns what matters)
- Context: match context for FiLM conditioning
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field

from ..schema_contracts import (
    EventRecord, PlayerRef, Coordinates,
    EVENT_TYPE_CATEGORIES, POSITION_CATEGORIES,
    get_event_type_index, get_position_index
)


@dataclass
class EventGraphConfig:
    """Configuration for event graph construction."""
    # Node features
    node_feature_dim: int = 32          # Dimension of node features
    use_position_encoding: bool = True   # Include position one-hot
    use_location_features: bool = True   # Include x, y coordinates
    use_role_features: bool = True       # Include actor/teammate/opponent flags
    
    # Edge features (NEW)
    use_edge_features: bool = True       # Include edge features
    edge_feature_dim: int = 4            # distance, same_team, relative_x, relative_y
    
    # Context features
    context_feature_dim: int = 8         # Dimension of context features
    
    # Graph settings
    max_players_per_event: int = 22      # Max nodes per graph (full lineup)
    include_ball_node: bool = False      # Add virtual ball node
    
    # Coordinate normalization (StatsBomb: 120x80)
    pitch_length: float = 120.0
    pitch_width: float = 80.0


@dataclass
class EventGraph:
    """
    Single event graph representation.
    
    Attributes:
        node_features: [N, D] tensor of node features
        edge_features: [N, N, E] tensor of edge features (distance, same_team, etc.)
        context_features: [C] tensor of context features for FiLM
        attention_mask: [N, N] boolean mask for attention (True = attend)
        player_ids: List of player IDs corresponding to nodes
        event_type_idx: Index of event type
        event_id: Original event ID
    """
    node_features: torch.Tensor
    context_features: torch.Tensor
    attention_mask: torch.Tensor
    player_ids: List[int]
    event_type_idx: int
    event_id: str
    n_players: int = 0
    edge_features: Optional[torch.Tensor] = None  # NEW: [N, N, E] edge features
    
    def to(self, device: torch.device) -> "EventGraph":
        """Move tensors to device."""
        return EventGraph(
            node_features=self.node_features.to(device),
            context_features=self.context_features.to(device),
            attention_mask=self.attention_mask.to(device),
            player_ids=self.player_ids,
            event_type_idx=self.event_type_idx,
            event_id=self.event_id,
            n_players=self.n_players,
            edge_features=self.edge_features.to(device) if self.edge_features is not None else None,
        )


@dataclass  
class EventGraphBatch:
    """
    Batched event graphs for efficient processing.
    
    Attributes:
        node_features: [B, N, D] batched node features (padded)
        edge_features: [B, N, N, E] batched edge features (padded)
        context_features: [B, C] batched context features
        attention_mask: [B, N, N] batched attention masks
        player_ids: List of lists of player IDs
        event_type_indices: [B] event type indices
        event_ids: List of event IDs
        batch_mask: [B, N] which nodes are real (not padding)
    """
    node_features: torch.Tensor
    context_features: torch.Tensor
    attention_mask: torch.Tensor
    player_ids: List[List[int]]
    event_type_indices: torch.Tensor
    event_ids: List[str]
    batch_mask: torch.Tensor
    edge_features: Optional[torch.Tensor] = None  # NEW: [B, N, N, E]
    
    def to(self, device: torch.device) -> "EventGraphBatch":
        """Move tensors to device."""
        return EventGraphBatch(
            node_features=self.node_features.to(device),
            context_features=self.context_features.to(device),
            attention_mask=self.attention_mask.to(device),
            player_ids=self.player_ids,
            event_type_indices=self.event_type_indices.to(device),
            event_ids=self.event_ids,
            batch_mask=self.batch_mask.to(device),
            edge_features=self.edge_features.to(device) if self.edge_features is not None else None,
        )
    
    @property
    def batch_size(self) -> int:
        return self.node_features.shape[0]


class EventGraphBuilder:
    """
    Builds event graphs from EventRecord objects.
    
    Each event becomes a graph where:
    - Nodes = players involved (actor + teammates + opponents from freeze frame)
    - Edges = fully connected (let attention learn relevance)
    - Context = match context for FiLM conditioning
    
    Usage:
        builder = EventGraphBuilder(config)
        graph = builder.build_graph(event_record)
        batch = builder.batch_graphs([graph1, graph2, ...])
    """
    
    def __init__(self, config: Optional[EventGraphConfig] = None):
        """Initialize the builder with configuration."""
        self.config = config or EventGraphConfig()
        
        # Pre-compute feature dimensions
        self._compute_feature_dims()
    
    def _compute_feature_dims(self):
        """Compute the actual feature dimensions based on config."""
        dim = 0
        
        # Position one-hot
        if self.config.use_position_encoding:
            dim += len(POSITION_CATEGORIES) + 1  # +1 for unknown
        
        # Location features (normalized x, y)
        if self.config.use_location_features:
            dim += 2
        
        # Role features (is_actor, on_ball, is_teammate, is_opponent)
        if self.config.use_role_features:
            dim += 4
        
        # Add event type embedding (shared across nodes)
        dim += len(EVENT_TYPE_CATEGORIES) + 1
        
        self._node_feature_dim = dim
        
        # Edge feature dimension: distance, same_team, relative_x, relative_y
        self._edge_feature_dim = 4 if self.config.use_edge_features else 0
        
        # Context dimension (period one-hot + minute + score_diff + is_home)
        self._context_feature_dim = 8  # 5 (period) + 1 (minute) + 1 (score_diff) + 1 (is_home)
    
    def _compute_edge_features(
        self,
        players: List[PlayerRef],
        n_players: int,
    ) -> np.ndarray:
        """
        Compute edge features between all pairs of players.
        
        Edge features:
        - distance: Euclidean distance between players (normalized by pitch diagonal)
        - same_team: 1 if same team, 0 otherwise
        - relative_x: Normalized x difference (direction aware)
        - relative_y: Normalized y difference (direction aware)
        
        Args:
            players: List of PlayerRef objects
            n_players: Number of actual players
            
        Returns:
            [max_players, max_players, 4] edge feature tensor
        """
        max_players = self.config.max_players_per_event
        edge_features = np.zeros((max_players, max_players, 4), dtype=np.float32)
        
        # Pitch diagonal for distance normalization
        pitch_diagonal = np.sqrt(self.config.pitch_length**2 + self.config.pitch_width**2)
        
        # Get locations and team IDs
        locations = []
        team_ids = []
        for player in players[:n_players]:
            if player.location:
                locations.append([player.location.x, player.location.y])
            else:
                locations.append([60.0, 40.0])  # Center of pitch as default
            team_ids.append(player.team_id)
        
        locations = np.array(locations, dtype=np.float32)
        
        # Compute pairwise features
        for i in range(n_players):
            for j in range(n_players):
                if i == j:
                    # Self-edge: zero distance, same team
                    edge_features[i, j, 0] = 0.0  # distance
                    edge_features[i, j, 1] = 1.0  # same_team
                    edge_features[i, j, 2] = 0.0  # relative_x
                    edge_features[i, j, 3] = 0.0  # relative_y
                else:
                    # Distance
                    dx = locations[j, 0] - locations[i, 0]
                    dy = locations[j, 1] - locations[i, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    edge_features[i, j, 0] = dist / pitch_diagonal  # Normalized distance
                    
                    # Same team
                    edge_features[i, j, 1] = 1.0 if team_ids[i] == team_ids[j] else 0.0
                    
                    # Relative position (normalized)
                    edge_features[i, j, 2] = dx / self.config.pitch_length
                    edge_features[i, j, 3] = dy / self.config.pitch_width
        
        return edge_features
    
    def _encode_player_features(
        self, 
        player: PlayerRef, 
        event_type_idx: int,
        actor_team_id: int,
    ) -> np.ndarray:
        """
        Encode a single player's features.
        
        Args:
            player: PlayerRef object
            event_type_idx: Index of the event type
            actor_team_id: Team ID of the event actor
            
        Returns:
            Feature vector for this player
        """
        features = []
        
        # Position one-hot
        if self.config.use_position_encoding:
            pos_onehot = np.zeros(len(POSITION_CATEGORIES) + 1, dtype=np.float32)
            pos_idx = get_position_index(player.position)
            pos_onehot[pos_idx] = 1.0
            features.append(pos_onehot)
        
        # Location features (normalized)
        if self.config.use_location_features:
            if player.location:
                loc = np.array([
                    player.location.x / self.config.pitch_length,
                    player.location.y / self.config.pitch_width,
                ], dtype=np.float32)
            else:
                loc = np.array([0.5, 0.5], dtype=np.float32)  # Default to center
            features.append(loc)
        
        # Role features
        if self.config.use_role_features:
            is_teammate = int(player.team_id == actor_team_id and not player.is_actor)
            is_opponent = int(player.team_id != actor_team_id)
            role_features = np.array([
                float(player.is_actor),
                float(player.on_ball),
                float(is_teammate),
                float(is_opponent),
            ], dtype=np.float32)
            features.append(role_features)
        
        # Event type one-hot (same for all players in event)
        event_onehot = np.zeros(len(EVENT_TYPE_CATEGORIES) + 1, dtype=np.float32)
        event_onehot[event_type_idx] = 1.0
        features.append(event_onehot)
        
        return np.concatenate(features)
    
    def _encode_context(self, event: EventRecord) -> np.ndarray:
        """
        Encode match context for FiLM conditioning.
        
        Args:
            event: EventRecord object
            
        Returns:
            Context feature vector
        """
        return event.context.to_context_vector()
    
    def build_graph(self, event: EventRecord) -> EventGraph:
        """
        Build a graph representation from a single event.
        
        Args:
            event: EventRecord object
            
        Returns:
            EventGraph object
        """
        players = event.players
        n_players = min(len(players), self.config.max_players_per_event)
        
        # Get actor team ID for role encoding
        actor_team_id = event.actor.team_id if event.actor else 0
        
        # Encode event type
        event_type_idx = get_event_type_index(event.event_type)
        
        # Build node features
        node_features = []
        player_ids = []
        
        for i, player in enumerate(players[:n_players]):
            features = self._encode_player_features(player, event_type_idx, actor_team_id)
            node_features.append(features)
            player_ids.append(player.player_id)
        
        # Pad if needed
        while len(node_features) < self.config.max_players_per_event:
            padding = np.zeros(self._node_feature_dim, dtype=np.float32)
            node_features.append(padding)
            player_ids.append(-1)  # Invalid player ID for padding
        
        node_features = np.stack(node_features[:self.config.max_players_per_event])
        
        # Build attention mask (fully connected for real nodes, no attention for padding)
        attention_mask = np.zeros((self.config.max_players_per_event, self.config.max_players_per_event), dtype=bool)
        attention_mask[:n_players, :n_players] = True
        
        # Build edge features (NEW)
        edge_features = None
        if self.config.use_edge_features:
            edge_features = self._compute_edge_features(players, n_players)
            edge_features = torch.from_numpy(edge_features)
        
        # Context features
        context_features = self._encode_context(event)
        
        return EventGraph(
            node_features=torch.from_numpy(node_features),
            context_features=torch.from_numpy(context_features),
            attention_mask=torch.from_numpy(attention_mask),
            player_ids=player_ids[:self.config.max_players_per_event],
            event_type_idx=event_type_idx,
            event_id=event.event_id,
            n_players=n_players,
            edge_features=edge_features,
        )
    
    def batch_graphs(self, graphs: List[EventGraph]) -> EventGraphBatch:
        """
        Batch multiple event graphs for efficient processing.
        
        Args:
            graphs: List of EventGraph objects
            
        Returns:
            EventGraphBatch object
        """
        batch_size = len(graphs)
        
        # Stack tensors
        node_features = torch.stack([g.node_features for g in graphs])
        context_features = torch.stack([g.context_features for g in graphs])
        attention_mask = torch.stack([g.attention_mask for g in graphs])
        
        # Stack edge features if available
        edge_features = None
        if graphs[0].edge_features is not None:
            edge_features = torch.stack([g.edge_features for g in graphs])
        
        # Collect metadata
        player_ids = [g.player_ids for g in graphs]
        event_type_indices = torch.tensor([g.event_type_idx for g in graphs], dtype=torch.long)
        event_ids = [g.event_id for g in graphs]
        
        # Build batch mask (which nodes are real)
        batch_mask = torch.zeros(batch_size, self.config.max_players_per_event, dtype=torch.bool)
        for i, g in enumerate(graphs):
            batch_mask[i, :g.n_players] = True
        
        return EventGraphBatch(
            node_features=node_features,
            context_features=context_features,
            attention_mask=attention_mask,
            player_ids=player_ids,
            event_type_indices=event_type_indices,
            event_ids=event_ids,
            batch_mask=batch_mask,
            edge_features=edge_features,
        )
    
    def build_graphs_from_events(
        self, 
        events: List[EventRecord],
        batch_size: Optional[int] = None,
    ) -> List[EventGraphBatch]:
        """
        Build batched graphs from a list of events.
        
        Args:
            events: List of EventRecord objects
            batch_size: Batch size (None for single batch with all events)
            
        Returns:
            List of EventGraphBatch objects
        """
        # Build individual graphs
        graphs = [self.build_graph(e) for e in events]
        
        if batch_size is None:
            return [self.batch_graphs(graphs)]
        
        # Split into batches
        batches = []
        for i in range(0, len(graphs), batch_size):
            batch_graphs = graphs[i:i + batch_size]
            batches.append(self.batch_graphs(batch_graphs))
        
        return batches
    
    @property
    def node_feature_dim(self) -> int:
        """Get the node feature dimension."""
        return self._node_feature_dim
    
    @property
    def edge_feature_dim(self) -> int:
        """Get the edge feature dimension."""
        return self._edge_feature_dim
    
    @property
    def context_feature_dim(self) -> int:
        """Get the context feature dimension."""
        return self._context_feature_dim


def create_event_graph_builder(
    max_players: int = 22,
    use_positions: bool = True,
    use_locations: bool = True,
) -> EventGraphBuilder:
    """
    Factory function to create an EventGraphBuilder with common settings.
    
    Args:
        max_players: Maximum players per event graph
        use_positions: Include position encodings
        use_locations: Include location features
        
    Returns:
        Configured EventGraphBuilder
    """
    config = EventGraphConfig(
        max_players_per_event=max_players,
        use_position_encoding=use_positions,
        use_location_features=use_locations,
        use_role_features=True,
    )
    return EventGraphBuilder(config)
