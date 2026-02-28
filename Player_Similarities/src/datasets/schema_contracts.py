"""
Schema Contracts for Stats360 Event Data

Defines unified dataclasses for event processing pipeline:
- EventRecord: Single event with all relevant attributes
- PlayerRef: Player reference within an event
- MatchContext: Match-level context (score, home/away, etc.)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import numpy as np


class PlayerRole(Enum):
    """Role of player within an event."""
    ACTOR = "actor"           # Player performing the action
    TEAMMATE = "teammate"     # Teammate involved/nearby
    OPPONENT = "opponent"     # Opponent involved/nearby
    UNKNOWN = "unknown"


@dataclass
class Coordinates:
    """2D pitch coordinates (StatsBomb: 120x80 standard)."""
    x: float
    y: float
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)
    
    def is_valid(self) -> bool:
        return 0 <= self.x <= 120 and 0 <= self.y <= 80
    
    @classmethod
    def from_list(cls, coords: List[float]) -> Optional["Coordinates"]:
        if coords and len(coords) >= 2:
            return cls(x=float(coords[0]), y=float(coords[1]))
        return None


@dataclass
class PlayerRef:
    """
    Reference to a player involved in an event.
    
    Attributes:
        player_id: Unique player identifier
        team_id: Team identifier
        player_name: Full name (for display)
        position: Position string (e.g., "Left Wing", "Center Back")
        role: Role in this specific event (actor/teammate/opponent)
        is_actor: True if this player is the primary actor
        on_ball: True if player has/had the ball
        location: Player's coordinates at event time (if available)
    """
    player_id: int
    team_id: int
    player_name: str = ""
    position: str = ""
    role: PlayerRole = PlayerRole.UNKNOWN
    is_actor: bool = False
    on_ball: bool = False
    location: Optional[Coordinates] = None
    
    def to_feature_dict(self) -> Dict[str, Any]:
        """Convert to feature dictionary for model input."""
        return {
            "player_id": self.player_id,
            "team_id": self.team_id,
            "is_actor": int(self.is_actor),
            "on_ball": int(self.on_ball),
            "role": self.role.value,
            "position": self.position,
            "location_x": self.location.x if self.location else 0.0,
            "location_y": self.location.y if self.location else 0.0,
            "has_location": int(self.location is not None),
        }


@dataclass
class MatchContext:
    """
    Match-level context for an event.
    
    Used for FiLM conditioning in the event GNN.
    """
    match_id: int
    period: int                    # 1, 2, 3 (ET1), 4 (ET2), 5 (PK)
    minute: int
    second: float
    score_home: int = 0
    score_away: int = 0
    is_home_team: bool = True      # Is the actor's team the home team?
    possession_team_id: Optional[int] = None
    
    @property
    def score_diff(self) -> int:
        """Score differential from actor's perspective."""
        if self.is_home_team:
            return self.score_home - self.score_away
        return self.score_away - self.score_home
    
    @property
    def match_time_normalized(self) -> float:
        """Normalized match time (0-1 for regular time, >1 for extra time)."""
        total_seconds = (self.minute * 60) + self.second
        return total_seconds / (90 * 60)  # Normalize to 90 mins
    
    def to_context_vector(self) -> np.ndarray:
        """
        Convert context to feature vector for FiLM conditioning.
        
        Returns:
            np.ndarray: [period_onehot(5), minute_norm, score_diff_norm, is_home]
        """
        # Period one-hot (5 possible periods)
        period_onehot = np.zeros(5, dtype=np.float32)
        if 1 <= self.period <= 5:
            period_onehot[self.period - 1] = 1.0
        
        # Normalized minute (0-1 for 0-90, can exceed for extra time)
        minute_norm = self.minute / 90.0
        
        # Score diff (clipped and normalized)
        score_diff_norm = np.clip(self.score_diff, -5, 5) / 5.0
        
        return np.concatenate([
            period_onehot,
            [minute_norm, score_diff_norm, float(self.is_home_team)]
        ]).astype(np.float32)


@dataclass
class EventRecord:
    """
    Unified event record for the embedding pipeline.
    
    This is the primary data contract between data adapters and model builders.
    Each event represents a single action (pass, shot, tackle, etc.) in a match.
    
    Attributes:
        event_id: Unique event identifier
        match_id: Match identifier
        event_type: Type of event (Pass, Shot, Tackle, etc.)
        outcome: Outcome of the event (Success, Fail, etc.)
        
        context: Match context (period, minute, score, etc.)
        
        actor: Primary player performing the action
        players: All players involved (actor + teammates + opponents)
        
        location: Event location on pitch
        end_location: End location (for passes, carries)
        
        tags: Optional additional tags (speed, zone, etc.)
    """
    # Identifiers
    event_id: str
    match_id: int
    
    # Event info
    event_type: str
    outcome: str = "Unknown"
    
    # Context
    context: MatchContext = field(default_factory=lambda: MatchContext(
        match_id=0, period=1, minute=0, second=0
    ))
    
    # Players involved
    actor: Optional[PlayerRef] = None
    players: List[PlayerRef] = field(default_factory=list)
    
    # Spatial
    location: Optional[Coordinates] = None
    end_location: Optional[Coordinates] = None
    ball_location: Optional[Coordinates] = None
    
    # Optional metadata
    tags: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[str] = None
    possession: Optional[int] = None
    possession_team_id: Optional[int] = None
    
    def get_actor_player_id(self) -> Optional[int]:
        """Get the actor's player ID."""
        return self.actor.player_id if self.actor else None
    
    def get_all_player_ids(self) -> List[int]:
        """Get all player IDs involved in event."""
        return [p.player_id for p in self.players]
    
    def get_teammates(self) -> List[PlayerRef]:
        """Get all teammates (excluding actor)."""
        if not self.actor:
            return []
        return [p for p in self.players 
                if p.team_id == self.actor.team_id and not p.is_actor]
    
    def get_opponents(self) -> List[PlayerRef]:
        """Get all opponents."""
        if not self.actor:
            return self.players
        return [p for p in self.players if p.team_id != self.actor.team_id]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "match_id": self.match_id,
            "event_type": self.event_type,
            "outcome": self.outcome,
            "period": self.context.period,
            "minute": self.context.minute,
            "second": self.context.second,
            "location_x": self.location.x if self.location else None,
            "location_y": self.location.y if self.location else None,
            "end_location_x": self.end_location.x if self.end_location else None,
            "end_location_y": self.end_location.y if self.end_location else None,
            "actor_player_id": self.get_actor_player_id(),
            "n_players": len(self.players),
            "tags": self.tags,
        }


# Type alias for a sequence of events
EventSequence = List[EventRecord]


# Event type categories for encoding
EVENT_TYPE_CATEGORIES = [
    "Pass", "Ball Receipt*", "Carry", "Shot", "Duel", "Clearance",
    "Interception", "Block", "Pressure", "Tackle", "Foul Committed",
    "Foul Won", "Ball Recovery", "Dispossessed", "Miscontrol",
    "Goal Keeper", "50/50", "Shield", "Dribble", "Dribbled Past",
    "Injury Stoppage", "Referee Ball-Drop", "Bad Behaviour", "Own Goal Against",
    "Player On", "Player Off", "Error", "Half Start", "Half End",
    "Starting XI", "Substitution", "Tactical Shift", "Own Goal For"
]

# Outcome categories
OUTCOME_CATEGORIES = [
    "Complete", "Incomplete", "Success", "Won", "Lost",
    "Saved", "Goal", "Off Target", "Blocked", "Unknown"
]

# Position categories (StatsBomb)
POSITION_CATEGORIES = [
    "Goalkeeper", "Right Back", "Right Center Back", "Center Back",
    "Left Center Back", "Left Back", "Right Wing Back", "Left Wing Back",
    "Right Defensive Midfield", "Center Defensive Midfield", "Left Defensive Midfield",
    "Right Midfield", "Right Center Midfield", "Center Midfield",
    "Left Center Midfield", "Left Midfield", "Right Wing", "Right Attacking Midfield",
    "Center Attacking Midfield", "Left Attacking Midfield", "Left Wing",
    "Right Center Forward", "Striker", "Left Center Forward", "Secondary Striker"
]


def get_event_type_index(event_type: str) -> int:
    """Get index for event type encoding."""
    try:
        return EVENT_TYPE_CATEGORIES.index(event_type)
    except ValueError:
        return len(EVENT_TYPE_CATEGORIES)  # Unknown type


def get_position_index(position: str) -> int:
    """Get index for position encoding."""
    try:
        return POSITION_CATEGORIES.index(position)
    except ValueError:
        return len(POSITION_CATEGORIES)  # Unknown position
