"""
Tests for schema contracts.
"""

import pytest
import numpy as np
from datetime import datetime

import sys
sys.path.append('../..')

from src.datasets.schema_contracts import (
    EventRecord,
    PlayerRef,
    MatchContext,
    Coordinates,
    PlayerRole,
    EVENT_TYPE_CATEGORIES,
    POSITION_CATEGORIES,
    get_event_type_index,
    get_position_index,
)


class TestCoordinates:
    """Tests for Coordinates dataclass."""
    
    def test_create_coordinates(self):
        """Test basic coordinate creation."""
        coords = Coordinates(x=60.0, y=40.0)
        assert coords.x == 60.0
        assert coords.y == 40.0
    
    def test_coordinates_to_array(self):
        """Test conversion to numpy array."""
        coords = Coordinates(x=60.0, y=40.0)
        arr = coords.to_array()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (2,)
        assert arr[0] == 60.0
        assert arr[1] == 40.0
    
    def test_coordinates_normalized(self):
        """Test normalized coordinates."""
        coords = Coordinates(x=120.0, y=80.0)
        norm = coords.normalized()
        assert norm[0] == 1.0
        assert norm[1] == 1.0
        
        coords2 = Coordinates(x=0.0, y=0.0)
        norm2 = coords2.normalized()
        assert norm2[0] == 0.0
        assert norm2[1] == 0.0


class TestPlayerRef:
    """Tests for PlayerRef dataclass."""
    
    def test_create_player_ref(self):
        """Test basic player reference creation."""
        player = PlayerRef(
            player_id=12345,
            player_name="Lionel Messi",
            position="Right Wing",
            team_id=1,
            team_name="Barcelona",
        )
        assert player.player_id == 12345
        assert player.player_name == "Lionel Messi"
        assert player.position == "Right Wing"
    
    def test_player_ref_with_freeze_frame(self):
        """Test player ref with freeze frame location."""
        player = PlayerRef(
            player_id=12345,
            player_name="Test Player",
            position="Midfielder",
            team_id=1,
            location=Coordinates(50.0, 30.0),
            is_teammate=True,
            is_actor=False,
        )
        assert player.location is not None
        assert player.is_teammate is True
        assert player.is_actor is False


class TestMatchContext:
    """Tests for MatchContext dataclass."""
    
    def test_create_match_context(self):
        """Test basic match context creation."""
        ctx = MatchContext(
            match_id=12345,
            period=1,
            minute=30,
            second=45,
            home_team="Barcelona",
            away_team="Real Madrid",
            home_score=1,
            away_score=0,
        )
        assert ctx.match_id == 12345
        assert ctx.period == 1
        assert ctx.minute == 30
    
    def test_match_context_score_diff(self):
        """Test score difference calculation."""
        ctx = MatchContext(
            match_id=1,
            period=1,
            minute=0,
            second=0,
            home_team="A",
            away_team="B",
            home_score=3,
            away_score=1,
        )
        assert ctx.score_diff == 2
        
        ctx2 = MatchContext(
            match_id=1,
            period=1,
            minute=0,
            second=0,
            home_team="A",
            away_team="B",
            home_score=0,
            away_score=2,
        )
        assert ctx2.score_diff == -2
    
    def test_match_context_to_tensor(self):
        """Test conversion to tensor."""
        ctx = MatchContext(
            match_id=1,
            period=2,
            minute=60,
            second=30,
            home_team="A",
            away_team="B",
            home_score=1,
            away_score=1,
        )
        tensor = ctx.to_tensor()
        assert len(tensor) >= 4  # At least period, minute, score_diff, ...


class TestEventRecord:
    """Tests for EventRecord dataclass."""
    
    def test_create_event_record(self):
        """Test basic event record creation."""
        event = EventRecord(
            event_id="abc123",
            event_type="Pass",
            actor=PlayerRef(
                player_id=1,
                player_name="Player 1",
                position="Midfielder",
                team_id=1,
            ),
            location=Coordinates(50.0, 40.0),
            match_context=MatchContext(
                match_id=1,
                period=1,
                minute=10,
                second=0,
                home_team="A",
                away_team="B",
                home_score=0,
                away_score=0,
            ),
        )
        assert event.event_id == "abc123"
        assert event.event_type == "Pass"
        assert event.actor.player_name == "Player 1"
    
    def test_event_record_with_freeze_frame(self):
        """Test event record with freeze frame."""
        freeze_frame = [
            PlayerRef(player_id=i, player_name=f"P{i}", position="MF", team_id=1)
            for i in range(10)
        ]
        
        event = EventRecord(
            event_id="def456",
            event_type="Shot",
            actor=PlayerRef(player_id=1, player_name="Shooter", position="ST", team_id=1),
            location=Coordinates(100.0, 40.0),
            match_context=MatchContext(
                match_id=1, period=2, minute=75, second=0,
                home_team="A", away_team="B", home_score=2, away_score=1,
            ),
            freeze_frame=freeze_frame,
        )
        assert len(event.freeze_frame) == 10


class TestCategories:
    """Tests for event type and position categories."""
    
    def test_event_type_categories(self):
        """Test event type categories exist."""
        assert len(EVENT_TYPE_CATEGORIES) > 0
        assert "Pass" in EVENT_TYPE_CATEGORIES
        assert "Shot" in EVENT_TYPE_CATEGORIES
        assert "Dribble" in EVENT_TYPE_CATEGORIES
    
    def test_position_categories(self):
        """Test position categories exist."""
        assert len(POSITION_CATEGORIES) > 0
        assert "Goalkeeper" in POSITION_CATEGORIES
        assert "Center Midfield" in POSITION_CATEGORIES
    
    def test_get_event_type_index(self):
        """Test event type to index conversion."""
        idx = get_event_type_index("Pass")
        assert isinstance(idx, int)
        assert idx >= 0
        
        # Unknown type should return last index
        unknown_idx = get_event_type_index("UnknownType")
        assert unknown_idx == len(EVENT_TYPE_CATEGORIES) - 1
    
    def test_get_position_index(self):
        """Test position to index conversion."""
        idx = get_position_index("Goalkeeper")
        assert isinstance(idx, int)
        assert idx >= 0
        
        # Unknown position should return last index
        unknown_idx = get_position_index("UnknownPosition")
        assert unknown_idx == len(POSITION_CATEGORIES) - 1


class TestPlayerRole:
    """Tests for PlayerRole enum."""
    
    def test_player_roles(self):
        """Test player role enum values."""
        assert PlayerRole.GOALKEEPER.value == 0
        assert PlayerRole.DEFENDER.value == 1
        assert PlayerRole.MIDFIELDER.value == 2
        assert PlayerRole.FORWARD.value == 3
    
    def test_role_from_position(self):
        """Test role inference from position."""
        # This would test a helper function if implemented
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
