"""
Base adapter interface for event data sources.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Iterator
from ..schema_contracts import EventRecord, MatchContext


class BaseEventAdapter(ABC):
    """
    Abstract base class for event data adapters.
    
    All data source adapters should inherit from this class and implement
    the required methods to emit standardized EventRecord objects.
    """
    
    @abstractmethod
    def load_match_events(self, match_id: int) -> List[EventRecord]:
        """
        Load all events for a specific match.
        
        Args:
            match_id: Match identifier
            
        Returns:
            List of EventRecord objects sorted by timestamp
        """
        pass
    
    @abstractmethod
    def load_matches(self, competition_id: int, season_id: int) -> List[Dict[str, Any]]:
        """
        Load match metadata for a competition/season.
        
        Args:
            competition_id: Competition identifier
            season_id: Season identifier
            
        Returns:
            List of match metadata dictionaries
        """
        pass
    
    @abstractmethod
    def get_available_competitions(self) -> List[Dict[str, Any]]:
        """
        Get list of available competitions.
        
        Returns:
            List of competition metadata dictionaries
        """
        pass
    
    def iter_events(self, match_ids: List[int]) -> Iterator[EventRecord]:
        """
        Iterator over events from multiple matches.
        
        Args:
            match_ids: List of match identifiers
            
        Yields:
            EventRecord objects
        """
        for match_id in match_ids:
            try:
                events = self.load_match_events(match_id)
                for event in events:
                    yield event
            except Exception as e:
                print(f"Warning: Failed to load match {match_id}: {e}")
                continue
    
    def load_player_events(
        self, 
        match_ids: List[int], 
        player_id: int
    ) -> List[EventRecord]:
        """
        Load all events involving a specific player.
        
        Args:
            match_ids: List of match identifiers
            player_id: Player identifier
            
        Returns:
            List of EventRecord objects where player is involved
        """
        player_events = []
        
        for event in self.iter_events(match_ids):
            if player_id in event.get_all_player_ids():
                player_events.append(event)
        
        return player_events
    
    def get_match_context(self, match_metadata: Dict[str, Any]) -> MatchContext:
        """
        Extract match context from metadata.
        
        Override in subclasses if match metadata format differs.
        """
        return MatchContext(
            match_id=match_metadata.get("match_id", 0),
            period=1,
            minute=0,
            second=0,
            score_home=0,
            score_away=0,
        )
