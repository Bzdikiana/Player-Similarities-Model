"""
Stats360 / StatsBomb Open Data Adapter

Converts Stats360/StatsBomb event data format to unified EventRecord objects.
Supports loading from:
- StatsBomb Open Data GitHub repository (default)
- Local JSON files
- API endpoints
"""

import json
import requests
import ssl
import urllib3
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np

from .base_adapter import BaseEventAdapter
from ..schema_contracts import (
    EventRecord, PlayerRef, MatchContext, Coordinates,
    PlayerRole, EVENT_TYPE_CATEGORIES
)

# Disable SSL warnings for corporate proxy environments
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class Stats360Adapter(BaseEventAdapter):
    """
    Adapter for Stats360/StatsBomb event data.
    
    This adapter handles the conversion from StatsBomb's nested JSON format
    to the unified EventRecord schema used by the embedding pipeline.
    
    Features:
    - Loads events, lineups, and match data
    - Extracts all players involved in each event (360 freeze frames)
    - Handles coordinate normalization
    - Supports both remote (GitHub) and local data sources
    
    Usage:
        adapter = Stats360Adapter()
        events = adapter.load_match_events(match_id=3788741)
    """
    
    # StatsBomb Open Data base URL
    STATSBOMB_BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        local_path: Optional[Path] = None,
        verify_ssl: bool = False,
    ):
        """
        Initialize the adapter.
        
        Args:
            base_url: Base URL for remote data (default: StatsBomb Open Data)
            local_path: Path to local data directory (if using local files)
            verify_ssl: Whether to verify SSL certificates (default: False for corporate networks)
        """
        self.base_url = base_url or self.STATSBOMB_BASE_URL
        self.local_path = Path(local_path) if local_path else None
        self.verify_ssl = verify_ssl
        
        # Cache for lineup data (player_id -> player_info)
        self._lineup_cache: Dict[int, Dict[int, Dict]] = {}
        
    def _fetch_json(self, url: str) -> Any:
        """Fetch JSON from URL with error handling."""
        try:
            response = requests.get(url, verify=self.verify_ssl, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise RuntimeError(f"Failed to fetch {url}: {e}")
    
    def _load_json_file(self, filepath: Path) -> Any:
        """Load JSON from local file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _get_data(self, path: str) -> Any:
        """Get data from either local file or remote URL."""
        if self.local_path:
            filepath = self.local_path / path
            return self._load_json_file(filepath)
        else:
            url = f"{self.base_url}/{path}"
            return self._fetch_json(url)
    
    def get_available_competitions(self) -> List[Dict[str, Any]]:
        """Load all available competitions."""
        return self._get_data("competitions.json")
    
    def load_matches(self, competition_id: int, season_id: int) -> List[Dict[str, Any]]:
        """Load all matches for a competition/season."""
        return self._get_data(f"matches/{competition_id}/{season_id}.json")
    
    def load_lineups(self, match_id: int) -> Dict[int, Dict]:
        """
        Load lineup data for a match and cache it.
        
        Returns:
            Dict mapping player_id -> player_info
        """
        if match_id in self._lineup_cache:
            return self._lineup_cache[match_id]
        
        try:
            lineups_data = self._get_data(f"lineups/{match_id}.json")
            
            player_map = {}
            for team in lineups_data:
                team_id = team.get("team_id")
                team_name = team.get("team_name", "")
                
                for player in team.get("lineup", []):
                    player_id = player.get("player_id")
                    player_map[player_id] = {
                        "player_id": player_id,
                        "player_name": player.get("player_name", ""),
                        "team_id": team_id,
                        "team_name": team_name,
                        "jersey_number": player.get("jersey_number"),
                        "positions": player.get("positions", []),
                    }
            
            self._lineup_cache[match_id] = player_map
            return player_map
            
        except Exception as e:
            print(f"Warning: Could not load lineups for match {match_id}: {e}")
            return {}
    
    def _extract_location(self, data: Any) -> Optional[Coordinates]:
        """Extract coordinates from StatsBomb location format."""
        if isinstance(data, (list, tuple)) and len(data) >= 2:
            try:
                return Coordinates(x=float(data[0]), y=float(data[1]))
            except (ValueError, TypeError):
                pass
        return None
    
    def _parse_player_ref(
        self,
        player_data: Dict,
        team_id: int,
        is_actor: bool = False,
        on_ball: bool = False,
        location: Optional[List[float]] = None,
        player_map: Dict[int, Dict] = None,
    ) -> Optional[PlayerRef]:
        """Parse a player reference from StatsBomb data."""
        if not player_data:
            return None
        
        player_id = player_data.get("id")
        if player_id is None:
            return None
        
        # Get additional info from lineup data
        lineup_info = player_map.get(player_id, {}) if player_map else {}
        
        return PlayerRef(
            player_id=player_id,
            team_id=team_id,
            player_name=player_data.get("name", lineup_info.get("player_name", "")),
            position=lineup_info.get("positions", [{}])[0].get("position", "") if lineup_info.get("positions") else "",
            role=PlayerRole.ACTOR if is_actor else PlayerRole.UNKNOWN,
            is_actor=is_actor,
            on_ball=on_ball,
            location=self._extract_location(location),
        )
    
    def _parse_freeze_frame(
        self,
        freeze_frame: List[Dict],
        actor_team_id: int,
        player_map: Dict[int, Dict] = None,
    ) -> List[PlayerRef]:
        """
        Parse 360 freeze frame data to extract all visible players.
        
        Args:
            freeze_frame: StatsBomb 360 freeze frame data
            actor_team_id: Team ID of the actor for determining teammates/opponents
            player_map: Player lookup from lineups
            
        Returns:
            List of PlayerRef for all visible players
        """
        players = []
        
        for ff_player in freeze_frame:
            player_id = ff_player.get("player", {}).get("id")
            if player_id is None:
                continue
            
            teammate = ff_player.get("teammate", False)
            actor = ff_player.get("actor", False)
            keeper = ff_player.get("keeper", False)
            
            # Determine team_id
            if teammate or actor:
                team_id = actor_team_id
                role = PlayerRole.TEAMMATE if not actor else PlayerRole.ACTOR
            else:
                team_id = -1  # Unknown opponent team ID
                role = PlayerRole.OPPONENT
            
            # Get lineup info
            lineup_info = player_map.get(player_id, {}) if player_map else {}
            
            player_ref = PlayerRef(
                player_id=player_id,
                team_id=lineup_info.get("team_id", team_id),
                player_name=ff_player.get("player", {}).get("name", lineup_info.get("player_name", "")),
                position=lineup_info.get("positions", [{}])[0].get("position", "") if lineup_info.get("positions") else "",
                role=role,
                is_actor=actor,
                on_ball=actor,  # Actor typically has the ball
                location=self._extract_location(ff_player.get("location")),
            )
            players.append(player_ref)
        
        return players
    
    def _parse_event(
        self,
        event_data: Dict,
        match_id: int,
        player_map: Dict[int, Dict] = None,
    ) -> Optional[EventRecord]:
        """
        Parse a single StatsBomb event into an EventRecord.
        
        Args:
            event_data: Raw StatsBomb event dictionary
            match_id: Match identifier
            player_map: Player lookup from lineups
            
        Returns:
            EventRecord or None if parsing fails
        """
        try:
            # Basic event info
            event_id = event_data.get("id", "")
            event_type_data = event_data.get("type", {})
            event_type = event_type_data.get("name", "Unknown") if event_type_data else "Unknown"
            
            # Skip certain event types
            skip_types = {"Starting XI", "Half Start", "Half End", "Substitution", "Player Off", "Player On"}
            if event_type in skip_types:
                return None
            
            # Context
            period = event_data.get("period", 1)
            minute = event_data.get("minute", 0)
            second = event_data.get("second", 0.0)
            
            # Team info
            team_data = event_data.get("team", {})
            team_id = team_data.get("id") if team_data else None
            
            # Get possession team
            possession_team_data = event_data.get("possession_team", {})
            possession_team_id = possession_team_data.get("id") if possession_team_data else None
            
            # Create match context
            context = MatchContext(
                match_id=match_id,
                period=period,
                minute=minute,
                second=second,
                score_home=0,  # Would need match data for this
                score_away=0,
                is_home_team=True,  # Would need match data for this
                possession_team_id=possession_team_id,
            )
            
            # Actor (primary player)
            player_data = event_data.get("player", {})
            position_data = event_data.get("position", {})
            position = position_data.get("name", "") if position_data else ""
            
            actor = None
            if player_data:
                lineup_info = player_map.get(player_data.get("id"), {}) if player_map else {}
                actor = PlayerRef(
                    player_id=player_data.get("id", 0),
                    team_id=team_id or 0,
                    player_name=player_data.get("name", ""),
                    position=position or (lineup_info.get("positions", [{}])[0].get("position", "") if lineup_info.get("positions") else ""),
                    role=PlayerRole.ACTOR,
                    is_actor=True,
                    on_ball=True,
                    location=self._extract_location(event_data.get("location")),
                )
            
            # All players from freeze frame (360 data)
            players = []
            freeze_frame = event_data.get("freeze_frame", [])
            if freeze_frame:
                players = self._parse_freeze_frame(freeze_frame, team_id or 0, player_map)
            elif actor:
                # No freeze frame, just add actor
                players = [actor]
            
            # Ensure actor is in players list
            if actor and not any(p.player_id == actor.player_id for p in players):
                players.insert(0, actor)
            
            # Locations
            location = self._extract_location(event_data.get("location"))
            
            # Get end location (varies by event type)
            end_location = None
            if event_type == "Pass":
                pass_data = event_data.get("pass", {})
                end_location = self._extract_location(pass_data.get("end_location"))
            elif event_type == "Carry":
                carry_data = event_data.get("carry", {})
                end_location = self._extract_location(carry_data.get("end_location"))
            elif event_type == "Shot":
                shot_data = event_data.get("shot", {})
                end_location = self._extract_location(shot_data.get("end_location"))
            
            # Outcome
            outcome = "Unknown"
            if event_type == "Pass":
                pass_data = event_data.get("pass", {})
                outcome_data = pass_data.get("outcome", {}) if pass_data else {}
                outcome = outcome_data.get("name", "Complete") if outcome_data else "Complete"
            elif event_type == "Shot":
                shot_data = event_data.get("shot", {})
                outcome_data = shot_data.get("outcome", {}) if shot_data else {}
                outcome = outcome_data.get("name", "Unknown") if outcome_data else "Unknown"
            elif event_type == "Duel":
                duel_data = event_data.get("duel", {})
                outcome_data = duel_data.get("outcome", {}) if duel_data else {}
                outcome = outcome_data.get("name", "Unknown") if outcome_data else "Unknown"
            
            # Tags (additional metadata)
            tags = {
                "play_pattern": event_data.get("play_pattern", {}).get("name", "") if event_data.get("play_pattern") else "",
                "possession": event_data.get("possession"),
                "under_pressure": event_data.get("under_pressure", False),
            }
            
            # Add event-specific tags
            if event_type == "Pass":
                pass_data = event_data.get("pass", {})
                if pass_data:
                    tags["pass_length"] = pass_data.get("length")
                    tags["pass_angle"] = pass_data.get("angle")
                    tags["pass_height"] = pass_data.get("height", {}).get("name", "") if pass_data.get("height") else ""
                    tags["pass_body_part"] = pass_data.get("body_part", {}).get("name", "") if pass_data.get("body_part") else ""
                    tags["pass_type"] = pass_data.get("type", {}).get("name", "") if pass_data.get("type") else ""
            
            return EventRecord(
                event_id=str(event_id),
                match_id=match_id,
                event_type=event_type,
                outcome=outcome,
                context=context,
                actor=actor,
                players=players,
                location=location,
                end_location=end_location,
                ball_location=location,  # Typically same as event location
                tags=tags,
                timestamp=event_data.get("timestamp"),
                possession=event_data.get("possession"),
                possession_team_id=possession_team_id,
            )
            
        except Exception as e:
            print(f"Warning: Failed to parse event: {e}")
            return None
    
    def load_match_events(self, match_id: int) -> List[EventRecord]:
        """
        Load all events for a match and convert to EventRecord objects.
        
        Args:
            match_id: Match identifier
            
        Returns:
            List of EventRecord objects sorted by timestamp
        """
        # Load lineup data for player lookup
        player_map = self.load_lineups(match_id)
        
        # Load raw events
        raw_events = self._get_data(f"events/{match_id}.json")
        
        # Parse events
        events = []
        for event_data in raw_events:
            event_record = self._parse_event(event_data, match_id, player_map)
            if event_record is not None:
                events.append(event_record)
        
        # Sort by match time
        events.sort(key=lambda e: (e.context.period, e.context.minute, e.context.second))
        
        return events
    
    def load_competition_events(
        self,
        competition_id: int,
        season_id: int,
        max_matches: Optional[int] = None,
        verbose: bool = True,
    ) -> List[EventRecord]:
        """
        Load all events for a competition/season.
        
        Args:
            competition_id: Competition identifier
            season_id: Season identifier
            max_matches: Maximum number of matches to load (None for all)
            verbose: Whether to print progress
            
        Returns:
            List of all EventRecord objects
        """
        matches = self.load_matches(competition_id, season_id)
        
        if max_matches:
            matches = matches[:max_matches]
        
        all_events = []
        
        for i, match in enumerate(matches):
            match_id = match.get("match_id")
            
            if verbose and (i + 1) % 10 == 0:
                print(f"Loading match {i + 1}/{len(matches)}...")
            
            try:
                events = self.load_match_events(match_id)
                all_events.extend(events)
            except Exception as e:
                if verbose:
                    print(f"Warning: Failed to load match {match_id}: {e}")
                continue
        
        if verbose:
            print(f"Loaded {len(all_events)} events from {len(matches)} matches")
        
        return all_events
    
    def get_player_events(
        self,
        events: List[EventRecord],
        player_id: int,
        as_actor_only: bool = True,
    ) -> List[EventRecord]:
        """
        Filter events for a specific player.
        
        Args:
            events: List of EventRecord objects
            player_id: Player identifier
            as_actor_only: If True, only return events where player is actor
            
        Returns:
            Filtered list of events
        """
        if as_actor_only:
            return [e for e in events if e.get_actor_player_id() == player_id]
        else:
            return [e for e in events if player_id in e.get_all_player_ids()]
    
    def get_unique_players(self, events: List[EventRecord]) -> Dict[int, Dict[str, Any]]:
        """
        Get unique players from events with metadata.
        
        Args:
            events: List of EventRecord objects
            
        Returns:
            Dict mapping player_id -> player metadata
        """
        players = {}
        
        for event in events:
            for player in event.players:
                if player.player_id not in players:
                    players[player.player_id] = {
                        "player_id": player.player_id,
                        "player_name": player.player_name,
                        "team_id": player.team_id,
                        "positions": set(),
                        "n_events": 0,
                        "n_events_as_actor": 0,
                    }
                
                players[player.player_id]["n_events"] += 1
                if player.is_actor:
                    players[player.player_id]["n_events_as_actor"] += 1
                if player.position:
                    players[player.player_id]["positions"].add(player.position)
        
        # Convert sets to lists
        for pid in players:
            players[pid]["positions"] = list(players[pid]["positions"])
        
        return players
