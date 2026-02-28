"""
Player Similarity API

High-level API for finding similar players:
- get_similar_players()
- Filtering by position, league, age, etc.
- Integration with embedding store
"""

import torch
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field

from .similarity import SimilaritySearch, SimilarityResult, CosineSimilarity
from .index import EmbeddingIndex, EmbeddingStore


@dataclass
class PlayerFilter:
    """Filters for similarity search."""
    
    # Role/position filters
    roles: Optional[List[int]] = None  # Only these roles
    same_role: bool = False  # Only same role as query
    
    # Data quality filters
    min_reliability: Optional[float] = None
    min_events: Optional[int] = None
    
    # Player attribute filters
    leagues: Optional[List[str]] = None
    teams: Optional[List[str]] = None
    nationalities: Optional[List[str]] = None
    min_age: Optional[int] = None
    max_age: Optional[int] = None
    
    # Exclusions
    exclude_same_team: bool = False
    exclude_player_ids: List[int] = field(default_factory=list)


@dataclass
class SimilarPlayer:
    """A similar player result."""
    
    player_id: int
    player_name: Optional[str]
    similarity_score: float
    rank: int
    
    # Optional attributes
    position: Optional[str] = None
    team: Optional[str] = None
    league: Optional[str] = None
    age: Optional[int] = None
    nationality: Optional[str] = None
    reliability: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "player_id": self.player_id,
            "player_name": self.player_name,
            "similarity_score": self.similarity_score,
            "rank": self.rank,
            "position": self.position,
            "team": self.team,
            "league": self.league,
            "age": self.age,
            "nationality": self.nationality,
            "reliability": self.reliability,
        }


class PlayerSimilarityAPI:
    """
    High-level API for player similarity queries.
    
    Usage:
        api = PlayerSimilarityAPI(embedding_store_path="embeddings")
        similar = api.get_similar_players(
            player_id=12345,
            k=10,
            filters=PlayerFilter(same_role=True)
        )
    """
    
    def __init__(
        self,
        embedding_store_path: str = "embeddings",
        player_metadata: Optional[Dict[int, Dict]] = None,
        default_version: Optional[str] = None,
    ):
        """
        Args:
            embedding_store_path: Path to embedding store
            player_metadata: Dict mapping player_id -> metadata
            default_version: Default embedding version to use
        """
        self.store = EmbeddingStore(embedding_store_path)
        self.player_metadata = player_metadata or {}
        self.default_version = default_version
        
        # Lazy-loaded index
        self._index: Optional[EmbeddingIndex] = None
        self._loaded_version: Optional[str] = None
    
    def _ensure_index(self, version: Optional[str] = None):
        """Load index if not already loaded."""
        version = version or self.default_version
        
        if self._index is not None and self._loaded_version == version:
            return
        
        embeddings, metadata = self.store.load(version)
        
        self._index = EmbeddingIndex(embeddings)
        self._loaded_version = version
    
    def get_similar_players(
        self,
        player_id: int,
        k: int = 10,
        filters: Optional[PlayerFilter] = None,
        version: Optional[str] = None,
        include_scores: bool = True,
    ) -> List[SimilarPlayer]:
        """
        Find similar players.
        
        Args:
            player_id: Query player ID
            k: Number of results
            filters: Optional filters to apply
            version: Embedding version to use
            include_scores: Whether to include similarity scores
            
        Returns:
            List of SimilarPlayer results
        """
        self._ensure_index(version)
        
        filters = filters or PlayerFilter()
        
        # Get more candidates than k to account for filtering
        search_k = k * 3
        
        result_ids, result_scores = self._index.query_by_id(
            player_id, search_k, exclude_self=True
        )
        
        # Apply filters
        results = []
        query_metadata = self.player_metadata.get(player_id, {})
        
        for pid, score in zip(result_ids, result_scores):
            meta = self.player_metadata.get(pid, {})
            
            # Check filters
            if not self._passes_filters(pid, meta, query_metadata, filters):
                continue
            
            # Create result
            result = SimilarPlayer(
                player_id=pid,
                player_name=meta.get("name"),
                similarity_score=score if include_scores else 0.0,
                rank=len(results) + 1,
                position=meta.get("position"),
                team=meta.get("team"),
                league=meta.get("league"),
                age=meta.get("age"),
                nationality=meta.get("nationality"),
                reliability=meta.get("reliability"),
            )
            results.append(result)
            
            if len(results) >= k:
                break
        
        return results
    
    def _passes_filters(
        self,
        player_id: int,
        metadata: Dict,
        query_metadata: Dict,
        filters: PlayerFilter,
    ) -> bool:
        """Check if a player passes all filters."""
        
        # Exclusions
        if player_id in filters.exclude_player_ids:
            return False
        
        if filters.exclude_same_team:
            if metadata.get("team") == query_metadata.get("team"):
                return False
        
        # Role filters
        if filters.same_role:
            if metadata.get("role") != query_metadata.get("role"):
                return False
        
        if filters.roles is not None:
            if metadata.get("role") not in filters.roles:
                return False
        
        # Quality filters
        if filters.min_reliability is not None:
            if metadata.get("reliability", 0) < filters.min_reliability:
                return False
        
        if filters.min_events is not None:
            if metadata.get("n_events", 0) < filters.min_events:
                return False
        
        # Attribute filters
        if filters.leagues is not None:
            if metadata.get("league") not in filters.leagues:
                return False
        
        if filters.teams is not None:
            if metadata.get("team") not in filters.teams:
                return False
        
        if filters.nationalities is not None:
            if metadata.get("nationality") not in filters.nationalities:
                return False
        
        if filters.min_age is not None:
            age = metadata.get("age")
            if age is None or age < filters.min_age:
                return False
        
        if filters.max_age is not None:
            age = metadata.get("age")
            if age is None or age > filters.max_age:
                return False
        
        return True
    
    def get_player_embedding(
        self,
        player_id: int,
        version: Optional[str] = None,
    ) -> Optional[torch.Tensor]:
        """Get the embedding for a player."""
        self._ensure_index(version)
        return self._index.get_embedding(player_id)
    
    def compare_players(
        self,
        player_id_1: int,
        player_id_2: int,
        version: Optional[str] = None,
    ) -> float:
        """
        Get similarity score between two specific players.
        
        Returns:
            Cosine similarity score
        """
        self._ensure_index(version)
        
        emb1 = self._index.get_embedding(player_id_1)
        emb2 = self._index.get_embedding(player_id_2)
        
        if emb1 is None or emb2 is None:
            raise ValueError("One or both players not in index")
        
        similarity = torch.dot(emb1, emb2).item()
        return similarity
    
    def find_player_archetype(
        self,
        player_id: int,
        n_similar: int = 50,
        version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a player's archetype based on similar players.
        
        Returns common characteristics of similar players.
        """
        similar = self.get_similar_players(player_id, k=n_similar, version=version)
        
        # Aggregate metadata
        positions = {}
        teams = {}
        leagues = {}
        
        for player in similar:
            if player.position:
                positions[player.position] = positions.get(player.position, 0) + 1
            if player.team:
                teams[player.team] = teams.get(player.team, 0) + 1
            if player.league:
                leagues[player.league] = leagues.get(player.league, 0) + 1
        
        # Sort by frequency
        positions = sorted(positions.items(), key=lambda x: -x[1])
        teams = sorted(teams.items(), key=lambda x: -x[1])
        leagues = sorted(leagues.items(), key=lambda x: -x[1])
        
        return {
            "player_id": player_id,
            "n_analyzed": len(similar),
            "top_positions": positions[:5],
            "top_teams": teams[:5],
            "top_leagues": leagues[:5],
            "avg_similarity": sum(p.similarity_score for p in similar) / len(similar) if similar else 0,
        }
    
    def list_versions(self) -> List[Dict]:
        """List available embedding versions."""
        return self.store.list_versions()
    
    def set_metadata(self, player_metadata: Dict[int, Dict]):
        """Update player metadata."""
        self.player_metadata = player_metadata


def get_similar_players(
    player_id: int,
    k: int = 10,
    embedding_path: str = "embeddings",
    version: Optional[str] = None,
    **filter_kwargs,
) -> List[Dict]:
    """
    Convenience function for finding similar players.
    
    Args:
        player_id: Query player ID
        k: Number of results
        embedding_path: Path to embedding store
        version: Embedding version
        **filter_kwargs: Filter arguments
        
    Returns:
        List of similar player dicts
    """
    api = PlayerSimilarityAPI(embedding_path)
    
    filters = None
    if filter_kwargs:
        filters = PlayerFilter(**filter_kwargs)
    
    results = api.get_similar_players(player_id, k, filters, version)
    return [r.to_dict() for r in results]
