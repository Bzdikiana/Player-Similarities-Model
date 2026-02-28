# ═══════════════════════════════════════════════════════════════════════════════
#                         API REFERENCE
#              Player Similarity Model v2.0
# ═══════════════════════════════════════════════════════════════════════════════

## Quick Reference

```python
from src.retrieval.api import SimilarityAPI

# Initialize with trained embeddings
api = SimilarityAPI(
    embeddings_path='outputs/enhanced_player_embeddings.pt',
    model_path='outputs/enhanced_model_weights.pt'
)

# Main methods
similar = api.find_similar("Lionel Messi", k=10)
score = api.compare("Lionel Messi", "Neymar")
```

---

## Table of Contents

1. [SimilarityAPI](#1-similarityapi)
2. [Data Classes](#2-data-classes)
3. [Index Classes](#3-index-classes)
4. [Examples](#4-examples)

---

## 1. SimilarityAPI

### 1.1 Initialization

```python
class SimilarityAPI:
    def __init__(
        self,
        embeddings_path: str,
        player_index: dict = None
    ):
        """
        Initialize the similarity API.
        
        Parameters
        ----------
        embeddings_path : str
            Path to saved player embeddings (.pt file)
        player_index : dict, optional
            Mapping of player_id -> player_name
        """
```

### 1.2 find_similar

```python
def find_similar(
    self,
    player_name: str,
    k: int = 10,
    position_group: str = None,
    exclude_same_team: bool = False
) -> List[Dict]:
    """
    Find the k most similar players to the given player.
    
    Parameters
    ----------
    player_name : str
        Name of the query player
    k : int, default=10
        Number of similar players to return
    position_group : str, optional
        Filter by position group: 'goalkeeper', 'center_back', 
        'full_back', 'defensive_mid', 'central_mid', 'winger', 'forward'
    exclude_same_team : bool, default=False
        Whether to exclude teammates
        
    Returns
    -------
    List[Dict]
        List of {player_name, similarity_score, position}
        
    Example
    -------
    >>> api.find_similar("Lionel Messi", k=5)
    [
        {'player': 'Neymar', 'similarity': 0.89, 'position': 'winger'},
        {'player': 'Mohamed Salah', 'similarity': 0.84, 'position': 'winger'},
        ...
    ]
    """
```

### 1.3 compare

```python
def compare(
    self,
    player1: str,
    player2: str
) -> float:
    """
    Get similarity score between two players.
    
    Parameters
    ----------
    player1, player2 : str
        Names of the players to compare
        
    Returns
    -------
    float
        Cosine similarity score between -1 and 1
        (higher = more similar)
        
    Example
    -------
    >>> api.compare("Lionel Messi", "Neymar")
    0.89
    """
```

### 1.4 get_embedding

```python
def get_embedding(self, player_name: str) -> torch.Tensor:
    """
    Get the raw 64-dimensional embedding for a player.
    
    Parameters
    ----------
    player_name : str
        Name of the player
        
    Returns
    -------
    torch.Tensor
        Shape [64] embedding vector
    """
```

---

## 2. Data Classes

### 2.1 Stats360Adapter

Located in `src/datasets/adapters/stats360_adapter.py`

```python
class Stats360Adapter:
    """
    Loads StatsBomb 360 data with freeze frames.
    """
    
    def __init__(self, verify_ssl: bool = True):
        """
        Parameters
        ----------
        verify_ssl : bool, default=True
            Whether to verify SSL certificates
        """
    
    def load_competitions(self) -> pd.DataFrame:
        """Load available competitions with 360 data."""
        
    def load_matches(
        self, 
        competition_id: int, 
        season_id: int
    ) -> pd.DataFrame:
        """Load matches for a competition/season."""
        
    def load_events_with_360(
        self, 
        match_id: int
    ) -> pd.DataFrame:
        """
        Load events merged with 360 freeze frame data.
        
        Returns DataFrame with columns:
        - Standard event columns (type, location, player, etc.)
        - freeze_frame: List of player positions
        - visible_area: Visible pitch polygon
        """
```

### 2.2 EventGraphBuilder

Located in `src/datasets/builders/event_graph_builder.py`

```python
class EventGraphBuilder:
    """
    Converts events into PyTorch Geometric graph objects.
    """
    
    def build_graph(self, event: dict) -> torch_geometric.data.Data:
        """
        Build graph from single event.
        
        Parameters
        ----------
        event : dict
            Event dict with 'freeze_frame' containing player positions
            
        Returns
        -------
        Data
            PyG Data object with:
            - x: Node features [num_nodes, node_dim]
            - edge_index: Edge connectivity [2, num_edges]
            - edge_attr: Edge features [num_edges, edge_dim]
            - actor_mask: Which node is the on-ball player
        """
```

---

## 3. Index Classes

### 3.1 PlayerIndex

Located in `src/retrieval/index.py`

```python
class PlayerIndex:
    """
    FAISS-based index for fast similarity search.
    """
    
    def __init__(
        self,
        embeddings: torch.Tensor,
        player_ids: List[str]
    ):
        """
        Parameters
        ----------
        embeddings : torch.Tensor
            Shape [num_players, 64] normalized embeddings
        player_ids : List[str]
            Player identifiers matching embedding rows
        """
    
    def search(
        self,
        query: torch.Tensor,
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find k nearest neighbors.
        
        Returns
        -------
        List[Tuple[str, float]]
            List of (player_id, similarity_score)
        """
```

---

## 4. Examples

### 4.1 Find Similar Players

```python
from src.retrieval.api import SimilarityAPI

# Load API
api = SimilarityAPI('outputs/enhanced_player_embeddings.pt')

# Find players similar to Messi
similar = api.find_similar("Lionel Messi", k=10)
for player in similar:
    print(f"{player['player']}: {player['similarity']:.2f}")
    
# Output:
# Neymar: 0.89
# Mohamed Salah: 0.84
# Kylian Mbappé: 0.82
# ...
```

### 4.2 Compare Two Players

```python
# Direct comparison
score = api.compare("Kevin De Bruyne", "Bruno Fernandes")
print(f"Similarity: {score:.2f}")  # 0.76
```

### 4.3 Position-Filtered Search

```python
# Find similar center backs only
similar_cbs = api.find_similar(
    "Virgil van Dijk",
    k=10,
    position_group='center_back'
)
```

### 4.4 Get Raw Embeddings

```python
# Get embedding for custom analysis
messi_emb = api.get_embedding("Lionel Messi")
neymar_emb = api.get_embedding("Neymar")

# Custom similarity
import torch.nn.functional as F
similarity = F.cosine_similarity(messi_emb, neymar_emb, dim=0)
```

---

## Position Groups

When using `position_group` parameter, use one of:

| Group | Description |
|-------|-------------|
| `'goalkeeper'` | Goalkeepers only |
| `'center_back'` | CB, RCB, LCB |
| `'full_back'` | RB, LB, RWB, LWB |
| `'defensive_mid'` | CDM, RDM, LDM |
| `'central_mid'` | CM, RM, LM |
| `'winger'` | RW, LW |
| `'forward'` | CF, ST, RCF, LCF |

---

**Author:** Armen Bzdikian  
**Contact:** bzdikiana11@gmail.com  
**Last Updated:** February 2026
