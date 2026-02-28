"""
Embedding Index

Manages storage and versioning of player embeddings:
1. Save/load embeddings to disk
2. Version tracking by model
3. Efficient retrieval
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import json
import hashlib


class EmbeddingStore:
    """
    Simple embedding storage with versioning.
    
    Stores embeddings organized by model version:
    
    store_dir/
        v1_20240101/
            embeddings.pt
            metadata.json
        v2_20240115/
            embeddings.pt
            metadata.json
    """
    
    def __init__(self, store_dir: str = "embeddings"):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache loaded embeddings
        self._cache: Dict[str, Dict] = {}
    
    def save(
        self,
        embeddings: Dict[int, torch.Tensor],
        version: str,
        metadata: Optional[Dict] = None,
    ) -> Path:
        """
        Save embeddings with version tag.
        
        Args:
            embeddings: Dict mapping player_id -> embedding
            version: Version string (e.g., 'v1', 'model_20240101')
            metadata: Additional metadata to store
            
        Returns:
            Path to saved directory
        """
        # Create version directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_dir = self.store_dir / f"{version}_{timestamp}"
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to tensors for efficient storage
        player_ids = list(embeddings.keys())
        embedding_matrix = torch.stack([embeddings[pid] for pid in player_ids])
        
        # Save embeddings
        torch.save({
            "player_ids": player_ids,
            "embeddings": embedding_matrix,
        }, version_dir / "embeddings.pt")
        
        # Save metadata
        meta = {
            "version": version,
            "timestamp": timestamp,
            "n_players": len(player_ids),
            "embedding_dim": embedding_matrix.shape[1],
        }
        if metadata:
            meta.update(metadata)
        
        with open(version_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        
        return version_dir
    
    def load(
        self,
        version: Optional[str] = None,
        use_cache: bool = True,
    ) -> Tuple[Dict[int, torch.Tensor], Dict]:
        """
        Load embeddings by version.
        
        Args:
            version: Version to load (latest if None)
            use_cache: Whether to use cached embeddings
            
        Returns:
            (embeddings dict, metadata dict)
        """
        # Find version directory
        if version is None:
            # Get latest
            version_dirs = sorted(self.store_dir.iterdir())
            if not version_dirs:
                raise FileNotFoundError("No saved embeddings found")
            version_dir = version_dirs[-1]
        else:
            # Find matching version
            matching = [d for d in self.store_dir.iterdir() if d.name.startswith(version)]
            if not matching:
                raise FileNotFoundError(f"No embeddings found for version: {version}")
            version_dir = sorted(matching)[-1]  # Latest of this version
        
        version_key = version_dir.name
        
        # Check cache
        if use_cache and version_key in self._cache:
            return self._cache[version_key]
        
        # Load
        data = torch.load(version_dir / "embeddings.pt")
        player_ids = data["player_ids"]
        embedding_matrix = data["embeddings"]
        
        embeddings = {
            pid: embedding_matrix[i]
            for i, pid in enumerate(player_ids)
        }
        
        with open(version_dir / "metadata.json") as f:
            metadata = json.load(f)
        
        result = (embeddings, metadata)
        
        if use_cache:
            self._cache[version_key] = result
        
        return result
    
    def list_versions(self) -> List[Dict]:
        """List all available versions."""
        versions = []
        
        for version_dir in sorted(self.store_dir.iterdir()):
            if not version_dir.is_dir():
                continue
            
            meta_path = version_dir / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                meta["path"] = str(version_dir)
                versions.append(meta)
        
        return versions
    
    def delete_version(self, version: str):
        """Delete a specific version."""
        import shutil
        
        matching = [d for d in self.store_dir.iterdir() if d.name.startswith(version)]
        for d in matching:
            shutil.rmtree(d)
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()


class EmbeddingIndex:
    """
    Efficient embedding index for similarity search.
    
    Provides fast nearest-neighbor lookup using:
    - Brute force (default, exact)
    - FAISS (optional, approximate but fast for large N)
    """
    
    def __init__(
        self,
        embeddings: Optional[Dict[int, torch.Tensor]] = None,
        use_faiss: bool = False,
    ):
        """
        Args:
            embeddings: Initial embeddings to index
            use_faiss: Whether to use FAISS for indexing
        """
        self.use_faiss = use_faiss
        self.faiss_index = None
        
        self.player_ids: List[int] = []
        self.embedding_matrix: Optional[torch.Tensor] = None
        self.metadata: Dict[int, Dict] = {}
        
        if embeddings:
            self.build(embeddings)
    
    def build(
        self,
        embeddings: Dict[int, torch.Tensor],
        metadata: Optional[Dict[int, Dict]] = None,
    ):
        """
        Build the index from embeddings.
        
        Args:
            embeddings: Dict mapping player_id -> embedding
            metadata: Optional metadata for each player
        """
        self.player_ids = list(embeddings.keys())
        self.embedding_matrix = torch.stack([embeddings[pid] for pid in self.player_ids])
        self.embedding_matrix = torch.nn.functional.normalize(self.embedding_matrix, p=2, dim=-1)
        
        if metadata:
            self.metadata = metadata
        
        # Build FAISS index if requested
        if self.use_faiss:
            self._build_faiss_index()
    
    def _build_faiss_index(self):
        """Build FAISS index for approximate NN search."""
        try:
            import faiss
            
            d = self.embedding_matrix.shape[1]
            
            # Use inner product (cosine on normalized vectors)
            self.faiss_index = faiss.IndexFlatIP(d)
            
            # Add vectors
            embeddings_np = self.embedding_matrix.numpy().astype(np.float32)
            self.faiss_index.add(embeddings_np)
            
        except ImportError:
            print("FAISS not available, falling back to brute force")
            self.use_faiss = False
    
    def query(
        self,
        query_embedding: torch.Tensor,
        k: int = 10,
    ) -> Tuple[List[int], List[float]]:
        """
        Find k nearest neighbors.
        
        Args:
            query_embedding: [D] query vector
            k: Number of results
            
        Returns:
            (player_ids, scores)
        """
        if self.embedding_matrix is None:
            raise ValueError("Index not built")
        
        # Normalize query
        query_embedding = torch.nn.functional.normalize(
            query_embedding.unsqueeze(0), p=2, dim=-1
        ).squeeze(0)
        
        if self.use_faiss and self.faiss_index is not None:
            # FAISS search
            query_np = query_embedding.numpy().astype(np.float32).reshape(1, -1)
            scores, indices = self.faiss_index.search(query_np, k)
            
            result_ids = [self.player_ids[i] for i in indices[0]]
            result_scores = scores[0].tolist()
        else:
            # Brute force
            scores = torch.matmul(self.embedding_matrix, query_embedding)
            top_scores, top_indices = torch.topk(scores, min(k, len(scores)))
            
            result_ids = [self.player_ids[i] for i in top_indices.tolist()]
            result_scores = top_scores.tolist()
        
        return result_ids, result_scores
    
    def query_by_id(
        self,
        player_id: int,
        k: int = 10,
        exclude_self: bool = True,
    ) -> Tuple[List[int], List[float]]:
        """
        Find similar players by ID.
        
        Args:
            player_id: Query player ID
            k: Number of results
            exclude_self: Whether to exclude query player
            
        Returns:
            (player_ids, scores)
        """
        if player_id not in self.player_ids:
            raise ValueError(f"Player {player_id} not in index")
        
        idx = self.player_ids.index(player_id)
        query_emb = self.embedding_matrix[idx]
        
        # Get k+1 if excluding self
        search_k = k + 1 if exclude_self else k
        result_ids, result_scores = self.query(query_emb, search_k)
        
        if exclude_self and player_id in result_ids:
            idx = result_ids.index(player_id)
            result_ids.pop(idx)
            result_scores.pop(idx)
        
        return result_ids[:k], result_scores[:k]
    
    def add(self, player_id: int, embedding: torch.Tensor):
        """Add a single player to the index."""
        embedding = torch.nn.functional.normalize(embedding.unsqueeze(0), p=2, dim=-1).squeeze(0)
        
        if self.embedding_matrix is None:
            self.embedding_matrix = embedding.unsqueeze(0)
            self.player_ids = [player_id]
        else:
            self.embedding_matrix = torch.cat([
                self.embedding_matrix,
                embedding.unsqueeze(0),
            ], dim=0)
            self.player_ids.append(player_id)
        
        # Rebuild FAISS index
        if self.use_faiss:
            self._build_faiss_index()
    
    def remove(self, player_id: int):
        """Remove a player from the index."""
        if player_id not in self.player_ids:
            return
        
        idx = self.player_ids.index(player_id)
        
        self.player_ids.pop(idx)
        self.embedding_matrix = torch.cat([
            self.embedding_matrix[:idx],
            self.embedding_matrix[idx + 1:],
        ], dim=0)
        
        if player_id in self.metadata:
            del self.metadata[player_id]
        
        # Rebuild FAISS index
        if self.use_faiss:
            self._build_faiss_index()
    
    def get_embedding(self, player_id: int) -> Optional[torch.Tensor]:
        """Get embedding for a player."""
        if player_id not in self.player_ids:
            return None
        
        idx = self.player_ids.index(player_id)
        return self.embedding_matrix[idx]
    
    def __len__(self) -> int:
        return len(self.player_ids)
    
    def __contains__(self, player_id: int) -> bool:
        return player_id in self.player_ids
