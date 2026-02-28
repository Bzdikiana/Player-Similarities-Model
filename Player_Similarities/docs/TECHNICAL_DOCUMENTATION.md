# ═══════════════════════════════════════════════════════════════════════════════
#                    TECHNICAL DOCUMENTATION
#              Player Similarity Model v2.0 (GNN + Transformer)
# ═══════════════════════════════════════════════════════════════════════════════

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Data Pipeline](#2-data-pipeline)
3. [Model Architecture](#3-model-architecture)
4. [Training Pipeline](#4-training-pipeline)
5. [Inference & Retrieval](#5-inference--retrieval)
6. [Configuration](#6-configuration)

---

## 1. System Overview

### 1.1 Purpose

This system learns **player embeddings** using deep learning on StatsBomb 360 event data. The core idea:

> **Two players are similar if, given similar game situations, they make similar decisions.**

This version uses:
- **Graph Neural Networks** to encode spatial relationships in each event
- **Temporal Transformers** to capture action sequences over time
- **Contrastive Learning** to pull same-player embeddings together

### 1.2 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PLAYER SIMILARITY v2.0                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   StatsBomb 360 Events                                                       │
│          │                                                                   │
│          ▼                                                                   │
│   ┌──────────────────┐                                                       │
│   │ Stats360Adapter  │ ─── Loads events + 360 freeze frames                 │
│   └──────────────────┘                                                       │
│          │                                                                   │
│          ▼                                                                   │
│   ┌──────────────────┐                                                       │
│   │ EventGraphBuilder│ ─── Converts each event to a graph                   │
│   └──────────────────┘     (players as nodes, spatial edges)                │
│          │                                                                   │
│          ▼                                                                   │
│   ┌──────────────────┐                                                       │
│   │ TemporalSequence │ ─── Groups events into sequences per player          │
│   │ Builder          │                                                       │
│   └──────────────────┘                                                       │
│          │                                                                   │
│          ▼                                                                   │
│   ┌──────────────────────────────────────────────────────────────────┐      │
│   │                    ENCODER ARCHITECTURE                          │      │
│   │  ┌─────────────────┐   ┌──────────────────┐   ┌──────────────┐  │      │
│   │  │ Graph Attention │──▶│ Temporal         │──▶│ Projection   │  │      │
│   │  │ Network (3 lyrs)│   │ Transformer (2)  │   │ Head (MLP)   │  │      │
│   │  └─────────────────┘   └──────────────────┘   └──────────────┘  │      │
│   └──────────────────────────────────────────────────────────────────┘      │
│          │                                                                   │
│          ▼                                                                   │
│   ┌──────────────────┐                                                       │
│   │ 64-dim Embedding │ ─── One vector per player                            │
│   └──────────────────┘                                                       │
│          │                                                                   │
│          ▼                                                                   │
│   ┌──────────────────┐                                                       │
│   │ FAISS Index      │ ─── Fast similarity search                           │
│   └──────────────────┘                                                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Pipeline

### 2.1 Data Source

**StatsBomb 360 Data** provides:

| Data Type | Description | Why We Need It |
|-----------|-------------|----------------|
| Events | On-ball actions (pass, shot, dribble, etc.) | Core input features |
| 360 Freeze Frames | Positions of all 22 players at moment of action | Spatial context for graph |
| Lineups | Player positions and team info | Position labels |

### 2.2 Stats360Adapter

Located in `src/datasets/adapters/stats360_adapter.py`

```python
class Stats360Adapter:
    """Loads StatsBomb 360 data and merges events with freeze frames."""
    
    def load_competitions(self) -> pd.DataFrame
    def load_matches(self, competition_id, season_id) -> pd.DataFrame
    def load_events_with_360(self, match_id) -> pd.DataFrame
```

**Key Processing**:
- Merges event data with 360 freeze frames on `id` field
- Normalizes coordinates to attacking direction (always left-to-right)
- Extracts visible teammates/opponents from freeze frame

### 2.3 EventGraphBuilder

Located in `src/datasets/builders/event_graph_builder.py`

Converts each event into a **graph**:
- **Nodes**: Actor (on-ball player) + visible teammates + opponents
- **Edges**: Spatial relationships between players
- **Edge Features**: Distance, same_team flag, relative position

```python
class EventGraphBuilder:
    def build_graph(self, event: dict) -> Data:
        """
        Returns PyG Data object with:
        - x: Node features [num_players, node_dim]
        - edge_index: COO format edges [2, num_edges]
        - edge_attr: Edge features [num_edges, edge_dim]
        """
```

### 2.4 TemporalSequenceBuilder

Located in `src/datasets/builders/temporal_sequence_builder.py`

Groups events into sequences per player:

```python
class TemporalSequenceBuilder:
    def build_sequences(self, events_df, sequence_length=32):
        """
        For each player, creates sequences of their actions.
        Returns list of (player_id, [event_graphs])
        """
```

---

## 3. Model Architecture

### 3.1 Graph Attention Network (GNN)

Located in `src/models/encoders/event_graph_encoder.py`

**Purpose**: Encode spatial relationships within a single event

```
Input:  Event graph (nodes = players, edges = spatial)
Output: Single vector summarizing the event context
```

**Architecture**:
- 3 Graph Attention layers with 4 attention heads
- Edge features incorporated via edge attention
- Global mean pooling to get event embedding

### 3.2 Temporal Transformer

Located in `src/models/encoders/temporal_transformer.py`

**Purpose**: Capture patterns across a sequence of events

```
Input:  Sequence of event embeddings [seq_len, hidden_dim]
Output: Single player embedding [embedding_dim]
```

**Architecture**:
- Time2Vec positional encoding (learnable time representation)
- 2 Transformer encoder layers
- Hierarchical pooling: segments → player

### 3.3 Projection Head

Simple MLP that projects to final embedding space (64 dimensions).

---

## 4. Training Pipeline

### 4.1 Contrastive Learning (InfoNCE)

Located in `src/training/losses.py`

**Goal**: Pull embeddings of the same player together, push different players apart.

$$L_{InfoNCE} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_k \exp(\text{sim}(z_i, z_k)/\tau)}$$

### 4.2 Auxiliary Task: Position Prediction

Current implementation predicts player position (14 classes).

**Known Limitation**: Position alone doesn't capture playing style—a CM and CDM may play very differently. See FUTURE_IMPROVEMENTS.md for planned changes.

### 4.3 Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 50 |
| Batch Size | 32 |
| Learning Rate | 1e-4 |
| Temperature | 0.07 |
| Position Loss Weight | 0.01 |

---

## 5. Inference & Retrieval

### 5.1 FAISS Index

Located in `src/retrieval/index.py`

Uses Facebook's FAISS library for fast approximate nearest neighbor search on normalized embeddings.

### 5.2 Similarity API

Located in `src/retrieval/api.py`

```python
class SimilarityAPI:
    def find_similar(self, player_name: str, k: int = 10):
        """Find k most similar players to given player"""
        
    def compare(self, player1: str, player2: str):
        """Get similarity score between two players"""
```

---

## 6. Configuration

Config files located in `src/configs/`:
- `model.yaml` - Model architecture parameters
- `train.yaml` - Training hyperparameters
- `data.yaml` - Data loading settings
- `eval.yaml` - Evaluation settings

---

## File Structure

```
src/
├── configs/           # YAML configuration files
├── datasets/
│   ├── adapters/      # Data loading (stats360_adapter.py)
│   └── builders/      # Graph/sequence construction
├── models/
│   ├── layers/        # GNN layers, attention mechanisms
│   └── encoders/      # Full encoder modules
├── training/          # Losses, training loop, metrics
└── retrieval/         # FAISS index, similarity API
```

---

**Author:** Armen Bzdikian  
**Contact:** bzdikiana11@gmail.com  
**Last Updated:** February 2026
