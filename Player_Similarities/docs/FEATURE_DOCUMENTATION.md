# ═══════════════════════════════════════════════════════════════════════════════
#                    FEATURE DOCUMENTATION
#              Player Similarity Model v2.0
# ═══════════════════════════════════════════════════════════════════════════════

## Overview

This document describes the **learned features** in our GNN + Transformer model. Unlike v1.0 which used hand-crafted features, v2.0 learns representations directly from raw event data.

---

## Table of Contents

1. [Input Features](#1-input-features)
2. [Node Features (Graph)](#2-node-features-graph)
3. [Edge Features (Graph)](#3-edge-features-graph)
4. [Temporal Features](#4-temporal-features)
5. [Learned Embeddings](#5-learned-embeddings)

---

## 1. Input Features

### 1.1 Raw Event Data

Each event from StatsBomb contains:

| Field | Type | Description |
|-------|------|-------------|
| `type` | categorical | Event type (Pass, Shot, Dribble, etc.) |
| `location` | [x, y] | Pitch coordinates (0-120, 0-80) |
| `player` | string | Player name |
| `team` | string | Team name |
| `timestamp` | float | Time in match (seconds) |
| `period` | int | 1 = first half, 2 = second half |

### 1.2 360 Freeze Frame Data

For events with 360 data:

| Field | Type | Description |
|-------|------|-------------|
| `freeze_frame` | list | List of visible player positions |
| `visible_area` | polygon | Visible pitch area |

Each player in freeze_frame:
```python
{
    'location': [x, y],
    'teammate': True/False,
    'actor': True/False,  # Is this the on-ball player?
    'keeper': True/False
}
```

---

## 2. Node Features (Graph)

Each node in the event graph represents a player. Node features encode:

### 2.1 Feature Vector (16 dimensions)

| Index | Feature | Description |
|-------|---------|-------------|
| 0-1 | Location (x, y) | Normalized pitch position |
| 2 | Is Actor | 1 if on-ball player, 0 otherwise |
| 3 | Is Teammate | 1 if same team as actor |
| 4 | Is Keeper | 1 if goalkeeper |
| 5-6 | Relative Position | Position relative to ball |
| 7-8 | Distance to Goal | Distance to both goals |
| 9-15 | Action Type (one-hot) | Type of event (if actor) |

### 2.2 Action Type Encoding

| Index | Action Type |
|-------|-------------|
| 9 | Pass |
| 10 | Shot |
| 11 | Dribble |
| 12 | Carry |
| 13 | Tackle/Duel |
| 14 | Ball Receipt |
| 15 | Other |

---

## 3. Edge Features (Graph)

Edges connect players who are spatially close (within threshold distance).

### 3.1 Edge Feature Vector (8 dimensions)

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | Distance | Euclidean distance between players |
| 1 | Same Team | 1 if both same team, 0 otherwise |
| 2-3 | Relative Position | (dx, dy) between players |
| 4 | Angle | Angle of edge |
| 5 | Passing Lane | Is there a clear passing lane? |
| 6 | Pressing | Is defender close to ball carrier? |
| 7 | Support | Is teammate in support position? |

### 3.2 Edge Construction

```python
# Connect players within distance threshold
DISTANCE_THRESHOLD = 25  # yards

for player_a in players:
    for player_b in players:
        if distance(player_a, player_b) < DISTANCE_THRESHOLD:
            add_edge(player_a, player_b)
```

---

## 4. Temporal Features

### 4.1 Time2Vec Encoding

Instead of raw timestamps, we use learnable time encoding:

```
Time2Vec(t) = [linear(t), sin(ω₁t + φ₁), sin(ω₂t + φ₂), ...]
```

This captures:
- **Linear component**: Absolute time (early vs late game)
- **Periodic components**: Patterns (first vs second half rhythm)

### 4.2 Sequence Position

Within each player's sequence:
- Position 0 = earliest event
- Position 31 = latest event (in 32-event window)

The transformer learns which events in the sequence are most important for the player's style.

---

## 5. Learned Embeddings

### 5.1 What the Model Learns

The model learns to encode:

| Aspect | How It's Captured |
|--------|-------------------|
| **Spatial patterns** | GNN aggregates neighborhood info |
| **Decision-making** | Action type + spatial context |
| **Playing style** | Temporal patterns across events |
| **Role** | Position prediction auxiliary task |

### 5.2 Embedding Dimensions

Final player embedding: **64 dimensions**

These 64 dimensions are **not interpretable** individually—they're learned representations that capture playing style holistically.

### 5.3 What Similarity Captures

When two players have similar embeddings, it means:

| Captured | Example |
|----------|---------|
| ✅ Where they operate | Both play in left half-space |
| ✅ What actions they choose | Both prefer dribbles over passes |
| ✅ Decision timing | Both make quick decisions |
| ✅ Spatial awareness | Both use space similarly |
| ✅ Response to pressure | Both retain ball under press |

| NOT Captured | Why |
|--------------|-----|
| ❌ Outcome quality | Model sees decisions, not whether shot went in |
| ❌ Physical attributes | No speed/height data |
| ❌ Off-ball movement | 360 data only at on-ball moments |

---

## 6. Position Labels

### 6.1 Current: 14 Positions

```
Goalkeeper
Right Back, Left Back, Right Wing Back, Left Wing Back
Right Center Back, Left Center Back, Center Back
Right Defensive Midfield, Left Defensive Midfield, Center Defensive Midfield
Right Midfield, Left Midfield, Center Midfield
Right Wing, Left Wing
Right Center Forward, Left Center Forward, Center Forward
```

### 6.2 Planned: 7 Position Groups

| Group | Combines |
|-------|----------|
| Goalkeeper | GK |
| Center Back | RCB, LCB, CB |
| Full Back | RB, LB, RWB, LWB |
| Defensive Mid | RDM, LDM, CDM |
| Central Mid | RM, LM, CM |
| Winger | RW, LW |
| Forward | RCF, LCF, CF |

**Why consolidate?**
- CM and CDM often play similarly
- Left/Right distinction is arbitrary
- Scouts think in functional roles, not exact positions

---

## 7. Feature Quality Metrics

### 7.1 Position Prediction Accuracy

How well embeddings predict position:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| 14-class accuracy | 54.9% | Good (random = 7.1%) |
| 7-class (planned) | ~65% (expected) | Better grouping |

### 7.2 Similarity Gap

Difference between same-player and different-player similarities:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Same-player similarity | ~0.85 | Embeddings are consistent |
| Different-player similarity | ~0.23 | Different players are separated |
| **Gap** | **0.624** | Strong discrimination |

---

**Author:** Armen Bzdikian  
**Contact:** bzdikiana11@gmail.com  
**Last Updated:** February 2026
