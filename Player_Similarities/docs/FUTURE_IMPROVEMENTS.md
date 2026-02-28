# ═══════════════════════════════════════════════════════════════════════════════
#                     FUTURE IMPROVEMENTS
#              Player Similarity Model v2.0
# ═══════════════════════════════════════════════════════════════════════════════

## Overview

This document tracks planned improvements with a focus on making the model capture **real football context** and **actual playing style**, not just surface-level statistics or position labels.

---

## Table of Contents

1. [Critical: Position Prediction Rethink](#1-critical-position-prediction-rethink)
2. [Week 1 Tasks](#2-week-1-tasks)
3. [Week 2 Tasks](#3-week-2-tasks)
4. [Weeks 3-4 Tasks](#4-weeks-3-4-tasks)
5. [Football Context Improvements](#5-football-context-improvements)
6. [Technical Debt](#6-technical-debt)

---

## 1. Critical: Position Prediction Rethink

### The Problem

Our current auxiliary task predicts **position** (14 classes), but this has fundamental issues:

```
Current Position Classes (14):
├── Goalkeeper
├── Right Back, Left Back, Right Wing Back, Left Wing Back
├── Right Center Back, Left Center Back, Center Back
├── Right Defensive Midfield, Left Defensive Midfield, Center Defensive Midfield
├── Right Midfield, Left Midfield, Center Midfield
├── Right Wing, Left Wing
├── Right Center Forward, Left Center Forward, Center Forward
└── ... (too granular!)
```

**Why this is wrong:**

1. **Positions don't equal playing style**
   - A CM can play like a CDM (Casemiro at United)
   - A CDM can play like a CB (Fabinho dropping back)
   - Two CMs can be completely different (Kroos vs. Kanté)

2. **54.9% accuracy sounds good but is misleading**
   - The model might be learning "this player stays deep" = CDM
   - That's spatial, not stylistic
   - We want to capture HOW they play, not WHERE they stand

3. **14 positions is too granular**
   - Left Center Back vs Right Center Back? Same thing.
   - Right Defensive Midfield vs Center Defensive Midfield? Functionally similar.

### The Solution: Consolidated Position Groups

**Reduce 14 positions → 7 functional groups:**

| Group | Positions Merged | Football Logic |
|-------|-----------------|----------------|
| **Goalkeeper** | GK | Unique role, keep separate |
| **Center Back** | RCB, LCB, CB | All do the same job |
| **Full Back** | RB, LB, RWB, LWB | Overlap significantly in modern football |
| **Defensive Mid** | RDM, LDM, CDM | Hold players, all similar function |
| **Central Mid** | RM, LM, CM | Box-to-box players |
| **Winger** | RW, LW | Wide attackers |
| **Forward** | RCF, LCF, CF, ST | Goal scorers |

**Code change needed:**

```python
# In training/losses.py or data preprocessing

POSITION_MAPPING = {
    # Goalkeeper
    'Goalkeeper': 0,
    
    # Center Backs → 1
    'Right Center Back': 1,
    'Left Center Back': 1,
    'Center Back': 1,
    
    # Full Backs → 2
    'Right Back': 2,
    'Left Back': 2,
    'Right Wing Back': 2,
    'Left Wing Back': 2,
    
    # Defensive Mids → 3
    'Right Defensive Midfield': 3,
    'Left Defensive Midfield': 3,
    'Center Defensive Midfield': 3,
    
    # Central Mids → 4
    'Right Midfield': 4,
    'Left Midfield': 4,
    'Center Midfield': 4,
    
    # Wingers → 5
    'Right Wing': 5,
    'Left Wing': 5,
    
    # Forwards → 6
    'Right Center Forward': 6,
    'Left Center Forward': 6,
    'Center Forward': 6,
}
```

### Better Auxiliary Task: Action Distribution Prediction

Instead of (or in addition to) position, predict the player's **action distribution**:

```python
# What percentage of this player's actions are:
action_distribution = {
    'passes': 0.45,
    'dribbles': 0.15,
    'shots': 0.05,
    'tackles': 0.10,
    'clearances': 0.08,
    'aerial_duels': 0.07,
    'recoveries': 0.10
}

# This captures STYLE not just POSITION
# Messi: high dribble%, low tackle%
# Kanté: high tackle%, high recovery%
# Busquets: high pass%, low shot%
```

---

## 2. Week 1 Tasks

| Task | Description | Priority |
|------|-------------|----------|
| **Consolidate positions** | Reduce 14 → 7 position groups | 🔴 HIGH |
| **Input validation** | Sanitize API inputs, handle edge cases | 🟡 MEDIUM |
| **Position filtering** | Add position filter to similarity search | 🔴 HIGH |
| **Generate figures** | Run analytics notebook for docs | 🟡 MEDIUM |

### Position Filtering for Similarity Search

```python
def find_similar(self, player_name: str, k: int = 10, position_group: str = None):
    """
    Find similar players, optionally filtered by position.
    
    position_group: One of ['goalkeeper', 'center_back', 'full_back', 
                           'defensive_mid', 'central_mid', 'winger', 'forward']
    """
    if position_group:
        candidates = self.filter_by_position_group(candidates, position_group)
    # ... rest of search
```

---

## 3. Week 2 Tasks

| Task | Description | Priority |
|------|-------------|----------|
| **Action distribution aux task** | Predict action type distribution | 🔴 HIGH |
| **Hyperparameter tuning** | Temperature, batch size experiments | 🟡 MEDIUM |
| **Streamlit demo** | Interactive web interface | 🟡 MEDIUM |
| **Training stability** | Address loss spikes with gradient clipping | 🟡 MEDIUM |

### Hyperparameter Experiments

| Parameter | Current | Try | Why |
|-----------|---------|-----|-----|
| Temperature | 0.07 | 0.05, 0.10 | Sharper vs softer contrastive |
| Batch Size | 32 | 64 | More negatives |
| Position Groups | 14 | 7 | Less granular |
| Aux Task | Position | Action Distribution | Better captures style |

---

## 4. Weeks 3-4 Tasks

| Task | Description | Why |
|------|-------------|-----|
| **Expert validation** | Show to scouts, get feedback | Validate real-world usefulness |
| **Same-position similarity** | Only compare within position groups | More meaningful for scouting |
| **Style clustering** | K-means on embeddings within position | Discover playing styles |
| **Full documentation** | Complete all docs, examples | Ready for handoff |

---

## 5. Football Context Improvements

### 5.1 Understand the Gaps

Current model captures spatial decisions but may miss:

| Gap | Example | How to Fix |
|-----|---------|------------|
| **Outcome quality** | Two players who shoot a lot, but one scores | Add xG-weighted features |
| **Off-ball movement** | Runs without the ball | Need tracking data (not in StatsBomb) |
| **Decision timing** | Quick vs slow decisions | Add time-to-action feature |
| **Pressure response** | How they play under pressure | Already have this from 360 data ✓ |

### 5.2 Football-Meaningful Position Groups

Think about positions from a **scouting perspective**:

```
When a scout says "find me a player like Rodri", they mean:
❌ "Find me another CDM" (too vague)
✓ "Find me a deep-lying playmaker who controls tempo"

When they say "find me a Messi replacement", they mean:
❌ "Find me another RW" (misses the point)
✓ "Find me a ball-carrier who creates in tight spaces"
```

Our consolidated positions help with this:
- CDM/DM → all deep midfielders who could play Rodri's role
- RW/LW → all wide players who could play Messi's role

### 5.3 Style Archetypes (Future)

Once embeddings are good, cluster within position groups to discover styles:

```
Full Backs:
├── Cluster 1: Overlapping attackers (Alexander-Arnold, Cancelo)
├── Cluster 2: Defensive stay-at-home (Wan-Bissaka)
└── Cluster 3: Inverted (Zinchenko, Walker at CB)

Central Mids:
├── Cluster 1: Deep playmakers (Kroos, Busquets)
├── Cluster 2: Box-to-box (Valverde, Bellingham)
└── Cluster 3: Destroyers (Kanté, Casemiro)
```

---

## 6. Technical Debt

```python
# Issues to address:
# ─────────────────────────────────────────────────────────

# 1. Position labels are too granular
num_positions = 14  # Should be 7

# 2. Loss spikes during training
#    - Need gradient clipping
#    - Learning rate warmup

# 3. No action distribution tracking
#    - Should track what % of actions are passes/dribbles/shots

# 4. Hardcoded paths
PATH = 'outputs/embeddings.pt'  # Should use config

# 5. No position filtering in similarity API
```

---

## Summary: What We're Fixing

| Current Problem | Solution | Impact |
|----------------|----------|--------|
| 14 positions too granular | Consolidate to 7 groups | Better captures functional roles |
| Position ≠ style | Add action distribution task | Captures HOW players play |
| Cross-position comparisons | Position filtering in API | More useful for scouts |
| Model predicts WHERE not HOW | Style-based auxiliary tasks | True similarity |

---

**Author:** Armen Bzdikian  
**Contact:** bzdikiana11@gmail.com  
**Last Updated:** February 2026
