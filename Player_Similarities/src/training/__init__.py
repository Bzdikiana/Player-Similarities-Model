"""
Training modules: losses, trainer, metrics, callbacks.
"""

from .losses import (
    InfoNCELoss, 
    ReliabilityWeightedLoss, 
    AuxiliaryLosses,
    MultiTaskLoss,
    PositionPredictionHead,
    PositionPredictionLoss,
    EventTypePredictionHead,
    EventDistributionLoss,
    HardNegativeMiningLoss,
    compute_reliability_weights,
)
from .train import Trainer, TrainingConfig
from .metrics import compute_recall_at_k, compute_ndcg, EmbeddingMetrics
from .callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

__all__ = [
    "InfoNCELoss",
    "ReliabilityWeightedLoss",
    "AuxiliaryLosses",
    "MultiTaskLoss",
    "PositionPredictionHead",
    "PositionPredictionLoss",
    "EventTypePredictionHead",
    "EventDistributionLoss",
    "HardNegativeMiningLoss",
    "compute_reliability_weights",
    "Trainer",
    "TrainingConfig",
    "compute_recall_at_k",
    "compute_ndcg",
    "EmbeddingMetrics",
    "EarlyStopping",
    "ModelCheckpoint",
    "LearningRateScheduler",
]
