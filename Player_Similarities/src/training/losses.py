"""
Training Losses

Implements loss functions for the player embedding pipeline:
1. InfoNCE contrastive loss (primary)
2. Reliability-weighted losses
3. Auxiliary losses (role classification, consistency, etc.)
4. Multi-task learning heads (position prediction, event prediction)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List


class InfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss for player embeddings.
    
    Pulls together embeddings of the same player (positives) and
    pushes apart embeddings of different players (negatives).
    
    For a batch of player embeddings:
    - Positives: Same player, different match/time (via data augmentation or sampling)
    - Negatives: Different players in the batch
    
    Loss = -log(exp(sim(z_i, z_j) / τ) / Σ_k exp(sim(z_i, z_k) / τ))
    
    Args:
        temperature: Softmax temperature (lower = sharper distribution)
        reduction: Loss reduction ('mean', 'sum', 'none')
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        reduction: str = "mean",
    ):
        super().__init__()
        
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            embeddings: [B, D] L2-normalized embeddings
            labels: [B] player IDs (same ID = positive pair)
            mask: [B] optional mask for valid samples
            
        Returns:
            Scalar loss
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]
        
        # Compute similarity matrix
        # [B, B] = [B, D] @ [D, B]
        sim_matrix = torch.matmul(embeddings, embeddings.t()) / self.temperature
        
        # Create positive mask (same player = 1)
        labels = labels.view(-1, 1)
        positive_mask = torch.eq(labels, labels.t()).float()  # [B, B]
        
        # Remove self-similarity from positives
        self_mask = torch.eye(batch_size, device=device)
        positive_mask = positive_mask - self_mask
        
        # For numerical stability, subtract max
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()
        
        # Compute log_sum_exp for denominator (exclude self)
        exp_logits = torch.exp(logits) * (1 - self_mask)
        log_sum_exp = torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
        
        # Compute loss for each positive pair
        # Sum over all positives for each anchor
        n_positives = positive_mask.sum(dim=1)
        
        # Handle case with no positives
        has_positive = n_positives > 0
        
        # Log probability of positives
        log_prob = logits - log_sum_exp  # [B, B]
        
        # Mean log prob over positives for each anchor
        # Masked sum / count
        pos_log_prob = (positive_mask * log_prob).sum(dim=1)
        pos_log_prob = pos_log_prob / n_positives.clamp(min=1)
        
        # Loss is negative log prob
        loss = -pos_log_prob
        
        # Zero out loss for anchors without positives
        loss = loss * has_positive.float()
        
        # Apply sample mask
        if mask is not None:
            loss = loss * mask.float()
            valid_count = mask.sum().clamp(min=1)
        else:
            valid_count = has_positive.sum().clamp(min=1)
        
        # Reduce
        if self.reduction == "mean":
            return loss.sum() / valid_count
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.
    
    Extension of InfoNCE that handles multiple positives per anchor
    more elegantly by treating each positive pair independently.
    
    Reference: Khosla et al., "Supervised Contrastive Learning"
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
    ):
        super().__init__()
        
        self.temperature = temperature
        self.base_temperature = base_temperature
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute SupCon loss.
        
        Args:
            features: [B, D] L2-normalized embeddings
            labels: [B] player IDs
            mask: [B] optional validity mask
            
        Returns:
            Scalar loss
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Mask for same class (excluding self)
        labels = labels.contiguous().view(-1, 1)
        class_mask = torch.eq(labels, labels.T).float().to(device)
        
        # Self mask
        self_mask = torch.eye(batch_size, device=device)
        
        # Positive mask: same class, not self
        positive_mask = class_mask - self_mask
        
        # Negative mask: different class
        negative_mask = 1 - class_mask
        
        # Compute logits
        anchor_dot_contrast = torch.matmul(features, features.T) / self.temperature
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Compute log_softmax
        exp_logits = torch.exp(logits) * (1 - self_mask)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
        
        # Compute mean of log-likelihood over positives
        n_positives = positive_mask.sum(dim=1)
        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / n_positives.clamp(min=1)
        
        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        
        # Only include samples with at least one positive
        has_positive = n_positives > 0
        loss = loss * has_positive.float()
        
        if mask is not None:
            loss = loss * mask.float()
            return loss.sum() / mask.sum().clamp(min=1)
        
        return loss.sum() / has_positive.sum().clamp(min=1)


class ReliabilityWeightedLoss(nn.Module):
    """
    Wrapper that weights any loss by reliability scores.
    
    Players with higher reliability (more data) contribute more to the loss.
    This prevents noisy gradients from low-data players.
    
    weighted_loss = (reliability * base_loss).mean() / reliability.mean()
    """
    
    def __init__(
        self,
        base_loss: nn.Module,
        min_weight: float = 0.1,
    ):
        super().__init__()
        
        self.base_loss = base_loss
        self.min_weight = min_weight
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        reliability: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute reliability-weighted loss.
        
        Args:
            embeddings: [B, D] embeddings
            labels: [B] labels
            reliability: [B] or [B, 1] reliability scores in [0, 1]
            
        Returns:
            Weighted scalar loss
        """
        # Get per-sample loss
        base_loss_fn = self.base_loss
        base_loss_fn.reduction = "none"
        
        per_sample_loss = base_loss_fn(embeddings, labels, **kwargs)  # [B]
        
        # Flatten reliability
        if reliability.dim() > 1:
            reliability = reliability.squeeze(-1)
        
        # Clamp minimum weight
        weights = reliability.clamp(min=self.min_weight)
        
        # Weighted average
        weighted_loss = (weights * per_sample_loss).sum() / weights.sum().clamp(min=1e-8)
        
        return weighted_loss


class AuxiliaryLosses(nn.Module):
    """
    Collection of auxiliary losses for multi-task training:
    
    1. Role classification loss
    2. Event type prediction loss (reconstructive)
    3. Global consistency loss
    4. Triplet loss (optional alternative to InfoNCE)
    """
    
    def __init__(
        self,
        n_roles: int = 10,
        n_event_types: int = 30,
        consistency_weight: float = 0.1,
        role_weight: float = 0.1,
        event_pred_weight: float = 0.05,
    ):
        super().__init__()
        
        self.consistency_weight = consistency_weight
        self.role_weight = role_weight
        self.event_pred_weight = event_pred_weight
        
        # Role classification
        self.role_loss = nn.CrossEntropyLoss(reduction="mean")
        
        # Event type prediction
        self.event_loss = nn.CrossEntropyLoss(reduction="mean")
    
    def role_classification_loss(
        self,
        role_logits: torch.Tensor,
        role_labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute role classification loss.
        
        Args:
            role_logits: [B, n_roles] predicted logits
            role_labels: [B] ground truth role indices
            mask: [B] optional validity mask
        """
        if mask is not None:
            role_logits = role_logits[mask]
            role_labels = role_labels[mask]
        
        if role_labels.numel() == 0:
            return torch.tensor(0.0, device=role_logits.device)
        
        return self.role_loss(role_logits, role_labels)
    
    def event_prediction_loss(
        self,
        event_logits: torch.Tensor,
        event_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute event type prediction loss.
        
        Args:
            event_logits: [B, n_events] predicted logits
            event_labels: [B] ground truth event type indices
        """
        return self.event_loss(event_logits, event_labels)
    
    def consistency_loss(
        self,
        data_embeddings: torch.Tensor,
        global_embeddings: torch.Tensor,
        reliability: Optional[torch.Tensor] = None,
        margin: float = 0.1,
    ) -> torch.Tensor:
        """
        Compute consistency loss between data and global embeddings.
        
        Encourages data embeddings to stay close to global priors,
        especially for low-data players.
        
        Args:
            data_embeddings: [B, D] from event GNN + Transformer
            global_embeddings: [B, D] from global player graph
            reliability: [B] reliability scores
            margin: Only penalize if distance > margin
        """
        # Cosine distance
        cos_sim = F.cosine_similarity(data_embeddings, global_embeddings, dim=-1)
        distance = 1 - cos_sim  # [B]
        
        # Margin loss
        loss = F.relu(distance - margin)
        
        # Weight by inverse reliability (more regularization for low-data)
        if reliability is not None:
            if reliability.dim() > 1:
                reliability = reliability.squeeze(-1)
            weights = 1 - reliability  # High reliability = low weight
            loss = loss * weights
        
        return loss.mean()
    
    def forward(
        self,
        role_logits: Optional[torch.Tensor] = None,
        role_labels: Optional[torch.Tensor] = None,
        event_logits: Optional[torch.Tensor] = None,
        event_labels: Optional[torch.Tensor] = None,
        data_embeddings: Optional[torch.Tensor] = None,
        global_embeddings: Optional[torch.Tensor] = None,
        reliability: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all auxiliary losses.
        
        Returns:
            Dict with individual losses and total weighted sum
        """
        losses = {}
        total = 0.0
        
        # Role classification
        if role_logits is not None and role_labels is not None:
            role_loss = self.role_classification_loss(role_logits, role_labels)
            losses["role_loss"] = role_loss
            total = total + self.role_weight * role_loss
        
        # Event prediction
        if event_logits is not None and event_labels is not None:
            event_loss = self.event_prediction_loss(event_logits, event_labels)
            losses["event_loss"] = event_loss
            total = total + self.event_pred_weight * event_loss
        
        # Consistency
        if data_embeddings is not None and global_embeddings is not None:
            consistency = self.consistency_loss(
                data_embeddings, global_embeddings, reliability
            )
            losses["consistency_loss"] = consistency
            total = total + self.consistency_weight * consistency
        
        losses["aux_total"] = total
        return losses


class TripletLoss(nn.Module):
    """
    Alternative to InfoNCE using triplet margin loss.
    
    For each anchor, pulls positive closer and pushes negative farther.
    """
    
    def __init__(
        self,
        margin: float = 0.3,
        distance: str = "cosine",
    ):
        super().__init__()
        
        self.margin = margin
        self.distance = distance
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Args:
            anchor: [B, D] anchor embeddings
            positive: [B, D] positive embeddings (same player)
            negative: [B, D] negative embeddings (different player)
        """
        if self.distance == "cosine":
            pos_dist = 1 - F.cosine_similarity(anchor, positive)
            neg_dist = 1 - F.cosine_similarity(anchor, negative)
        else:  # euclidean
            pos_dist = F.pairwise_distance(anchor, positive)
            neg_dist = F.pairwise_distance(anchor, negative)
        
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


def compute_reliability_weights(
    n_events: torch.Tensor,
    k: float = 50.0,
    min_weight: float = 0.1,
) -> torch.Tensor:
    """
    Compute reliability weights using formula: r = n / (n + k)
    
    Args:
        n_events: [B] number of events per player
        k: Smoothing constant
        min_weight: Minimum weight
        
    Returns:
        [B] reliability weights
    """
    weights = n_events.float() / (n_events.float() + k)
    return weights.clamp(min=min_weight)


# =============================================================================
# Multi-Task Learning Components
# =============================================================================

class PositionPredictionHead(nn.Module):
    """
    Auxiliary task: Predict player's position from their embedding.
    
    This encourages the embedding to encode position-relevant information
    while still being flexible enough for similarity search.
    
    Args:
        input_dim: Input embedding dimension
        hidden_dim: Hidden layer dimension
        n_positions: Number of position classes (25 in StatsBomb)
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 64,
        n_positions: int = 26,  # 25 positions + 1 unknown
    ):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_positions),
        )
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Predict position logits.
        
        Args:
            embeddings: [B, D] player embeddings
            
        Returns:
            [B, n_positions] position logits
        """
        return self.head(embeddings)


class EventTypePredictionHead(nn.Module):
    """
    Auxiliary task: Predict distribution of event types for a player.
    
    Instead of single-label classification, predicts a distribution over
    event types based on the player's typical actions.
    
    Args:
        input_dim: Input embedding dimension
        hidden_dim: Hidden layer dimension
        n_event_types: Number of event type classes
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 64,
        n_event_types: int = 35,  # 34 types + 1 unknown
    ):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_event_types),
        )
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Predict event type distribution logits.
        
        Args:
            embeddings: [B, D] player embeddings
            
        Returns:
            [B, n_event_types] event type logits
        """
        return self.head(embeddings)


class NextActionPredictionHead(nn.Module):
    """
    Auxiliary task: Predict what action a player will do NEXT given current context.
    
    This is much more useful than position prediction because it captures:
    - Decision-making patterns (Messi tends to dribble, defenders tend to clear)
    - Situational awareness (what does this player do in this situation?)
    - Playing style (aggressive vs conservative, direct vs patient)
    
    The input is the player embedding which encodes their past actions.
    The output is a distribution over possible next actions.
    
    Football context:
    - Messi: High probability for Dribble, Shot, Through Pass
    - Van Dijk: High probability for Clearance, Long Pass, Aerial Duel
    - Busquets: High probability for Short Pass, Ball Recovery, Interception
    
    Args:
        input_dim: Input embedding dimension  
        hidden_dim: Hidden layer dimension
        n_action_types: Number of action classes (Pass, Shot, Dribble, etc.)
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 64,
        n_action_types: int = 35,  # Based on EVENT_TYPE_CATEGORIES
    ):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_action_types),
        )
        
        self.n_action_types = n_action_types
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Predict next action logits.
        
        Args:
            embeddings: [B, D] player embeddings representing their style/context
            
        Returns:
            [B, n_action_types] logits for each possible next action
        """
        return self.head(embeddings)


class NextActionPredictionLoss(nn.Module):
    """
    Loss for next action prediction.
    
    Given the player's embedding (which represents their playing style/context),
    predict what action they will perform next.
    
    This teaches the embedding to capture:
    - What type of player is this? (attacker who shoots vs creator who passes)
    - What is their decision-making pattern?
    
    Args:
        label_smoothing: Smoothing for cross-entropy (actions aren't always deterministic)
        class_weights: Optional weights for imbalanced action types
    """
    
    def __init__(
        self,
        label_smoothing: float = 0.1,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        
        self.loss_fn = nn.CrossEntropyLoss(
            reduction="mean",
            label_smoothing=label_smoothing,
            weight=class_weights,
        )
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute next action prediction loss.
        
        Args:
            logits: [B, n_actions] predicted action logits
            labels: [B] ground truth next action indices
            mask: [B] optional validity mask
            
        Returns:
            Tuple of (loss, accuracy)
        """
        if mask is not None:
            logits = logits[mask]
            labels = labels[mask]
        
        if labels.numel() == 0:
            return torch.tensor(0.0, device=logits.device), torch.tensor(0.0)
        
        loss = self.loss_fn(logits, labels)
        
        # Compute accuracy
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            accuracy = (preds == labels).float().mean()
        
        return loss, accuracy


class PositionPredictionLoss(nn.Module):
    """
    Loss for position prediction auxiliary task.
    
    Uses cross-entropy for single-label or KL divergence for soft labels.
    
    Args:
        label_smoothing: Amount of label smoothing (0 for hard labels)
    """
    
    def __init__(self, label_smoothing: float = 0.1):
        super().__init__()
        
        self.loss_fn = nn.CrossEntropyLoss(
            reduction="mean",
            label_smoothing=label_smoothing,
        )
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute position prediction loss.
        
        Args:
            logits: [B, n_positions] predicted logits
            labels: [B] ground truth position indices
            mask: [B] optional validity mask
            
        Returns:
            Scalar loss
        """
        if mask is not None:
            logits = logits[mask]
            labels = labels[mask]
        
        if labels.numel() == 0:
            return torch.tensor(0.0, device=logits.device)
        
        return self.loss_fn(logits, labels)


class EventDistributionLoss(nn.Module):
    """
    Loss for event type distribution prediction.
    
    Uses KL divergence to match predicted distribution with actual
    event type distribution for the player.
    
    Args:
        temperature: Softmax temperature for smoothing
    """
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        
        self.temperature = temperature
    
    def forward(
        self,
        logits: torch.Tensor,
        target_distribution: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute event distribution KL divergence loss.
        
        Args:
            logits: [B, n_event_types] predicted logits
            target_distribution: [B, n_event_types] actual distribution (counts or probs)
            mask: [B] optional validity mask
            
        Returns:
            Scalar loss
        """
        # Normalize target to probability distribution
        target_sum = target_distribution.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        target_probs = target_distribution / target_sum
        
        # Apply temperature and softmax to predictions
        pred_log_probs = F.log_softmax(logits / self.temperature, dim=-1)
        
        # KL divergence: target * (log(target) - log(pred))
        # = target * log(target) - target * log(pred)
        # Since target * log(target) is constant, minimize -target * log(pred)
        loss = -torch.sum(target_probs * pred_log_probs, dim=-1)  # [B]
        
        if mask is not None:
            loss = loss[mask]
        
        if loss.numel() == 0:
            return torch.tensor(0.0, device=logits.device)
        
        return loss.mean()


class MultiTaskLoss(nn.Module):
    """
    Combined multi-task loss for player embedding training.
    
    Combines:
    1. Primary loss: InfoNCE contrastive loss
    2. Next Action prediction auxiliary loss (recommended)
    3. Position prediction auxiliary loss (optional, less useful)
    4. Event distribution prediction loss (optional)
    
    Next Action Prediction is the BEST auxiliary task because:
    - It captures decision-making: "What would this player do in this situation?"
    - It differentiates playing styles: Messi → dribble/shot, Busquets → pass/recover
    - It's directly related to how scouts evaluate players
    
    Args:
        temperature: InfoNCE temperature
        position_weight: Weight for position prediction loss (recommend 0.01 or off)
        action_weight: Weight for next action prediction loss (recommend 0.1)
        event_weight: Weight for event distribution loss
        use_position_loss: Whether to use position prediction
        use_action_loss: Whether to use next action prediction (RECOMMENDED)
        use_event_loss: Whether to use event distribution prediction
        n_positions: Number of position classes
        n_action_types: Number of action type classes
        n_event_types: Number of event type classes
        embedding_dim: Embedding dimension for auxiliary heads
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        position_weight: float = 0.01,  # Reduced from 0.1
        action_weight: float = 0.1,     # NEW: next action prediction
        event_weight: float = 0.05,
        use_position_loss: bool = False,  # Disabled by default
        use_action_loss: bool = True,     # NEW: enabled by default
        use_event_loss: bool = False,
        n_positions: int = 26,
        n_action_types: int = 35,
        n_event_types: int = 35,
        embedding_dim: int = 64,
    ):
        super().__init__()
        
        self.position_weight = position_weight
        self.action_weight = action_weight
        self.event_weight = event_weight
        self.use_position_loss = use_position_loss
        self.use_action_loss = use_action_loss
        self.use_event_loss = use_event_loss
        
        # Primary contrastive loss
        self.contrastive_loss = InfoNCELoss(temperature=temperature)
        
        # Auxiliary heads and losses
        if use_position_loss:
            self.position_head = PositionPredictionHead(
                input_dim=embedding_dim,
                n_positions=n_positions,
            )
            self.position_loss = PositionPredictionLoss()
        
        # NEW: Next Action Prediction
        if use_action_loss:
            self.action_head = NextActionPredictionHead(
                input_dim=embedding_dim,
                n_action_types=n_action_types,
            )
            self.action_loss = NextActionPredictionLoss(label_smoothing=0.1)
        
        if use_event_loss:
            self.event_head = EventTypePredictionHead(
                input_dim=embedding_dim,
                n_event_types=n_event_types,
            )
            self.event_loss = EventDistributionLoss()
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        position_labels: Optional[torch.Tensor] = None,
        next_action_labels: Optional[torch.Tensor] = None,  # NEW
        event_distributions: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.
        
        Args:
            embeddings: [B, D] L2-normalized player embeddings
            labels: [B] player IDs for contrastive loss
            position_labels: [B] position indices for position prediction (optional)
            next_action_labels: [B] next action indices (Pass, Shot, Dribble, etc.)
            event_distributions: [B, n_event_types] event type distributions
            mask: [B] validity mask
            
        Returns:
            Dict with individual losses and total
        """
        losses = {}
        
        # Primary contrastive loss
        contrastive = self.contrastive_loss(embeddings, labels, mask)
        losses["contrastive_loss"] = contrastive
        total = contrastive
        
        # NEW: Next Action Prediction auxiliary loss (RECOMMENDED)
        if self.use_action_loss and next_action_labels is not None:
            action_logits = self.action_head(embeddings)
            action_loss, action_acc = self.action_loss(action_logits, next_action_labels, mask)
            losses["action_loss"] = action_loss
            losses["action_accuracy"] = action_acc
            total = total + self.action_weight * action_loss
        
        # Position prediction auxiliary loss (less recommended)
        if self.use_position_loss and position_labels is not None:
            pos_logits = self.position_head(embeddings)
            pos_loss = self.position_loss(pos_logits, position_labels, mask)
            losses["position_loss"] = pos_loss
            total = total + self.position_weight * pos_loss
            
            # Add position accuracy for monitoring
            with torch.no_grad():
                pos_preds = pos_logits.argmax(dim=-1)
                if mask is not None:
                    pos_acc = (pos_preds[mask] == position_labels[mask]).float().mean()
                else:
                    pos_acc = (pos_preds == position_labels).float().mean()
                losses["position_accuracy"] = pos_acc
        
        # Event distribution auxiliary loss
        if self.use_event_loss and event_distributions is not None:
            event_logits = self.event_head(embeddings)
            event_loss = self.event_loss(event_logits, event_distributions, mask)
            losses["event_loss"] = event_loss
            total = total + self.event_weight * event_loss
        
        losses["total_loss"] = total
        return losses
    
    def get_auxiliary_heads(self) -> Dict[str, nn.Module]:
        """Get auxiliary prediction heads for inference."""
        heads = {}
        if self.use_position_loss:
            heads["position"] = self.position_head
        if self.use_action_loss:
            heads["action"] = self.action_head
        if self.use_event_loss:
            heads["event"] = self.event_head
        return heads


class HardNegativeMiningLoss(nn.Module):
    """
    InfoNCE loss with hard negative mining.
    
    Instead of using all negatives in the batch, focuses on the hardest
    negatives (most similar different players). This can improve learning
    for fine-grained similarity distinctions.
    
    Args:
        temperature: Softmax temperature
        n_hard_negatives: Number of hard negatives to use per anchor
        use_semi_hard: Use semi-hard negatives (harder than positive but not too hard)
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        n_hard_negatives: int = 16,
        use_semi_hard: bool = True,
    ):
        super().__init__()
        
        self.temperature = temperature
        self.n_hard_negatives = n_hard_negatives
        self.use_semi_hard = use_semi_hard
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute InfoNCE with hard negative mining.
        
        Args:
            embeddings: [B, D] L2-normalized embeddings
            labels: [B] player IDs
            mask: [B] optional validity mask
            
        Returns:
            Scalar loss
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.t())  # [B, B]
        
        # Create masks
        labels_col = labels.view(-1, 1)
        positive_mask = torch.eq(labels_col, labels_col.t()).float()
        self_mask = torch.eye(batch_size, device=device)
        positive_mask = positive_mask - self_mask
        negative_mask = 1 - positive_mask - self_mask
        
        losses = []
        
        for i in range(batch_size):
            # Skip if no positives
            pos_indices = torch.where(positive_mask[i] > 0)[0]
            if len(pos_indices) == 0:
                continue
            
            # Get negative indices and similarities
            neg_indices = torch.where(negative_mask[i] > 0)[0]
            if len(neg_indices) == 0:
                continue
            
            neg_sims = sim_matrix[i, neg_indices]
            
            # Select hard negatives
            if self.use_semi_hard:
                # Semi-hard: harder than easiest positive but not too hard
                pos_sim = sim_matrix[i, pos_indices].min()
                semi_hard_mask = neg_sims < pos_sim
                if semi_hard_mask.any():
                    hard_neg_sims, _ = neg_sims[semi_hard_mask].topk(
                        min(self.n_hard_negatives, semi_hard_mask.sum()),
                        largest=True,
                    )
                else:
                    hard_neg_sims, _ = neg_sims.topk(
                        min(self.n_hard_negatives, len(neg_sims)),
                        largest=True,
                    )
            else:
                # Just take hardest negatives
                hard_neg_sims, _ = neg_sims.topk(
                    min(self.n_hard_negatives, len(neg_sims)),
                    largest=True,
                )
            
            # Compute loss for this anchor
            pos_sim = sim_matrix[i, pos_indices].mean() / self.temperature
            neg_logsumexp = torch.logsumexp(hard_neg_sims / self.temperature, dim=0)
            
            loss_i = -pos_sim + neg_logsumexp
            losses.append(loss_i)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return torch.stack(losses).mean()

