"""
Training Loop and Trainer

Implements the training pipeline for player embeddings:
- Data loading and batching
- Forward pass through event GNN + Transformer
- Loss computation
- Optimization
- Logging and checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
from datetime import datetime

from .losses import InfoNCELoss, AuxiliaryLosses, compute_reliability_weights
from .metrics import EmbeddingMetrics
from .callbacks import EarlyStopping, ModelCheckpoint


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 64
    max_epochs: int = 100
    warmup_epochs: int = 5
    
    # Loss weights
    contrastive_weight: float = 1.0
    auxiliary_weight: float = 0.1
    
    # Reliability
    reliability_k: float = 50.0
    min_reliability: float = 0.1
    
    # Regularization
    dropout: float = 0.1
    gradient_clip: float = 1.0
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 1
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = True
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Mixed precision
    use_amp: bool = True


class PlayerEmbeddingModel(nn.Module):
    """
    Complete player embedding model combining:
    1. Event Graph Encoder
    2. Temporal Transformer
    3. Fusion Head
    4. Optional Global Player Graph
    """
    
    def __init__(
        self,
        event_encoder: nn.Module,
        temporal_encoder: nn.Module,
        fusion_head: nn.Module,
        global_graph: Optional[nn.Module] = None,
    ):
        super().__init__()
        
        self.event_encoder = event_encoder
        self.temporal_encoder = temporal_encoder
        self.fusion_head = fusion_head
        self.global_graph = global_graph
    
    def forward(
        self,
        event_batch: Dict[str, torch.Tensor],
        sequence_batch: Dict[str, torch.Tensor],
        player_indices: Optional[torch.Tensor] = None,
        external_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.
        
        Args:
            event_batch: Batched event graphs
            sequence_batch: Batched player sequences
            player_indices: Indices for global graph lookup
            external_features: Features for reliability head
            
        Returns:
            Dict with embeddings and auxiliary outputs
        """
        # 1. Encode events
        event_output = self.event_encoder(
            node_features=event_batch["node_features"],
            context_features=event_batch["context_features"],
            attention_mask=event_batch["attention_mask"],
        )
        event_embeddings = event_output["event_embedding"]
        
        # 2. Encode temporal sequences
        temporal_output = self.temporal_encoder(
            event_embeddings=sequence_batch["event_embeddings"],
            time_positions=sequence_batch["time_positions"],
            attention_mask=sequence_batch["attention_mask"],
        )
        player_embeddings = temporal_output["player_embedding"]
        
        # 3. Get global embeddings if available
        global_embeddings = None
        if self.global_graph is not None and player_indices is not None:
            global_embeddings = self.global_graph(player_indices)
        
        # 4. Fusion head
        fusion_output = self.fusion_head(
            data_embedding=player_embeddings,
            global_embedding=global_embeddings,
            external_features=external_features,
        )
        
        return {
            "embedding": fusion_output["embedding"],
            "reliability": fusion_output["reliability"],
            "role_logits": fusion_output["role_logits"],
            "data_embedding": fusion_output["data_embedding"],
            "event_embeddings": event_embeddings,
            "global_embedding": global_embeddings,
        }
    
    def get_player_embedding(
        self,
        event_batch: Dict[str, torch.Tensor],
        sequence_batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Get final player embedding for inference.
        """
        output = self.forward(event_batch, sequence_batch)
        return output["embedding"]


class Trainer:
    """
    Training loop manager.
    
    Handles:
    - Data iteration
    - Forward/backward passes
    - Optimization
    - Logging
    - Checkpointing
    - Early stopping
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Move model to device
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
        )
        
        # Losses
        self.contrastive_loss = InfoNCELoss(temperature=0.07)
        self.auxiliary_losses = AuxiliaryLosses()
        
        # Metrics
        self.metrics = EmbeddingMetrics()
        
        # Callbacks
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta,
        )
        
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint = ModelCheckpoint(
            dirpath=self.checkpoint_dir,
            save_best_only=config.save_best_only,
        )
        
        # Mixed precision
        self.scaler = torch.amp.GradScaler('cuda') if config.use_amp else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.history: List[Dict[str, float]] = []
    
    def train_epoch(self) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        
        epoch_losses = {
            "total_loss": 0.0,
            "contrastive_loss": 0.0,
            "aux_loss": 0.0,
        }
        n_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass with AMP
            with torch.amp.autocast('cuda', enabled=self.config.use_amp):
                outputs = self.model(
                    event_batch=batch["events"],
                    sequence_batch=batch["sequences"],
                    player_indices=batch.get("player_indices"),
                    external_features=batch.get("external_features"),
                )
                
                # Compute losses
                embeddings = outputs["embedding"]
                labels = batch["player_ids"]
                
                # Contrastive loss
                con_loss = self.contrastive_loss(embeddings, labels)
                
                # Auxiliary losses
                aux_output = self.auxiliary_losses(
                    role_logits=outputs.get("role_logits"),
                    role_labels=batch.get("role_labels"),
                    data_embeddings=outputs.get("data_embedding"),
                    global_embeddings=outputs.get("global_embedding"),
                    reliability=outputs.get("reliability"),
                )
                aux_loss = aux_output["aux_total"]
                
                # Total loss
                total_loss = (
                    self.config.contrastive_weight * con_loss +
                    self.config.auxiliary_weight * aux_loss
                )
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip,
                )
                self.optimizer.step()
            
            # Accumulate losses
            epoch_losses["total_loss"] += total_loss.item()
            epoch_losses["contrastive_loss"] += con_loss.item()
            epoch_losses["aux_loss"] += aux_loss.item()
            n_batches += 1
            
            self.global_step += 1
            
            # Log
            if batch_idx % self.config.log_interval == 0:
                print(
                    f"  Batch {batch_idx}/{len(self.train_loader)} | "
                    f"Loss: {total_loss.item():.4f}"
                )
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= max(n_batches, 1)
        
        return epoch_losses
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        
        val_losses = {
            "val_total_loss": 0.0,
            "val_contrastive_loss": 0.0,
        }
        all_embeddings = []
        all_labels = []
        n_batches = 0
        
        for batch in self.val_loader:
            batch = self._move_batch_to_device(batch)
            
            outputs = self.model(
                event_batch=batch["events"],
                sequence_batch=batch["sequences"],
            )
            
            embeddings = outputs["embedding"]
            labels = batch["player_ids"]
            
            # Loss
            con_loss = self.contrastive_loss(embeddings, labels)
            
            val_losses["val_total_loss"] += con_loss.item()
            val_losses["val_contrastive_loss"] += con_loss.item()
            n_batches += 1
            
            # Collect for metrics
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())
        
        # Average losses
        for key in val_losses:
            val_losses[key] /= max(n_batches, 1)
        
        # Compute embedding metrics
        if all_embeddings:
            all_embeddings = torch.cat(all_embeddings, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            
            metrics = self.metrics.compute_all(all_embeddings, all_labels)
            val_losses.update(metrics)
        
        return val_losses
    
    def train(self, n_epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            n_epochs: Number of epochs (overrides config if provided)
            
        Returns:
            Training history
        """
        n_epochs = n_epochs or self.config.max_epochs
        
        print(f"Starting training for {n_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(n_epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            print(f"\nEpoch {epoch + 1}/{n_epochs}")
            print("-" * 50)
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = {}
            if self.val_loader is not None and (epoch + 1) % self.config.eval_interval == 0:
                val_metrics = self.validate()
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            epoch_metrics["epoch"] = epoch
            epoch_metrics["lr"] = self.optimizer.param_groups[0]["lr"]
            epoch_metrics["time"] = time.time() - start_time
            
            self.history.append(epoch_metrics)
            
            # Print summary
            print(f"  Train Loss: {train_metrics['total_loss']:.4f}")
            if val_metrics:
                print(f"  Val Loss: {val_metrics['val_total_loss']:.4f}")
                if "recall_at_1" in val_metrics:
                    print(f"  Recall@1: {val_metrics['recall_at_1']:.4f}")
            print(f"  Time: {epoch_metrics['time']:.1f}s")
            
            # Checkpointing
            val_loss = val_metrics.get("val_total_loss", train_metrics["total_loss"])
            self.checkpoint.save(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                loss=val_loss,
            )
            
            # Early stopping
            if self.early_stopping.step(val_loss):
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break
        
        # Save final model
        self.save_checkpoint("final_model.pt")
        
        return self._format_history()
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch tensors to device."""
        moved = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device)
            elif isinstance(value, dict):
                moved[key] = self._move_batch_to_device(value)
            else:
                moved[key] = value
        return moved
    
    def _format_history(self) -> Dict[str, List[float]]:
        """Format history as dict of lists."""
        if not self.history:
            return {}
        
        keys = self.history[0].keys()
        return {key: [h[key] for h in self.history] for key in keys}
    
    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "history": self.history,
            "config": self.config.__dict__,
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.history = checkpoint.get("history", [])
    
    def export_embeddings(
        self,
        data_loader: DataLoader,
        output_path: str,
    ) -> Dict[int, torch.Tensor]:
        """
        Export player embeddings for all players in data loader.
        
        Returns:
            Dict mapping player_id -> embedding tensor
        """
        self.model.eval()
        
        embeddings_dict = {}
        
        with torch.no_grad():
            for batch in data_loader:
                batch = self._move_batch_to_device(batch)
                
                outputs = self.model(
                    event_batch=batch["events"],
                    sequence_batch=batch["sequences"],
                )
                
                embeddings = outputs["embedding"].cpu()
                player_ids = batch["player_ids"].cpu().tolist()
                
                for pid, emb in zip(player_ids, embeddings):
                    embeddings_dict[pid] = emb
        
        # Save
        torch.save(embeddings_dict, output_path)
        
        return embeddings_dict


def create_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    **config_kwargs,
) -> Trainer:
    """
    Factory function to create a Trainer with common settings.
    """
    config = TrainingConfig(**config_kwargs)
    return Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
    )
