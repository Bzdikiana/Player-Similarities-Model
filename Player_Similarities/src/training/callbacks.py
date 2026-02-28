"""
Training Callbacks

Callbacks for training loop:
- Early stopping
- Model checkpointing
- Learning rate scheduling
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import json
from datetime import datetime


class EarlyStopping:
    """
    Early stopping callback.
    
    Stops training when validation loss stops improving.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
        verbose: bool = True,
    ):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for metrics like accuracy
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def step(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping: {self.counter}/{self.patience}")
        
        if self.counter >= self.patience:
            self.should_stop = True
            return True
        
        return False
    
    def reset(self):
        """Reset the callback."""
        self.counter = 0
        self.best_score = None
        self.should_stop = False


class ModelCheckpoint:
    """
    Model checkpointing callback.
    
    Saves model checkpoints during training.
    """
    
    def __init__(
        self,
        dirpath: str = "checkpoints",
        filename: str = "model-{epoch:02d}-{loss:.4f}.pt",
        save_best_only: bool = True,
        save_weights_only: bool = False,
        mode: str = "min",
        verbose: bool = True,
    ):
        """
        Args:
            dirpath: Directory to save checkpoints
            filename: Checkpoint filename template
            save_best_only: Only save when validation improves
            save_weights_only: Only save model weights
            mode: 'min' or 'max' for tracking best
            verbose: Whether to print messages
        """
        self.dirpath = Path(dirpath)
        self.dirpath.mkdir(parents=True, exist_ok=True)
        
        self.filename = filename
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.mode = mode
        self.verbose = verbose
        
        self.best_score = None
        self.best_path = None
    
    def save(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        loss: float = 0.0,
        **kwargs,
    ):
        """
        Save checkpoint if conditions are met.
        
        Args:
            model: Model to save
            optimizer: Optional optimizer to save
            epoch: Current epoch
            loss: Current validation loss/metric
            **kwargs: Additional items to save
        """
        # Check if this is best
        is_best = False
        if self.best_score is None:
            is_best = True
        elif self.mode == "min" and loss < self.best_score:
            is_best = True
        elif self.mode == "max" and loss > self.best_score:
            is_best = True
        
        if self.save_best_only and not is_best:
            return
        
        if is_best:
            self.best_score = loss
        
        # Build checkpoint
        if self.save_weights_only:
            checkpoint = model.state_dict()
        else:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "loss": loss,
                "timestamp": datetime.now().isoformat(),
            }
            
            if optimizer is not None:
                checkpoint["optimizer_state_dict"] = optimizer.state_dict()
            
            checkpoint.update(kwargs)
        
        # Save
        filename = self.filename.format(epoch=epoch, loss=loss)
        filepath = self.dirpath / filename
        
        torch.save(checkpoint, filepath)
        
        if self.verbose:
            print(f"  Saved checkpoint: {filepath}")
        
        if is_best:
            self.best_path = filepath
            
            # Also save as best.pt
            best_filepath = self.dirpath / "best.pt"
            torch.save(checkpoint, best_filepath)
    
    def load_best(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Load the best checkpoint."""
        if self.best_path is None:
            best_filepath = self.dirpath / "best.pt"
        else:
            best_filepath = self.best_path
        
        if not best_filepath.exists():
            raise FileNotFoundError(f"No checkpoint found at {best_filepath}")
        
        checkpoint = torch.load(best_filepath)
        
        if self.save_weights_only:
            model.load_state_dict(checkpoint)
            return {}
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
            return checkpoint


class LearningRateScheduler:
    """
    Learning rate scheduler callback.
    
    Wraps PyTorch schedulers with additional features.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler_type: str = "cosine",
        warmup_epochs: int = 5,
        total_epochs: int = 100,
        min_lr: float = 1e-6,
        verbose: bool = True,
    ):
        """
        Args:
            optimizer: PyTorch optimizer
            scheduler_type: 'cosine', 'step', 'plateau'
            warmup_epochs: Linear warmup epochs
            total_epochs: Total training epochs
            min_lr: Minimum learning rate
            verbose: Whether to print LR changes
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.verbose = verbose
        
        # Store initial LR
        self.base_lr = optimizer.param_groups[0]["lr"]
        
        # Create scheduler
        if scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_epochs - warmup_epochs,
                eta_min=min_lr,
            )
        elif scheduler_type == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1,
            )
        elif scheduler_type == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=5,
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
        self.current_epoch = 0
    
    def step(self, epoch: int, val_loss: Optional[float] = None):
        """
        Update learning rate.
        
        Args:
            epoch: Current epoch
            val_loss: Validation loss (for plateau scheduler)
        """
        self.current_epoch = epoch
        
        # Warmup
        if epoch < self.warmup_epochs:
            warmup_lr = self.base_lr * (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = warmup_lr
            
            if self.verbose:
                print(f"  Warmup LR: {warmup_lr:.2e}")
            return
        
        # Regular scheduling
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if val_loss is not None:
                self.scheduler.step(val_loss)
        else:
            self.scheduler.step()
        
        current_lr = self.optimizer.param_groups[0]["lr"]
        
        if self.verbose:
            print(f"  Learning rate: {current_lr:.2e}")
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]


class ProgressLogger:
    """
    Training progress logging callback.
    """
    
    def __init__(
        self,
        log_file: Optional[str] = None,
        log_interval: int = 10,
    ):
        self.log_file = Path(log_file) if log_file else None
        self.log_interval = log_interval
        self.history = []
        
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def log_epoch(self, metrics: Dict[str, float]):
        """Log epoch metrics."""
        self.history.append(metrics)
        
        if self.log_file:
            with open(self.log_file, "w") as f:
                json.dump(self.history, f, indent=2)
    
    def log_batch(self, batch_idx: int, total_batches: int, loss: float):
        """Log batch progress."""
        if batch_idx % self.log_interval == 0:
            progress = (batch_idx + 1) / total_batches * 100
            print(f"  [{progress:5.1f}%] Batch {batch_idx + 1}/{total_batches} | Loss: {loss:.4f}")
