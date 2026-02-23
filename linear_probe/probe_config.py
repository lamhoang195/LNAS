from dataclasses import dataclass, field
from typing import List
import json, os

@dataclass
class ProbeConfig:
    # Model
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    dtype: str = "float16"
    max_sequence_length: int = 4096
    device: str = "cuda"

    # Probe: trống là probe trên tất cả các layer, nếu có thì chỉ probe trên các layer được chỉ định
    layers: List[int] = field(default_factory=list)

    # Data
    train_file: str = "data/train/train.json"
    eval_file: str = "data/eval/eval.json"
    cache_root: str = "activation_cache"
    
    # Training hyperparameters
    window_size: int = 16 # M for SWiM smoothing
    temperature: float = 1.0 # τ for SWiM loss
    learning_rate: float = 1e-3
    weight_decay: float = 0.0 # Regularization L2 → chống overfitting
    num_epochs: int = 20
    batch_size: int = 4
    gradient_accumulation_steps: int = 4 # Để effective batch size = batch_size * gradient_accumulation_steps
    max_grad_norm: float = 1.0 # Gradient clipping để ổn định training → tránh exploding gradients
    seed: int = 42
    save_dir: str = "checkpoints"
    early_stop_patience: int = 5

    # Inference
    ema_alpha: float = 0.1
    threshold: float = 0.5
    checkpoint_path: str = "checkpoints/best_model.pt"

    log_dir: str = "logs"

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        from dataclasses import asdict
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        
    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as f:
            return cls(**json.load(f))