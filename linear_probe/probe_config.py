from dataclasses import dataclass, field
from typing import List
import json, os

@dataclass
class ProbeConfig:
    # Model
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    dtype: str = "bfloat16"
    max_sequence_length: int = 8192
    device: str = "cuda"

    # Probe: trống là probe trên tất cả các layer, nếu có thì chỉ probe trên các layer được chỉ định
    layers: List[int] = field(default_factory=list)

    # Data
    train_file: str = "data/train/"
    eval_file: str = ""  # empty → auto split per file from train_file
    train_ratio: float = 0.80   # fraction of each file used for training
    eval_ratio:  float = 0.20   # fraction of each file used for evaluation
    max_train_samples: int = -1  # -1 = no cap
    max_eval_samples:  int = -1  # -1 = no cap
    cache_batch_size:  int = 8   # batch size for activation caching (LLM forward pass)
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

    # Eval
    eval_interval: int = 1  # run eval every N epochs

    # Loss type: "softmax_weighted" | "cummax" | "annealed_cummax"
    loss_type: str = "softmax_weighted"

    # Inference
    ema_alpha: float = 0.1
    threshold: float = 0.5
    checkpoint_path: str = "checkpoints/linear_probe_model.pt"

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