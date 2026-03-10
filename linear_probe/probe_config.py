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
    # attn_implementation: "flash_attention_2" (A100/H100), "sdpa" (fallback)
    attn_impl: str = "flash_attention_2"

    # Probe: [] = tất cả transformer layers (full-layer cache), hoặc chỉ định subset
    layers: List[int] = field(default_factory=list)

    # Data
    train_file: str = "data/train/"
    eval_file: str = "data/test/linear_probe_test/"  # dùng test set làm validation để chọn model
    # batch size for activation caching (LLM forward pass) — A100 80GB: ~32
    cache_batch_size:  int = 32
    cache_root: str = "activation_cache"
    
    # Training hyperparameters
    window_size: int = 16 # M for SWiM smoothing
    temperature: float = 1.0 # τ for SWiM loss
    learning_rate: float = 1e-3
    weight_decay: float = 0.0 # Regularization L2 → chống overfitting
    num_epochs: int = 20
    # probe rất nhỏ (linear), A100 80GB: batch 32-64 (probe training không cần LLM)
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0 # Gradient clipping để ổn định training → tránh exploding gradients
    seed: int = 42
    save_dir: str = "checkpoints"
    early_stop_patience: int = 5
    num_workers: int = 4  # DataLoader workers for cached activation loading

    # GPU optimization
    use_amp: bool = True      # Automatic Mixed Precision (bfloat16 forward/backward)
    use_compile: bool = True  # torch.compile probe + smoother

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