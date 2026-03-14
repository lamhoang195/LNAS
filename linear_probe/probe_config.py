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
    # "flash_attention_2" on supported GPUs, otherwise train.py falls back to "sdpa".
    attn_impl: str = "flash_attention_2"

    # Empty list means: concatenate all transformer hidden states except embeddings.
    layers: List[int] = field(default_factory=list)

    # Data
    train_file: str = "data/train/"
    eval_file: str = "data/test/linear_probe_test/"
    # Fraction of train data to use (instruction 5.2: ~25% often sufficient).
    train_fraction: float = 1.0

    # Training hyperparameters
    window_size: int = 16
    temperature: float = 1.0
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    num_epochs: int = 20
    # Batch size is shared by tokenization, activation collection and probe training.
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    seed: int = 42
    save_dir: str = "checkpoints"
    early_stop_patience: int = 5
    num_workers: int = 2

    # GPU optimization
    use_amp: bool = True
    use_compile: bool = True

    # Eval
    eval_interval: int = 1

    # "softmax_weighted" is the main loss from the write-up.
    loss_type: str = "softmax_weighted"

    # Inference / evaluation
    ema_alpha: float = 0.1
    threshold: float = 0.5
    checkpoint_path: str = "checkpoints/best_probe.pt"

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