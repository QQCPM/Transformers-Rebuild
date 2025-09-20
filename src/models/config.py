"""
Model Configuration for Educational Transformer

This configuration follows ARENA Chapter 1.1 guidelines and is designed for
educational understanding and mechanistic interpretability research.
"""

from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class TransformerConfig:
    """
    Configuration class for educational transformer implementation.

    Designed to be compatible with TransformerLens and ARENA curriculum.
    All parameters are chosen for educational clarity and interpretability.
    """

    # Model Architecture
    d_model: int = 512              # Hidden dimension / embedding dimension
    n_heads: int = 8                # Number of attention heads
    n_layers: int = 6               # Number of transformer blocks
    d_mlp: int = 2048              # MLP hidden dimension (typically 4 * d_model)
    d_head: int = 64               # Dimension per attention head (d_model / n_heads)

    # Vocabulary and Sequence
    vocab_size: int = 50257         # GPT-2 vocabulary size for compatibility
    max_position_embeddings: int = 1024  # Maximum sequence length

    # Regularization
    dropout: float = 0.1            # Dropout probability
    attention_dropout: float = 0.1   # Attention-specific dropout

    # Activation Functions
    activation_function: str = "gelu"  # Activation function for MLP

    # Layer Normalization
    layer_norm_eps: float = 1e-5    # Layer normalization epsilon

    # Initialization
    initializer_range: float = 0.02  # Standard deviation for weight initialization

    # Training Configuration
    learning_rate: float = 1e-4     # Learning rate
    weight_decay: float = 0.01      # Weight decay for regularization

    # Device Configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Educational Flags
    use_cache: bool = True          # Cache attention patterns for analysis
    output_attentions: bool = True  # Return attention weights
    output_hidden_states: bool = True  # Return all hidden states

    # TransformerLens Compatibility
    use_attn_result: bool = True    # Include attention result in residual stream
    use_split_qkv_input: bool = True  # Split QKV for easier analysis
    use_hook_points: bool = True    # Enable hook points for interpretability

    def __post_init__(self):
        """Validate and compute derived parameters."""
        # Ensure d_head is consistent with d_model and n_heads
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")

        # Set d_head based on d_model and n_heads if not explicitly set
        expected_d_head = self.d_model // self.n_heads
        if self.d_head != expected_d_head:
            print(f"Warning: d_head ({self.d_head}) doesn't match d_model//n_heads ({expected_d_head}). Using calculated value.")
            self.d_head = expected_d_head

    @classmethod
    def small_config(cls) -> 'TransformerConfig':
        """Small configuration for quick experimentation and debugging."""
        return cls(
            d_model=256,
            n_heads=4,
            n_layers=2,
            d_mlp=1024,
            max_position_embeddings=512,
            vocab_size=1000  # Smaller vocab for quick testing
        )

    @classmethod
    def gpt2_small_config(cls) -> 'TransformerConfig':
        """Configuration matching GPT-2 small for compatibility."""
        return cls(
            d_model=768,
            n_heads=12,
            n_layers=12,
            d_mlp=3072,
            max_position_embeddings=1024,
            vocab_size=50257
        )

    @classmethod
    def educational_config(cls) -> 'TransformerConfig':
        """Configuration optimized for educational understanding."""
        return cls(
            d_model=512,
            n_heads=8,
            n_layers=4,
            d_mlp=2048,
            max_position_embeddings=256,
            vocab_size=10000,
            dropout=0.0,  # No dropout for clearer analysis
            attention_dropout=0.0
        )

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'TransformerConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


# Default configuration for the project
DEFAULT_CONFIG = TransformerConfig.educational_config()


# Utility function for getting model size
def get_model_size(config: TransformerConfig) -> dict:
    """
    Calculate approximate model size and parameter count.

    Args:
        config: Transformer configuration

    Returns:
        Dictionary with model size information
    """
    # Embedding parameters
    embedding_params = config.vocab_size * config.d_model
    position_embedding_params = config.max_position_embeddings * config.d_model

    # Transformer block parameters (per layer)
    # Attention: Q, K, V projections + output projection
    attention_params = 4 * config.d_model * config.d_model
    # MLP: two linear layers
    mlp_params = 2 * config.d_model * config.d_mlp
    # Layer norms: 2 per block (pre-attention and pre-MLP)
    layernorm_params = 2 * config.d_model

    transformer_block_params = attention_params + mlp_params + layernorm_params
    total_transformer_params = transformer_block_params * config.n_layers

    # Final layer norm
    final_layernorm_params = config.d_model

    # Output head (if tied with embeddings, this is 0)
    output_head_params = config.vocab_size * config.d_model

    total_params = (
        embedding_params +
        position_embedding_params +
        total_transformer_params +
        final_layernorm_params +
        output_head_params
    )

    return {
        "total_parameters": total_params,
        "embedding_parameters": embedding_params,
        "position_embedding_parameters": position_embedding_params,
        "transformer_parameters": total_transformer_params,
        "parameters_per_layer": transformer_block_params,
        "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
    }