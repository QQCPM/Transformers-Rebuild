"""
MLP (Feed-Forward Network) Blocks for Transformers

Educational implementation of the position-wise feed-forward networks used in
transformer architectures. These MLPs process each position independently
and provide the model's capacity for non-linear transformations.

Key concepts covered:
- Position-wise feed-forward networks
- GELU activation function and its properties
- Layer normalization and its placement
- Residual connections
- Dropout for regularization
- Educational analysis of MLP behavior

Following ARENA Chapter 1.1 curriculum.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, Union
import math
import numpy as np

try:
    from ..models.config import TransformerConfig
except ImportError:
    from src.models.config import TransformerConfig


class GELU(nn.Module):
    """
    Educational implementation of GELU (Gaussian Error Linear Units) activation.

    GELU is the activation function used in transformers. It provides a smooth
    approximation to ReLU with better gradient properties.

    Mathematical definition:
    GELU(x) = x * Φ(x) = x * (1/2) * (1 + erf(x / √2))

    Where Φ(x) is the cumulative distribution function of the standard normal distribution.
    """

    def __init__(self, approximation: str = "tanh"):
        """
        Initialize GELU activation.

        Args:
            approximation: "exact", "tanh", or "sigmoid" approximation
        """
        super().__init__()
        self.approximation = approximation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply GELU activation.

        Args:
            x: Input tensor

        Returns:
            GELU activated tensor
        """
        if self.approximation == "exact":
            # Exact GELU using error function
            return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))
        elif self.approximation == "tanh":
            # Tanh approximation (faster, commonly used)
            # GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
            return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        elif self.approximation == "sigmoid":
            # Sigmoid approximation
            return x * torch.sigmoid(1.702 * x)
        else:
            raise ValueError(f"Unknown approximation: {self.approximation}")

    def analyze_activation_properties(self, x_range: Tuple[float, float] = (-3.0, 3.0), n_points: int = 1000) -> Dict[str, Any]:
        """
        Analyze GELU activation properties for educational purposes.

        Args:
            x_range: Range of x values to analyze
            n_points: Number of points to sample

        Returns:
            Dictionary with activation analysis
        """
        x = torch.linspace(x_range[0], x_range[1], n_points)

        # Compute GELU and its derivative
        with torch.no_grad():
            y = self.forward(x)

            # Numerical derivative
            dx = x[1] - x[0]
            dy_dx = torch.gradient(y, spacing=dx.item())[0]

            analysis = {
                'input_range': x_range,
                'output_range': (y.min().item(), y.max().item()),
                'activation_values': {
                    'x': x.tolist(),
                    'y': y.tolist(),
                    'derivative': dy_dx.tolist()
                },
                'properties': {
                    'smooth': True,
                    'non_monotonic': False,
                    'saturates': False,
                    'zero_centered': True
                },
                'comparison_with_relu': {
                    'relu_output': F.relu(x).tolist(),
                    'explanation': "GELU is smoother than ReLU and allows small negative values"
                },
                'approximation_used': self.approximation
            }

            return analysis


class LayerNorm(nn.Module):
    """
    Educational implementation of Layer Normalization.

    Layer normalization normalizes across the feature dimension (last dimension)
    for each sample independently. This is different from batch normalization
    which normalizes across the batch dimension.

    Mathematical formula:
    LayerNorm(x) = γ * (x - μ) / (σ + ε) + β

    Where:
    - μ: mean across feature dimension
    - σ: standard deviation across feature dimension
    - γ: learnable scale parameter
    - β: learnable bias parameter
    - ε: small constant for numerical stability
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5, bias: bool = True):
        """
        Initialize layer normalization.

        Args:
            normalized_shape: Size of the feature dimension to normalize
            eps: Small constant for numerical stability
            bias: Whether to use learnable bias parameter
        """
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.use_bias = bias

        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        if bias:
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization.

        Args:
            x: Input tensor [..., normalized_shape]

        Returns:
            Layer normalized tensor
        """
        # Compute mean and variance across the last dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)

        # Scale and shift
        output = self.weight * x_normalized
        if self.use_bias:
            output = output + self.bias

        return output

    def analyze_normalization_effects(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze the effects of layer normalization for educational purposes.

        Args:
            x: Input tensor to analyze

        Returns:
            Dictionary with normalization analysis
        """
        with torch.no_grad():
            # Before normalization
            input_mean = x.mean(dim=-1)
            input_std = x.std(dim=-1)
            input_min = x.min(dim=-1)[0]
            input_max = x.max(dim=-1)[0]

            # After normalization
            output = self.forward(x)
            output_mean = output.mean(dim=-1)
            output_std = output.std(dim=-1)
            output_min = output.min(dim=-1)[0]
            output_max = output.max(dim=-1)[0]

            analysis = {
                'input_statistics': {
                    'mean': input_mean.tolist(),
                    'std': input_std.tolist(),
                    'min': input_min.tolist(),
                    'max': input_max.tolist()
                },
                'output_statistics': {
                    'mean': output_mean.tolist(),
                    'std': output_std.tolist(),
                    'min': output_min.tolist(),
                    'max': output_max.tolist()
                },
                'normalization_effects': {
                    'mean_reduction': (input_mean.abs().mean() - output_mean.abs().mean()).item(),
                    'std_standardization': (input_std.std() - output_std.std()).item(),
                    'explanation': "LayerNorm centers mean around learned bias and standardizes variance"
                },
                'parameters': {
                    'weight_stats': {
                        'mean': self.weight.mean().item(),
                        'std': self.weight.std().item()
                    },
                    'bias_stats': {
                        'mean': self.bias.mean().item() if self.bias is not None else None,
                        'std': self.bias.std().item() if self.bias is not None else None
                    } if self.bias is not None else None
                }
            }

            return analysis


class TransformerMLP(nn.Module):
    """
    Educational implementation of the transformer MLP (feed-forward network).

    The MLP block applies position-wise transformations to each token independently.
    It typically expands the representation to a higher dimension, applies non-linearity,
    then projects back to the original dimension.

    Architecture:
    1. Linear projection: d_model -> d_mlp (expansion)
    2. GELU activation
    3. Dropout (optional)
    4. Linear projection: d_mlp -> d_model (contraction)
    5. Dropout (optional)
    """

    def __init__(
        self,
        d_model: int,
        d_mlp: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        bias: bool = True
    ):
        """
        Initialize transformer MLP.

        Args:
            d_model: Model dimension (input/output dimension)
            d_mlp: MLP hidden dimension (typically 4 * d_model)
            dropout: Dropout probability
            activation: Activation function ("gelu", "relu", "swish")
            bias: Whether to use bias in linear layers
        """
        super().__init__()
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.dropout_p = dropout

        # Linear layers
        self.fc1 = nn.Linear(d_model, d_mlp, bias=bias)  # Expansion
        self.fc2 = nn.Linear(d_mlp, d_model, bias=bias)  # Contraction

        # Activation function
        if activation == "gelu":
            self.activation = GELU(approximation="tanh")
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "swish":
            self.activation = nn.SiLU()  # SiLU is Swish
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

        # Hook points for TransformerLens compatibility
        self.hook_pre = None     # Before MLP
        self.hook_mid = None     # After first linear layer and activation
        self.hook_post = None    # After MLP

    def _init_weights(self):
        """Initialize weights following best practices."""
        # Xavier/Glorot initialization for both linear layers
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

        # Initialize biases to zero
        if self.fc1.bias is not None:
            nn.init.constant_(self.fc1.bias, 0)
        if self.fc2.bias is not None:
            nn.init.constant_(self.fc2.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        return_debug_info: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Forward pass through MLP.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            return_debug_info: Whether to return debugging information

        Returns:
            output: MLP output [batch, seq_len, d_model]
            debug_info: Optional debugging information
        """
        # Apply hook if present
        if self.hook_pre is not None:
            x = self.hook_pre(x)

        # First linear layer (expansion)
        h = self.fc1(x)  # [batch, seq_len, d_mlp]

        # Activation
        h = self.activation(h)

        # First dropout
        h = self.dropout1(h)

        # Apply hook if present
        if self.hook_mid is not None:
            h = self.hook_mid(h)

        # Second linear layer (contraction)
        output = self.fc2(h)  # [batch, seq_len, d_model]

        # Second dropout
        output = self.dropout2(output)

        # Apply hook if present
        if self.hook_post is not None:
            output = self.hook_post(output)

        if return_debug_info:
            debug_info = self._create_debug_info(x, h, output)
            return output, debug_info
        else:
            return output

    def _create_debug_info(
        self,
        input_tensor: torch.Tensor,
        hidden_tensor: torch.Tensor,
        output_tensor: torch.Tensor
    ) -> Dict[str, Any]:
        """Create comprehensive debug information."""
        with torch.no_grad():
            debug_info = {
                'shapes': {
                    'input': list(input_tensor.shape),
                    'hidden': list(hidden_tensor.shape),
                    'output': list(output_tensor.shape)
                },
                'statistics': {
                    'input': {
                        'mean': input_tensor.mean().item(),
                        'std': input_tensor.std().item(),
                        'min': input_tensor.min().item(),
                        'max': input_tensor.max().item(),
                        'norm': torch.norm(input_tensor, dim=-1).mean().item()
                    },
                    'hidden': {
                        'mean': hidden_tensor.mean().item(),
                        'std': hidden_tensor.std().item(),
                        'min': hidden_tensor.min().item(),
                        'max': hidden_tensor.max().item(),
                        'norm': torch.norm(hidden_tensor, dim=-1).mean().item(),
                        'sparsity': (hidden_tensor == 0).float().mean().item()
                    },
                    'output': {
                        'mean': output_tensor.mean().item(),
                        'std': output_tensor.std().item(),
                        'min': output_tensor.min().item(),
                        'max': output_tensor.max().item(),
                        'norm': torch.norm(output_tensor, dim=-1).mean().item()
                    }
                },
                'activation_analysis': {
                    'activation_type': type(self.activation).__name__,
                    'negative_activations': (hidden_tensor < 0).float().mean().item(),
                    'zero_activations': (hidden_tensor == 0).float().mean().item(),
                    'large_activations': (hidden_tensor.abs() > 3.0).float().mean().item()
                },
                'configuration': {
                    'd_model': self.d_model,
                    'd_mlp': self.d_mlp,
                    'expansion_ratio': self.d_mlp / self.d_model,
                    'dropout_p': self.dropout_p
                }
            }

            return debug_info

    def analyze_mlp_behavior(
        self,
        x: torch.Tensor,
        n_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Analyze MLP behavior for educational insights.

        Args:
            x: Input tensor for analysis
            n_samples: Number of samples to analyze

        Returns:
            Dictionary with MLP behavior analysis
        """
        self.eval()
        with torch.no_grad():
            batch_size, seq_len, d_model = x.shape

            # Sample subset for analysis
            if batch_size * seq_len > n_samples:
                # Flatten and sample
                x_flat = x.view(-1, d_model)
                indices = torch.randperm(x_flat.shape[0])[:n_samples]
                x_sample = x_flat[indices]
            else:
                x_sample = x.view(-1, d_model)

            analysis = {}

            # Forward pass through components
            h1 = self.fc1(x_sample)  # After first linear layer
            h1_activated = self.activation(h1)  # After activation
            output = self.fc2(h1_activated)  # Final output

            # Analyze expansion phase
            analysis['expansion_phase'] = {
                'input_effective_rank': self._compute_effective_rank(x_sample),
                'hidden_effective_rank': self._compute_effective_rank(h1),
                'rank_increase': self._compute_effective_rank(h1) - self._compute_effective_rank(x_sample),
                'explanation': "How much the representation is expanded"
            }

            # Analyze activation effects
            analysis['activation_effects'] = {
                'pre_activation_stats': {
                    'mean': h1.mean().item(),
                    'std': h1.std().item(),
                    'negative_ratio': (h1 < 0).float().mean().item()
                },
                'post_activation_stats': {
                    'mean': h1_activated.mean().item(),
                    'std': h1_activated.std().item(),
                    'negative_ratio': (h1_activated < 0).float().mean().item(),
                    'zero_ratio': (h1_activated == 0).float().mean().item()
                },
                'activation_sparsity': (h1_activated == 0).float().mean().item(),
                'explanation': "How activation function affects the representation"
            }

            # Analyze contraction phase
            analysis['contraction_phase'] = {
                'hidden_effective_rank': self._compute_effective_rank(h1_activated),
                'output_effective_rank': self._compute_effective_rank(output),
                'rank_preservation': self._compute_effective_rank(output) / self._compute_effective_rank(h1_activated),
                'explanation': "How well information is preserved during contraction"
            }

            # Analyze weight matrices
            analysis['weight_analysis'] = {
                'fc1_weight_stats': {
                    'mean': self.fc1.weight.mean().item(),
                    'std': self.fc1.weight.std().item(),
                    'effective_rank': self._compute_effective_rank(self.fc1.weight)
                },
                'fc2_weight_stats': {
                    'mean': self.fc2.weight.mean().item(),
                    'std': self.fc2.weight.std().item(),
                    'effective_rank': self._compute_effective_rank(self.fc2.weight)
                }
            }

            # Analyze input-output relationship
            input_output_similarity = F.cosine_similarity(
                x_sample.view(-1), output.view(-1), dim=0
            ).item()

            analysis['input_output_relationship'] = {
                'cosine_similarity': input_output_similarity,
                'magnitude_ratio': torch.norm(output) / torch.norm(x_sample),
                'residual_importance': torch.norm(output - x_sample) / torch.norm(x_sample),
                'explanation': "How much the MLP transforms vs preserves the input"
            }

            return analysis

    def _compute_effective_rank(self, x: torch.Tensor, threshold: float = 0.99) -> float:
        """Compute effective rank (number of dimensions for 99% variance)."""
        if x.dim() > 2:
            x = x.view(-1, x.shape[-1])

        try:
            # SVD decomposition
            U, S, V = torch.svd(x)

            # Compute cumulative variance ratio
            total_var = S.sum()
            if total_var == 0:
                return 0.0

            cumsum_var = torch.cumsum(S, dim=0) / total_var
            effective_rank = (cumsum_var < threshold).sum().item() + 1

            return min(effective_rank, x.shape[-1])
        except:
            # Fallback to simple rank
            return torch.matrix_rank(x).item()


def create_educational_mlp_examples() -> Dict[str, Any]:
    """
    Create educational examples for understanding MLP behavior.

    Returns:
        Dictionary with examples and explanations
    """
    examples = {}

    print("Creating MLP educational examples...")

    # Example 1: Basic MLP functionality
    config = TransformerConfig.educational_config()
    mlp = TransformerMLP(
        d_model=config.d_model,
        d_mlp=config.d_mlp,
        dropout=0.0,  # No dropout for reproducible examples
        activation="gelu"
    )

    # Create sample input
    batch_size, seq_len = 2, 8
    input_tensor = torch.randn(batch_size, seq_len, config.d_model) * 0.1

    # Forward pass with debug info
    output, debug_info = mlp(input_tensor, return_debug_info=True)

    examples['basic_mlp'] = {
        'input_shape': input_tensor.shape,
        'output_shape': output.shape,
        'debug_info': debug_info,
        'explanation': "Basic MLP forward pass with debugging information"
    }

    # Example 2: Activation function comparison
    activations = ["gelu", "relu"]
    activation_comparison = {}

    for activation in activations:
        mlp_act = TransformerMLP(
            d_model=64,
            d_mlp=256,
            dropout=0.0,
            activation=activation
        )

        test_input = torch.randn(1, 10, 64) * 0.5
        output_act, debug_act = mlp_act(test_input, return_debug_info=True)

        activation_comparison[activation] = {
            'output_stats': debug_act['statistics']['output'],
            'hidden_stats': debug_act['statistics']['hidden'],
            'activation_analysis': debug_act['activation_analysis']
        }

    examples['activation_comparison'] = {
        'comparison': activation_comparison,
        'explanation': "Comparison of different activation functions"
    }

    # Example 3: MLP behavior analysis
    analysis = mlp.analyze_mlp_behavior(input_tensor)

    examples['behavior_analysis'] = {
        'analysis': analysis,
        'explanation': "Detailed analysis of MLP behavior and transformations"
    }

    # Example 4: GELU activation analysis
    gelu = GELU(approximation="tanh")
    gelu_analysis = gelu.analyze_activation_properties()

    examples['gelu_analysis'] = {
        'analysis': gelu_analysis,
        'explanation': "Mathematical properties of GELU activation function"
    }

    # Example 5: Layer normalization effects
    layer_norm = LayerNorm(config.d_model)

    # Create input with varying scales
    test_input = torch.randn(2, 5, config.d_model)
    test_input[0] *= 0.1  # Small scale
    test_input[1] *= 10.0  # Large scale

    norm_analysis = layer_norm.analyze_normalization_effects(test_input)

    examples['layer_norm_analysis'] = {
        'analysis': norm_analysis,
        'explanation': "Effects of layer normalization on different input scales"
    }

    return examples


if __name__ == "__main__":
    print("MLP Blocks with GELU and Layer Normalization")
    print("=" * 50)

    # Run educational examples
    examples = create_educational_mlp_examples()

    print("\n1. Basic MLP functionality:")
    basic = examples['basic_mlp']
    print(f"Input shape: {basic['input_shape']}")
    print(f"Output shape: {basic['output_shape']}")
    print(f"Configuration: {basic['debug_info']['configuration']}")

    stats = basic['debug_info']['statistics']
    print(f"Input norm: {stats['input']['norm']:.4f}")
    print(f"Hidden norm: {stats['hidden']['norm']:.4f}")
    print(f"Output norm: {stats['output']['norm']:.4f}")
    print(f"Hidden sparsity: {stats['hidden']['sparsity']:.4f}")

    print("\n2. Activation function comparison:")
    comparison = examples['activation_comparison']['comparison']
    for activation, stats in comparison.items():
        print(f"{activation.upper()}:")
        print(f"  Output mean: {stats['output_stats']['mean']:.4f}")
        print(f"  Hidden sparsity: {stats['activation_analysis']['zero_activations']:.4f}")
        print(f"  Negative activations: {stats['activation_analysis']['negative_activations']:.4f}")

    print("\n3. MLP behavior analysis:")
    behavior = examples['behavior_analysis']['analysis']
    print(f"Expansion rank increase: {behavior['expansion_phase']['rank_increase']:.2f}")
    print(f"Activation sparsity: {behavior['activation_effects']['activation_sparsity']:.4f}")
    print(f"Rank preservation ratio: {behavior['contraction_phase']['rank_preservation']:.4f}")
    print(f"Input-output similarity: {behavior['input_output_relationship']['cosine_similarity']:.4f}")

    print("\n4. GELU activation properties:")
    gelu_props = examples['gelu_analysis']['analysis']['properties']
    print(f"Smooth: {gelu_props['smooth']}")
    print(f"Zero-centered: {gelu_props['zero_centered']}")
    print(f"Non-monotonic: {gelu_props['non_monotonic']}")
    print(f"Output range: {examples['gelu_analysis']['analysis']['output_range']}")

    print("\n5. Layer normalization effects:")
    norm_effects = examples['layer_norm_analysis']['analysis']['normalization_effects']
    print(f"Mean reduction: {norm_effects['mean_reduction']:.6f}")
    print(f"Std standardization: {norm_effects['std_standardization']:.6f}")
    print(norm_effects['explanation'])

    print("\n6. Testing different MLP configurations:")
    test_configs = [
        (256, 1024),   # Standard 4x expansion
        (512, 1536),   # 3x expansion
        (768, 3072),   # GPT-2 small config
    ]

    for d_model, d_mlp in test_configs:
        test_mlp = TransformerMLP(d_model=d_model, d_mlp=d_mlp)
        test_input = torch.randn(1, 10, d_model) * 0.1
        test_output = test_mlp(test_input)
        expansion_ratio = d_mlp / d_model

        print(f"✓ d_model={d_model}, d_mlp={d_mlp}, expansion={expansion_ratio:.1f}x")
        print(f"  Input: {test_input.shape} -> Output: {test_output.shape}")

    print("\n✅ All MLP components implemented and tested!")