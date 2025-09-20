"""
Multi-Head Attention Mechanism

Educational implementation of the multi-head attention mechanism, the core
component of transformer architectures. This module provides both the
implementation and extensive educational tools for understanding attention.

Key concepts covered:
- Multi-head attention architecture
- Query, Key, Value projections
- Scaled dot-product attention
- Causal masking for autoregressive models
- Attention pattern analysis and visualization
- Hook points for TransformerLens compatibility

Following ARENA Chapter 1.1 curriculum.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, Union
import math
import numpy as np

try:
    from ..foundations.attention_math import AttentionMathematics
    from ..foundations.linear_algebra import (
        create_causal_mask,
        tensor_reshaping_for_multihead,
        tensor_reshaping_from_multihead
    )
    from ..models.config import TransformerConfig
except ImportError:
    from src.foundations.attention_math import AttentionMathematics
    from src.foundations.linear_algebra import (
        create_causal_mask,
        tensor_reshaping_for_multihead,
        tensor_reshaping_from_multihead
    )
    from src.models.config import TransformerConfig


class MultiHeadAttention(nn.Module):
    """
    Educational multi-head attention implementation.

    This implementation prioritizes clarity and educational value while
    maintaining compatibility with TransformerLens for interpretability.

    Architecture:
    1. Linear projections to create Q, K, V
    2. Split into multiple attention heads
    3. Scaled dot-product attention per head
    4. Concatenate head outputs
    5. Final linear projection
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        is_causal: bool = True
    ):
        """
        Initialize multi-head attention.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias in linear layers
            is_causal: Whether to use causal (autoregressive) attention
        """
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.is_causal = is_causal
        self.dropout_p = dropout

        # Scale factor for attention scores
        self.scale = 1.0 / math.sqrt(self.d_head)

        # Linear projections for Q, K, V (combined for efficiency)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        # Dropout for attention weights
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

        # Hook points for TransformerLens compatibility
        self.hook_q = None  # Query hook point
        self.hook_k = None  # Key hook point
        self.hook_v = None  # Value hook point
        self.hook_attn_scores = None  # Attention scores hook point
        self.hook_attn_weights = None  # Attention weights hook point
        self.hook_attn_out = None  # Attention output hook point

    def _init_weights(self):
        """Initialize weights following best practices."""
        # Xavier/Glorot initialization for QKV projection
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        if self.qkv_proj.bias is not None:
            nn.init.constant_(self.qkv_proj.bias, 0)

        # Xavier/Glorot initialization for output projection
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = True,
        return_debug_info: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict]]:
        """
        Forward pass of multi-head attention.

        Args:
            hidden_states: Input tensor [batch, seq_len, d_model]
            attention_mask: Optional attention mask [batch, seq_len, seq_len]
            return_attention_weights: Whether to return attention weights
            return_debug_info: Whether to return detailed debug information

        Returns:
            output: Attention output [batch, seq_len, d_model]
            attention_weights: Optional attention weights [batch, n_heads, seq_len, seq_len]
            debug_info: Optional debug information dictionary
        """
        batch_size, seq_len, d_model = hidden_states.shape

        # Step 1: Compute Q, K, V projections
        qkv = self.qkv_proj(hidden_states)  # [batch, seq_len, 3 * d_model]

        # Split into Q, K, V
        query, key, value = torch.chunk(qkv, 3, dim=-1)  # Each: [batch, seq_len, d_model]

        # Apply hooks if present (TransformerLens compatibility)
        if self.hook_q is not None:
            query = self.hook_q(query)
        if self.hook_k is not None:
            key = self.hook_k(key)
        if self.hook_v is not None:
            value = self.hook_v(value)

        # Step 2: Reshape for multi-head attention
        # [batch, seq_len, d_model] -> [batch, n_heads, seq_len, d_head]
        query = self._reshape_for_multihead(query)
        key = self._reshape_for_multihead(key)
        value = self._reshape_for_multihead(value)

        # Step 3: Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1))  # [batch, n_heads, seq_len, seq_len]
        attention_scores = attention_scores * self.scale

        # Apply hooks for attention scores
        if self.hook_attn_scores is not None:
            attention_scores = self.hook_attn_scores(attention_scores)

        # Step 4: Apply causal mask if needed
        if self.is_causal:
            causal_mask = create_causal_mask(seq_len, hidden_states.device)
            attention_scores = attention_scores.masked_fill(causal_mask == 0, -1e9)

        # Step 5: Apply additional attention mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 3:  # [batch, seq_len, seq_len]
                attention_mask = attention_mask.unsqueeze(1)  # [batch, 1, seq_len, seq_len]
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)

        # Step 6: Apply softmax to get attention probabilities
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply hooks for attention weights
        if self.hook_attn_weights is not None:
            attention_weights = self.hook_attn_weights(attention_weights)

        # Step 7: Apply attention to values
        attn_output = torch.matmul(attention_weights, value)  # [batch, n_heads, seq_len, d_head]

        # Apply hooks for attention output
        if self.hook_attn_out is not None:
            attn_output = self.hook_attn_out(attn_output)

        # Step 8: Concatenate heads
        attn_output = self._reshape_from_multihead(attn_output)  # [batch, seq_len, d_model]

        # Step 9: Apply output projection
        output = self.out_proj(attn_output)

        # Prepare return values
        if return_debug_info:
            debug_info = self._create_debug_info(
                query, key, value, attention_scores, attention_weights, attn_output, output
            )
            if return_attention_weights:
                return output, attention_weights, debug_info
            else:
                return output, debug_info
        elif return_attention_weights:
            return output, attention_weights
        else:
            return output

    def _reshape_for_multihead(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reshape tensor for multi-head computation."""
        batch_size, seq_len, d_model = tensor.shape
        return tensor.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

    def _reshape_from_multihead(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reshape tensor back from multi-head format."""
        batch_size, n_heads, seq_len, d_head = tensor.shape
        return tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def _create_debug_info(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_scores: torch.Tensor,
        attention_weights: torch.Tensor,
        attn_output: torch.Tensor,
        final_output: torch.Tensor
    ) -> Dict[str, Any]:
        """Create comprehensive debug information."""
        with torch.no_grad():
            debug_info = {
                'shapes': {
                    'query': list(query.shape),
                    'key': list(key.shape),
                    'value': list(value.shape),
                    'attention_scores': list(attention_scores.shape),
                    'attention_weights': list(attention_weights.shape),
                    'attn_output': list(attn_output.shape),
                    'final_output': list(final_output.shape)
                },
                'statistics': {
                    'attention_scores': {
                        'mean': attention_scores.mean().item(),
                        'std': attention_scores.std().item(),
                        'min': attention_scores.min().item(),
                        'max': attention_scores.max().item()
                    },
                    'attention_weights': {
                        'mean': attention_weights.mean().item(),
                        'std': attention_weights.std().item(),
                        'min': attention_weights.min().item(),
                        'max': attention_weights.max().item(),
                        'entropy': self._compute_attention_entropy(attention_weights)
                    },
                    'output': {
                        'mean': final_output.mean().item(),
                        'std': final_output.std().item(),
                        'norm': torch.norm(final_output, dim=-1).mean().item()
                    }
                },
                'configuration': {
                    'd_model': self.d_model,
                    'n_heads': self.n_heads,
                    'd_head': self.d_head,
                    'is_causal': self.is_causal,
                    'dropout_p': self.dropout_p
                }
            }
            return debug_info

    def _compute_attention_entropy(self, attention_weights: torch.Tensor) -> float:
        """Compute average attention entropy across all heads and positions."""
        # Add small epsilon to avoid log(0)
        log_attn = torch.log(attention_weights + 1e-12)
        entropy = -(attention_weights * log_attn).sum(dim=-1)
        return entropy.mean().item()

    def analyze_attention_patterns(
        self,
        hidden_states: torch.Tensor,
        token_labels: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Analyze attention patterns for educational insights.

        Args:
            hidden_states: Input tensor [batch, seq_len, d_model]
            token_labels: Optional list of token strings for visualization

        Returns:
            Dictionary with detailed attention analysis
        """
        self.eval()
        with torch.no_grad():
            output, attention_weights, debug_info = self.forward(
                hidden_states, return_attention_weights=True, return_debug_info=True
            )

            analysis = {}

            # Basic pattern analysis
            batch_size, n_heads, seq_len, _ = attention_weights.shape

            # Head-wise analysis
            head_analysis = {}
            for head_idx in range(n_heads):
                head_attn = attention_weights[:, head_idx, :, :]  # [batch, seq_len, seq_len]

                # Attention distance (how far each position looks)
                positions = torch.arange(seq_len, dtype=torch.float, device=hidden_states.device)
                pos_diff = positions.unsqueeze(0) - positions.unsqueeze(1)
                avg_distance = (head_attn.mean(0) * pos_diff.abs()).sum(dim=-1).mean().item()

                # Self-attention strength
                self_attention = torch.diagonal(head_attn.mean(0), dim1=-2, dim2=-1).mean().item()

                # Attention entropy
                entropy = self._compute_attention_entropy(head_attn.unsqueeze(1))

                head_analysis[f'head_{head_idx}'] = {
                    'average_attention_distance': avg_distance,
                    'self_attention_strength': self_attention,
                    'attention_entropy': entropy,
                    'dominant_pattern': self._classify_attention_pattern(head_attn.mean(0))
                }

            analysis['head_analysis'] = head_analysis

            # Cross-head comparison
            if n_heads > 1:
                # Compute head similarity
                head_similarities = {}
                for i in range(n_heads):
                    for j in range(i + 1, n_heads):
                        head_i = attention_weights[:, i, :, :].flatten()
                        head_j = attention_weights[:, j, :, :].flatten()
                        similarity = F.cosine_similarity(head_i, head_j, dim=0).item()
                        head_similarities[f'head_{i}_vs_head_{j}'] = similarity

                analysis['head_diversity'] = {
                    'pairwise_similarities': head_similarities,
                    'mean_similarity': np.mean(list(head_similarities.values())),
                    'interpretation': "Lower values indicate more diverse attention patterns"
                }

            # Layer-level patterns
            layer_attention = attention_weights.mean(dim=1)  # Average across heads

            analysis['layer_patterns'] = {
                'attention_concentration': self._measure_attention_concentration(layer_attention),
                'causal_compliance': self._measure_causal_compliance(layer_attention) if self.is_causal else None,
                'position_bias': self._measure_position_bias(layer_attention)
            }

            # Token-specific analysis (if labels provided)
            if token_labels is not None and len(token_labels) == seq_len:
                analysis['token_analysis'] = self._analyze_token_attention(
                    attention_weights, token_labels
                )

            analysis['debug_info'] = debug_info

            return analysis

    def _classify_attention_pattern(self, attention_matrix: torch.Tensor) -> str:
        """Classify the dominant pattern in an attention matrix."""
        seq_len = attention_matrix.shape[0]

        # Check for diagonal pattern (self-attention)
        diagonal_strength = torch.diagonal(attention_matrix).mean().item()

        # Check for local pattern (attending to nearby positions)
        local_mask = torch.abs(torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)) <= 2
        local_strength = (attention_matrix * local_mask.float()).sum().item() / local_mask.sum().item()

        # Check for global pattern (uniform attention)
        uniform_baseline = 1.0 / seq_len
        attention_variance = attention_matrix.var().item()

        if diagonal_strength > 0.5:
            return "self-attention"
        elif local_strength > 0.7:
            return "local"
        elif attention_variance < uniform_baseline * 0.1:
            return "global/uniform"
        else:
            return "mixed"

    def _measure_attention_concentration(self, attention_weights: torch.Tensor) -> Dict[str, float]:
        """Measure how concentrated attention is."""
        # Gini coefficient of attention distribution
        batch_size, seq_len, _ = attention_weights.shape
        concentration_metrics = {}

        for batch_idx in range(batch_size):
            attn = attention_weights[batch_idx]
            # Flatten and sort attention weights
            sorted_attn, _ = torch.sort(attn.flatten())
            n = len(sorted_attn)
            index = torch.arange(1, n + 1, dtype=torch.float, device=sorted_attn.device)
            gini = (2 * torch.sum(index * sorted_attn)) / (n * torch.sum(sorted_attn)) - (n + 1) / n
            concentration_metrics[f'batch_{batch_idx}_gini'] = gini.item()

        return {
            'mean_gini_coefficient': np.mean(list(concentration_metrics.values())),
            'interpretation': "Higher values indicate more concentrated attention"
        }

    def _measure_causal_compliance(self, attention_weights: torch.Tensor) -> Dict[str, float]:
        """Measure how well attention respects causal constraints."""
        batch_size, seq_len, _ = attention_weights.shape

        # Create upper triangular mask (future positions)
        future_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

        # Sum attention to future positions (should be ~0 for causal models)
        future_attention = (attention_weights * future_mask.unsqueeze(0)).sum(dim=(-2, -1))
        compliance = 1.0 - future_attention.mean().item()  # 1.0 = perfect compliance

        return {
            'causal_compliance_score': compliance,
            'mean_future_attention': future_attention.mean().item(),
            'interpretation': "Higher compliance means better causal behavior"
        }

    def _measure_position_bias(self, attention_weights: torch.Tensor) -> Dict[str, Any]:
        """Measure positional biases in attention."""
        seq_len = attention_weights.shape[-1]

        # Average attention received by each position
        position_attention = attention_weights.mean(dim=(0, 1))  # [seq_len]

        # Find positions with highest/lowest attention
        max_pos = torch.argmax(position_attention).item()
        min_pos = torch.argmin(position_attention).item()

        return {
            'position_attention_distribution': position_attention.tolist(),
            'most_attended_position': max_pos,
            'least_attended_position': min_pos,
            'attention_variance_across_positions': position_attention.var().item(),
            'interpretation': "How uniformly attention is distributed across positions"
        }

    def _analyze_token_attention(
        self,
        attention_weights: torch.Tensor,
        token_labels: list
    ) -> Dict[str, Any]:
        """Analyze attention patterns for specific tokens."""
        batch_size, n_heads, seq_len, _ = attention_weights.shape

        # Average across batch and heads for simplicity
        avg_attention = attention_weights.mean(dim=(0, 1))  # [seq_len, seq_len]

        token_analysis = {}
        for i, token in enumerate(token_labels):
            # Attention given by this token
            attention_given = avg_attention[i, :]
            # Attention received by this token
            attention_received = avg_attention[:, i]

            most_attended_idx = torch.argmax(attention_given).item()
            most_attended_token = token_labels[most_attended_idx] if most_attended_idx < len(token_labels) else "UNK"

            token_analysis[f'token_{i}_{token}'] = {
                'attention_given': attention_given.tolist(),
                'attention_received': attention_received.tolist(),
                'most_attended_token': most_attended_token,
                'self_attention': attention_given[i].item(),
                'total_attention_received': attention_received.sum().item()
            }

        return token_analysis


def create_educational_attention_examples() -> Dict[str, Any]:
    """
    Create educational examples for understanding multi-head attention.

    Returns:
        Dictionary with examples and explanations
    """
    examples = {}

    print("Creating multi-head attention examples...")

    # Example 1: Small-scale attention
    config = TransformerConfig.small_config()
    attention_layer = MultiHeadAttention(
        d_model=config.d_model,
        n_heads=config.n_heads,
        dropout=0.0,  # No dropout for reproducible examples
        is_causal=True
    )

    # Create simple input
    batch_size, seq_len = 1, 5
    hidden_states = torch.randn(batch_size, seq_len, config.d_model) * 0.1

    # Forward pass with debug info
    output, attention_weights, debug_info = attention_layer(
        hidden_states, return_attention_weights=True, return_debug_info=True
    )

    examples['basic_attention'] = {
        'input_shape': hidden_states.shape,
        'output_shape': output.shape,
        'attention_weights_shape': attention_weights.shape,
        'debug_info': debug_info,
        'explanation': "Basic multi-head attention forward pass"
    }

    # Example 2: Attention pattern analysis
    analysis = attention_layer.analyze_attention_patterns(hidden_states)

    examples['pattern_analysis'] = {
        'analysis': analysis,
        'explanation': "Detailed analysis of attention patterns and behaviors"
    }

    # Example 3: Causal vs non-causal comparison
    non_causal_attention = MultiHeadAttention(
        d_model=config.d_model,
        n_heads=config.n_heads,
        dropout=0.0,
        is_causal=False
    )

    output_causal, attn_causal = attention_layer(hidden_states, return_attention_weights=True)
    output_non_causal, attn_non_causal = non_causal_attention(hidden_states, return_attention_weights=True)

    examples['causal_comparison'] = {
        'causal_attention': attn_causal,
        'non_causal_attention': attn_non_causal,
        'output_difference_norm': torch.norm(output_causal - output_non_causal).item(),
        'explanation': "Comparison between causal and non-causal attention patterns"
    }

    return examples


if __name__ == "__main__":
    print("Multi-Head Attention Mechanism")
    print("=" * 50)

    # Run educational examples
    examples = create_educational_attention_examples()

    print("\n1. Basic attention example:")
    basic = examples['basic_attention']
    print(f"Input shape: {basic['input_shape']}")
    print(f"Output shape: {basic['output_shape']}")
    print(f"Attention weights shape: {basic['attention_weights_shape']}")
    print(f"Configuration: {basic['debug_info']['configuration']}")

    print(f"\nAttention statistics:")
    stats = basic['debug_info']['statistics']
    print(f"  Attention entropy: {stats['attention_weights']['entropy']:.4f}")
    print(f"  Attention mean: {stats['attention_weights']['mean']:.4f}")
    print(f"  Output norm: {stats['output']['norm']:.4f}")

    print("\n2. Pattern analysis:")
    pattern = examples['pattern_analysis']['analysis']

    print("Head analysis:")
    for head, info in pattern['head_analysis'].items():
        print(f"  {head}: pattern={info['dominant_pattern']}, "
              f"distance={info['average_attention_distance']:.2f}, "
              f"self_attn={info['self_attention_strength']:.3f}")

    if 'head_diversity' in pattern:
        print(f"Head diversity (mean similarity): {pattern['head_diversity']['mean_similarity']:.4f}")

    print(f"Layer attention concentration (Gini): {pattern['layer_patterns']['attention_concentration']['mean_gini_coefficient']:.4f}")

    if pattern['layer_patterns']['causal_compliance']:
        print(f"Causal compliance: {pattern['layer_patterns']['causal_compliance']['causal_compliance_score']:.4f}")

    print("\n3. Causal vs non-causal comparison:")
    comparison = examples['causal_comparison']
    print(f"Output difference norm: {comparison['output_difference_norm']:.6f}")

    # Show attention weight differences
    causal_attn = comparison['causal_attention'][0, 0]  # First batch, first head
    non_causal_attn = comparison['non_causal_attention'][0, 0]

    print("Causal attention pattern (first head):")
    for i in range(causal_attn.shape[0]):
        row_str = " ".join([f"{val:.3f}" for val in causal_attn[i].tolist()])
        print(f"  [{row_str}]")

    print("Non-causal attention pattern (first head):")
    for i in range(non_causal_attn.shape[0]):
        row_str = " ".join([f"{val:.3f}" for val in non_causal_attn[i].tolist()])
        print(f"  [{row_str}]")

    print("\n4. Configuration validation:")
    test_configs = [
        (256, 8),   # d_model=256, n_heads=8
        (512, 16),  # d_model=512, n_heads=16
        (768, 12),  # d_model=768, n_heads=12 (GPT-2 small)
    ]

    for d_model, n_heads in test_configs:
        try:
            test_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
            print(f"✓ d_model={d_model}, n_heads={n_heads}, d_head={test_attn.d_head}")
        except ValueError as e:
            print(f"✗ d_model={d_model}, n_heads={n_heads}: {e}")

    print("\n✅ Multi-head attention mechanism implemented!")