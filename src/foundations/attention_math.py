"""
Attention Mechanism Mathematics

Educational implementation of attention mathematics for transformers.
This module breaks down the attention mechanism into its fundamental components
with clear mathematical explanations.

Key concepts covered:
- Scaled dot-product attention
- Multi-head attention mathematics
- Query, Key, Value transformations
- Attention score computation and interpretation
- Causal masking for autoregressive models

Following ARENA Chapter 1.1 curriculum.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import math
import matplotlib.pyplot as plt
import numpy as np


class AttentionMathematics:
    """
    Educational class for understanding attention mathematics.

    This class implements attention step-by-step with extensive documentation
    and visualization capabilities for educational purposes.
    """

    @staticmethod
    def compute_qkv_projections(
        hidden_states: torch.Tensor,
        w_q: torch.Tensor,
        w_k: torch.Tensor,
        w_v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Query, Key, Value projections.

        Mathematical foundation:
        - Q = X @ W_Q  (Query: what am I looking for?)
        - K = X @ W_K  (Key: what do I contain?)
        - V = X @ W_V  (Value: what information do I have?)

        Args:
            hidden_states: Input tensor [batch, seq_len, d_model]
            w_q: Query weight matrix [d_model, d_model]
            w_k: Key weight matrix [d_model, d_model]
            w_v: Value weight matrix [d_model, d_model]

        Returns:
            query, key, value tensors each [batch, seq_len, d_model]
        """
        # Linear projections
        query = torch.matmul(hidden_states, w_q)
        key = torch.matmul(hidden_states, w_k)
        value = torch.matmul(hidden_states, w_v)

        return query, key, value

    @staticmethod
    def single_head_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Compute single-head attention with detailed breakdown.

        Mathematical steps:
        1. Attention scores: S = Q @ K^T
        2. Scaling: S = S / √d_k
        3. Masking: S[mask] = -∞
        4. Probabilities: A = softmax(S)
        5. Output: O = A @ V

        Args:
            query: Query tensor [batch, seq_len, d_head]
            key: Key tensor [batch, seq_len, d_head]
            value: Value tensor [batch, seq_len, d_head]
            mask: Optional mask [batch, seq_len, seq_len]
            dropout_p: Dropout probability
            temperature: Temperature for attention (default 1.0)

        Returns:
            output: Attention output [batch, seq_len, d_head]
            attention_weights: Attention probabilities [batch, seq_len, seq_len]
            debug_info: Dictionary with intermediate computations
        """
        batch_size, seq_len, d_head = query.shape
        debug_info = {}

        # Step 1: Compute raw attention scores Q @ K^T
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        debug_info['raw_scores'] = {
            'shape': attention_scores.shape,
            'mean': attention_scores.mean().item(),
            'std': attention_scores.std().item(),
            'explanation': "Raw dot products between queries and keys"
        }

        # Step 2: Scale by √d_k for numerical stability
        scale_factor = math.sqrt(d_head) * temperature
        scaled_scores = attention_scores / scale_factor
        debug_info['scaled_scores'] = {
            'scale_factor': scale_factor,
            'mean': scaled_scores.mean().item(),
            'std': scaled_scores.std().item(),
            'explanation': f"Scores scaled by √d_k = √{d_head} = {math.sqrt(d_head):.2f}"
        }

        # Step 3: Apply mask (if provided)
        if mask is not None:
            # Set masked positions to large negative value
            masked_scores = scaled_scores.masked_fill(mask == 0, -1e9)
            debug_info['masking'] = {
                'mask_shape': mask.shape,
                'num_masked': (mask == 0).sum().item(),
                'explanation': "Masked positions set to -∞ for softmax"
            }
        else:
            masked_scores = scaled_scores
            debug_info['masking'] = {'explanation': "No mask applied"}

        # Step 4: Apply softmax to get attention probabilities
        attention_weights = F.softmax(masked_scores, dim=-1)
        debug_info['attention_weights'] = {
            'shape': attention_weights.shape,
            'sum_per_row': attention_weights.sum(dim=-1).mean().item(),
            'max_attention': attention_weights.max().item(),
            'min_attention': attention_weights.min().item(),
            'explanation': "Softmax probabilities (each row sums to 1)"
        }

        # Step 5: Apply dropout (if training)
        if dropout_p > 0.0:
            attention_weights = F.dropout(attention_weights, p=dropout_p, training=True)
            debug_info['dropout'] = {'p': dropout_p, 'applied': True}
        else:
            debug_info['dropout'] = {'applied': False}

        # Step 6: Compute weighted sum of values
        output = torch.matmul(attention_weights, value)
        debug_info['output'] = {
            'shape': output.shape,
            'mean': output.mean().item(),
            'std': output.std().item(),
            'explanation': "Weighted sum of values using attention probabilities"
        }

        return output, attention_weights, debug_info

    @staticmethod
    def multi_head_attention_math(
        hidden_states: torch.Tensor,
        w_q: torch.Tensor,
        w_k: torch.Tensor,
        w_v: torch.Tensor,
        w_o: torch.Tensor,
        n_heads: int,
        mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Complete multi-head attention computation with mathematical breakdown.

        Multi-head attention allows the model to attend to different types of
        relationships simultaneously. Each head learns different attention patterns.

        Mathematical overview:
        1. Project input to Q, K, V
        2. Split into multiple heads
        3. Compute attention for each head independently
        4. Concatenate head outputs
        5. Apply final linear projection

        Args:
            hidden_states: Input [batch, seq_len, d_model]
            w_q, w_k, w_v: Weight matrices [d_model, d_model]
            w_o: Output projection [d_model, d_model]
            n_heads: Number of attention heads
            mask: Optional attention mask
            dropout_p: Dropout probability

        Returns:
            output: Final output [batch, seq_len, d_model]
            attention_weights: All head attentions [batch, n_heads, seq_len, seq_len]
            debug_info: Detailed computation breakdown
        """
        batch_size, seq_len, d_model = hidden_states.shape
        d_head = d_model // n_heads
        debug_info = {'n_heads': n_heads, 'd_head': d_head}

        # Step 1: Compute Q, K, V projections
        query, key, value = AttentionMathematics.compute_qkv_projections(
            hidden_states, w_q, w_k, w_v
        )
        debug_info['qkv_projections'] = {
            'query_shape': query.shape,
            'key_shape': key.shape,
            'value_shape': value.shape
        }

        # Step 2: Reshape for multi-head computation
        # [batch, seq_len, d_model] -> [batch, n_heads, seq_len, d_head]
        query = query.view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
        key = key.view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
        value = value.view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)

        debug_info['multi_head_reshape'] = {
            'query_multihead_shape': query.shape,
            'explanation': f"Split d_model={d_model} into {n_heads} heads of d_head={d_head}"
        }

        # Step 3: Compute attention for all heads simultaneously
        # Using batch operations for efficiency
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(d_head)

        if mask is not None:
            # Expand mask for all heads
            if mask.dim() == 3:  # [batch, seq_len, seq_len]
                mask = mask.unsqueeze(1)  # [batch, 1, seq_len, seq_len]
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(attention_scores, dim=-1)

        if dropout_p > 0.0:
            attention_weights = F.dropout(attention_weights, p=dropout_p, training=True)

        # Step 4: Apply attention to values
        head_outputs = torch.matmul(attention_weights, value)
        debug_info['head_outputs'] = {
            'shape': head_outputs.shape,
            'explanation': "Individual head outputs before concatenation"
        }

        # Step 5: Concatenate heads
        # [batch, n_heads, seq_len, d_head] -> [batch, seq_len, d_model]
        head_outputs = head_outputs.transpose(1, 2).contiguous()
        concatenated = head_outputs.view(batch_size, seq_len, d_model)
        debug_info['concatenation'] = {
            'concatenated_shape': concatenated.shape,
            'explanation': "Concatenated all head outputs"
        }

        # Step 6: Apply output projection
        output = torch.matmul(concatenated, w_o)
        debug_info['final_projection'] = {
            'output_shape': output.shape,
            'explanation': "Final linear projection"
        }

        return output, attention_weights, debug_info


def analyze_attention_patterns(
    attention_weights: torch.Tensor,
    token_labels: Optional[list] = None
) -> Dict[str, Any]:
    """
    Analyze attention patterns for educational insights.

    Args:
        attention_weights: Attention weights [batch, n_heads, seq_len, seq_len]
        token_labels: Optional list of token strings for visualization

    Returns:
        Dictionary with analysis results
    """
    batch_size, n_heads, seq_len, _ = attention_weights.shape
    analysis = {}

    # Overall statistics
    analysis['statistics'] = {
        'mean_attention': attention_weights.mean().item(),
        'std_attention': attention_weights.std().item(),
        'max_attention': attention_weights.max().item(),
        'min_attention': attention_weights.min().item(),
        'shape': attention_weights.shape
    }

    # Attention entropy (diversity of attention)
    # Higher entropy = more distributed attention
    # Lower entropy = more focused attention
    log_attention = torch.log(attention_weights + 1e-12)  # Add small epsilon for numerical stability
    entropy = -(attention_weights * log_attention).sum(dim=-1)
    analysis['entropy'] = {
        'mean_entropy': entropy.mean().item(),
        'std_entropy': entropy.std().item(),
        'interpretation': "Higher values indicate more distributed attention"
    }

    # Attention to diagonal (self-attention strength)
    diagonal_attention = torch.diagonal(attention_weights, dim1=-2, dim2=-1)
    analysis['self_attention'] = {
        'mean_diagonal': diagonal_attention.mean().item(),
        'std_diagonal': diagonal_attention.std().item(),
        'interpretation': "How much each position attends to itself"
    }

    # Head diversity (how different are the attention patterns across heads)
    if n_heads > 1:
        # Compute pairwise correlation between heads
        attention_flat = attention_weights.view(batch_size, n_heads, -1)
        correlations = []
        for i in range(n_heads):
            for j in range(i + 1, n_heads):
                corr = F.cosine_similarity(
                    attention_flat[:, i, :], attention_flat[:, j, :], dim=-1
                ).mean().item()
                correlations.append(corr)

        analysis['head_diversity'] = {
            'mean_correlation': np.mean(correlations),
            'std_correlation': np.std(correlations),
            'interpretation': "Lower correlation indicates more diverse attention patterns"
        }

    # Attention distance (how far does each position look)
    positions = torch.arange(seq_len, dtype=torch.float, device=attention_weights.device)
    position_diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # [seq_len, seq_len]
    attention_distance = (attention_weights.unsqueeze(-1) * position_diff.abs().unsqueeze(0).unsqueeze(0)).sum(dim=-2)
    analysis['attention_distance'] = {
        'mean_distance': attention_distance.mean().item(),
        'std_distance': attention_distance.std().item(),
        'interpretation': "Average distance of attention (0 = self, higher = longer range)"
    }

    return analysis


def educational_attention_examples() -> Dict[str, Any]:
    """
    Create educational examples of attention computations.

    Returns:
        Dictionary with various attention examples and explanations
    """
    examples = {}

    # Example 1: Simple 3-token attention
    print("Creating simple 3-token attention example...")
    seq_len, d_model = 3, 4
    query = torch.tensor([
        [[1.0, 0.0, 0.0, 0.0]],  # Token 0: looking for "first word"
        [[0.0, 1.0, 0.0, 0.0]],  # Token 1: looking for "second word"
        [[0.0, 0.0, 1.0, 0.0]]   # Token 2: looking for "third word"
    ]).transpose(0, 1)  # [1, 3, 4]

    key = torch.tensor([
        [[1.0, 0.0, 0.0, 0.0]],    # Token 0: contains "first word info"
        [[0.0, 1.0, 0.0, 0.0]],    # Token 1: contains "second word info"
        [[0.5, 0.5, 1.0, 0.0]]     # Token 2: contains mixed info
    ]).transpose(0, 1)  # [1, 3, 4]

    value = torch.tensor([
        [[1.0, 2.0, 3.0, 4.0]],    # Token 0: valuable info A
        [[5.0, 6.0, 7.0, 8.0]],    # Token 1: valuable info B
        [[9.0, 10.0, 11.0, 12.0]]  # Token 2: valuable info C
    ]).transpose(0, 1)  # [1, 3, 4]

    output, attention_weights, debug_info = AttentionMathematics.single_head_attention(
        query, key, value
    )

    examples['simple_3token'] = {
        'query': query,
        'key': key,
        'value': value,
        'attention_weights': attention_weights,
        'output': output,
        'debug_info': debug_info,
        'explanation': "Simple 3-token example with interpretable Q, K, V"
    }

    # Example 2: Causal attention with masking
    print("Creating causal attention example...")
    batch_size, seq_len, d_head = 1, 4, 8
    query = torch.randn(batch_size, seq_len, d_head)
    key = torch.randn(batch_size, seq_len, d_head)
    value = torch.randn(batch_size, seq_len, d_head)

    # Create causal mask
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    causal_mask = causal_mask.unsqueeze(0)  # Add batch dimension

    output_causal, attention_causal, debug_causal = AttentionMathematics.single_head_attention(
        query, key, value, mask=causal_mask
    )

    examples['causal_attention'] = {
        'causal_mask': causal_mask,
        'attention_weights': attention_causal,
        'output': output_causal,
        'debug_info': debug_causal,
        'explanation': "Causal (autoregressive) attention with lower triangular mask"
    }

    return examples


if __name__ == "__main__":
    print("Attention Mechanism Mathematics")
    print("=" * 50)

    # Run educational examples
    examples = educational_attention_examples()

    print("\n1. Simple 3-token attention:")
    simple_example = examples['simple_3token']
    print("Attention weights:")
    print(simple_example['attention_weights'].squeeze(0).round(decimals=3))
    print("Explanation:", simple_example['explanation'])

    print("\n2. Causal attention:")
    causal_example = examples['causal_attention']
    print("Causal mask:")
    print(causal_example['causal_mask'].squeeze(0))
    print("Attention weights with causal mask:")
    print(causal_example['attention_weights'].squeeze(0).round(decimals=3))

    print("\n3. Multi-head attention test:")
    batch_size, seq_len, d_model = 2, 8, 512
    n_heads = 8

    # Create random inputs and weights
    hidden_states = torch.randn(batch_size, seq_len, d_model)
    w_q = torch.randn(d_model, d_model) * 0.02
    w_k = torch.randn(d_model, d_model) * 0.02
    w_v = torch.randn(d_model, d_model) * 0.02
    w_o = torch.randn(d_model, d_model) * 0.02

    output, attention_weights, debug_info = AttentionMathematics.multi_head_attention_math(
        hidden_states, w_q, w_k, w_v, w_o, n_heads
    )

    print(f"Input shape: {hidden_states.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Number of heads: {debug_info['n_heads']}")
    print(f"Head dimension: {debug_info['d_head']}")

    print("\n4. Attention pattern analysis:")
    analysis = analyze_attention_patterns(attention_weights)
    print(f"Mean attention: {analysis['statistics']['mean_attention']:.4f}")
    print(f"Attention entropy: {analysis['entropy']['mean_entropy']:.4f}")
    print(f"Self-attention strength: {analysis['self_attention']['mean_diagonal']:.4f}")
    if 'head_diversity' in analysis:
        print(f"Head correlation: {analysis['head_diversity']['mean_correlation']:.4f}")
    print(f"Average attention distance: {analysis['attention_distance']['mean_distance']:.4f}")

    print("\n✅ All attention mathematics implemented!")