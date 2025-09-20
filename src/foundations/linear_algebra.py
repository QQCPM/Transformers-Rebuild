"""
Linear Algebra Foundations for Transformers

Educational implementation of core linear algebra operations used in transformers.
This module focuses on clarity and understanding rather than performance optimization.

Key concepts covered:
- Matrix multiplication patterns in transformers
- Tensor reshaping for multi-head attention
- Broadcasting semantics
- Einsum operations for clarity

Following ARENA Chapter 1.1 curriculum.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import math


def batched_matrix_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Educational implementation of batched matrix multiplication.

    This is fundamental to attention mechanisms where we compute attention
    scores across multiple sequences and heads simultaneously.

    Args:
        a: Tensor of shape (..., n, k)
        b: Tensor of shape (..., k, m)

    Returns:
        Tensor of shape (..., n, m)

    Example:
        >>> # Batch of 2 sequences, each with 3x4 and 4x5 matrices
        >>> a = torch.randn(2, 3, 4)
        >>> b = torch.randn(2, 4, 5)
        >>> result = batched_matrix_multiply(a, b)
        >>> result.shape
        torch.Size([2, 3, 5])
    """
    # torch.matmul automatically handles batch dimensions
    return torch.matmul(a, b)


def scaled_dot_product_math(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Educational implementation of scaled dot-product attention mathematics.

    This breaks down the attention formula: Attention(Q,K,V) = softmax(QK^T/√d_k)V

    Args:
        query: Query tensor [batch, n_heads, seq_len, d_head]
        key: Key tensor [batch, n_heads, seq_len, d_head]
        value: Value tensor [batch, n_heads, seq_len, d_head]
        mask: Optional attention mask [batch, n_heads, seq_len, seq_len]
        dropout_p: Dropout probability for attention weights

    Returns:
        output: Attention output [batch, n_heads, seq_len, d_head]
        attention_weights: Attention probabilities [batch, n_heads, seq_len, seq_len]

    Mathematical breakdown:
        1. Compute attention scores: QK^T
        2. Scale by √d_k for stability
        3. Apply mask (for causal attention)
        4. Apply softmax to get probabilities
        5. Apply dropout for regularization
        6. Apply attention to values: attention_weights @ V
    """
    batch_size, n_heads, seq_len, d_head = query.shape

    # Step 1: Compute raw attention scores QK^T
    # query: [batch, n_heads, seq_len, d_head]
    # key.transpose(-2, -1): [batch, n_heads, d_head, seq_len]
    # attention_scores: [batch, n_heads, seq_len, seq_len]
    attention_scores = torch.matmul(query, key.transpose(-2, -1))

    # Step 2: Scale by √d_k for numerical stability
    # Without scaling, large d_head values can push softmax into saturation
    scale_factor = math.sqrt(d_head)
    attention_scores = attention_scores / scale_factor

    # Step 3: Apply attention mask (for causal attention or padding)
    if mask is not None:
        # Apply mask by setting masked positions to large negative value
        # This ensures softmax gives them probability ≈ 0
        attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

    # Step 4: Apply softmax to get attention probabilities
    # softmax along the last dimension (keys dimension)
    attention_weights = F.softmax(attention_scores, dim=-1)

    # Step 5: Apply dropout to attention weights (if training)
    if dropout_p > 0.0:
        attention_weights = F.dropout(attention_weights, p=dropout_p, training=True)

    # Step 6: Apply attention to values
    # attention_weights: [batch, n_heads, seq_len, seq_len]
    # value: [batch, n_heads, seq_len, d_head]
    # output: [batch, n_heads, seq_len, d_head]
    output = torch.matmul(attention_weights, value)

    return output, attention_weights


def tensor_reshaping_for_multihead(
    tensor: torch.Tensor,
    n_heads: int,
    d_head: int
) -> torch.Tensor:
    """
    Reshape tensor for multi-head attention computation.

    This is a key operation in multi-head attention where we need to
    split the hidden dimension into multiple heads.

    Args:
        tensor: Input tensor [batch, seq_len, d_model]
        n_heads: Number of attention heads
        d_head: Dimension per head (d_model // n_heads)

    Returns:
        Reshaped tensor [batch, n_heads, seq_len, d_head]

    Mathematical insight:
        We're essentially viewing the d_model dimension as n_heads × d_head
        and permuting dimensions to group by head.
    """
    batch_size, seq_len, d_model = tensor.shape

    # Verify dimensions are compatible
    assert d_model == n_heads * d_head, f"d_model ({d_model}) must equal n_heads ({n_heads}) * d_head ({d_head})"

    # Reshape: [batch, seq_len, d_model] -> [batch, seq_len, n_heads, d_head]
    tensor = tensor.view(batch_size, seq_len, n_heads, d_head)

    # Transpose: [batch, seq_len, n_heads, d_head] -> [batch, n_heads, seq_len, d_head]
    tensor = tensor.transpose(1, 2)

    return tensor


def tensor_reshaping_from_multihead(
    tensor: torch.Tensor
) -> torch.Tensor:
    """
    Reshape tensor back from multi-head format to single tensor.

    This reverses the reshaping done for multi-head attention.

    Args:
        tensor: Input tensor [batch, n_heads, seq_len, d_head]

    Returns:
        Reshaped tensor [batch, seq_len, d_model]
    """
    batch_size, n_heads, seq_len, d_head = tensor.shape

    # Transpose: [batch, n_heads, seq_len, d_head] -> [batch, seq_len, n_heads, d_head]
    tensor = tensor.transpose(1, 2)

    # Reshape: [batch, seq_len, n_heads, d_head] -> [batch, seq_len, d_model]
    d_model = n_heads * d_head
    tensor = tensor.contiguous().view(batch_size, seq_len, d_model)

    return tensor


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create a causal (lower triangular) attention mask.

    In autoregressive language models, each position can only attend to
    positions that come before it (and itself).

    Args:
        seq_len: Sequence length
        device: Device to create tensor on

    Returns:
        Causal mask [seq_len, seq_len] where 1 = can attend, 0 = masked

    Example:
        >>> mask = create_causal_mask(4, torch.device('cpu'))
        >>> print(mask)
        tensor([[1, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 1]])
    """
    # Create lower triangular matrix
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask


def broadcasting_examples() -> dict:
    """
    Educational examples of broadcasting in transformer operations.

    Broadcasting is crucial for efficient computation in transformers,
    especially when applying operations across batch and sequence dimensions.

    Returns:
        Dictionary with examples and explanations
    """
    examples = {}

    # Example 1: Adding bias to attention scores
    batch_size, n_heads, seq_len = 2, 8, 10
    attention_scores = torch.randn(batch_size, n_heads, seq_len, seq_len)
    bias = torch.randn(1, 1, seq_len, seq_len)  # Broadcastable bias

    # Broadcasting: bias is applied to all batches and heads
    biased_scores = attention_scores + bias
    examples['bias_broadcasting'] = {
        'attention_scores_shape': attention_scores.shape,
        'bias_shape': bias.shape,
        'result_shape': biased_scores.shape,
        'explanation': "Bias broadcasts across batch and head dimensions"
    }

    # Example 2: Position-wise scaling
    hidden_states = torch.randn(batch_size, seq_len, 512)  # [batch, seq, d_model]
    position_scale = torch.randn(1, seq_len, 1)  # Scale per position

    scaled_hidden = hidden_states * position_scale
    examples['position_scaling'] = {
        'hidden_states_shape': hidden_states.shape,
        'position_scale_shape': position_scale.shape,
        'result_shape': scaled_hidden.shape,
        'explanation': "Position scale broadcasts across batch and d_model dimensions"
    }

    # Example 3: Head-wise normalization
    multihead_output = torch.randn(batch_size, n_heads, seq_len, 64)
    head_norm = torch.randn(1, n_heads, 1, 1)  # Normalization per head

    normalized_output = multihead_output / head_norm
    examples['head_normalization'] = {
        'multihead_output_shape': multihead_output.shape,
        'head_norm_shape': head_norm.shape,
        'result_shape': normalized_output.shape,
        'explanation': "Head normalization broadcasts across batch, sequence, and head dimensions"
    }

    return examples


def einsum_attention_patterns() -> dict:
    """
    Educational examples using einsum for attention computations.

    Einsum provides a clear, mathematical notation for complex tensor operations
    and is often more readable than multiple transpose/reshape operations.

    Returns:
        Dictionary with einsum examples and explanations
    """
    batch_size, n_heads, seq_len, d_head = 2, 8, 10, 64

    query = torch.randn(batch_size, n_heads, seq_len, d_head)
    key = torch.randn(batch_size, n_heads, seq_len, d_head)
    value = torch.randn(batch_size, n_heads, seq_len, d_head)

    examples = {}

    # Example 1: Attention scores using einsum
    # Traditional: torch.matmul(query, key.transpose(-2, -1))
    # Einsum notation: 'bhqd,bhkd->bhqk'
    attention_scores_einsum = torch.einsum('bhqd,bhkd->bhqk', query, key)
    attention_scores_matmul = torch.matmul(query, key.transpose(-2, -1))

    examples['attention_scores'] = {
        'einsum_result_shape': attention_scores_einsum.shape,
        'matmul_result_shape': attention_scores_matmul.shape,
        'are_equal': torch.allclose(attention_scores_einsum, attention_scores_matmul),
        'einsum_notation': 'bhqd,bhkd->bhqk',
        'explanation': "b=batch, h=heads, q=query_seq, k=key_seq, d=head_dim"
    }

    # Example 2: Applying attention to values
    attention_weights = F.softmax(attention_scores_einsum / math.sqrt(d_head), dim=-1)
    # Traditional: torch.matmul(attention_weights, value)
    # Einsum: 'bhqk,bhkd->bhqd'
    output_einsum = torch.einsum('bhqk,bhkd->bhqd', attention_weights, value)
    output_matmul = torch.matmul(attention_weights, value)

    examples['attention_output'] = {
        'einsum_result_shape': output_einsum.shape,
        'matmul_result_shape': output_matmul.shape,
        'are_equal': torch.allclose(output_einsum, output_matmul),
        'einsum_notation': 'bhqk,bhkd->bhqd',
        'explanation': "Apply attention weights to values"
    }

    return examples


def educational_linear_layer_breakdown(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None
) -> dict:
    """
    Educational breakdown of linear layer computation.

    Shows step-by-step how nn.Linear works internally, which is fundamental
    to understanding transformer components.

    Args:
        input_tensor: Input [batch, seq_len, in_features]
        weight: Weight matrix [out_features, in_features]
        bias: Optional bias [out_features]

    Returns:
        Dictionary with step-by-step computation details
    """
    batch_size, seq_len, in_features = input_tensor.shape
    out_features, in_features_weight = weight.shape

    assert in_features == in_features_weight, "Input features must match weight dimensions"

    breakdown = {}

    # Step 1: Reshape input for matrix multiplication
    # [batch, seq_len, in_features] -> [batch * seq_len, in_features]
    input_reshaped = input_tensor.view(-1, in_features)
    breakdown['step1_reshape'] = {
        'original_shape': input_tensor.shape,
        'reshaped_shape': input_reshaped.shape,
        'explanation': "Flatten batch and sequence dimensions for matrix multiplication"
    }

    # Step 2: Matrix multiplication
    # [batch * seq_len, in_features] @ [in_features, out_features] -> [batch * seq_len, out_features]
    output_flat = torch.matmul(input_reshaped, weight.t())
    breakdown['step2_matmul'] = {
        'input_shape': input_reshaped.shape,
        'weight_transposed_shape': weight.t().shape,
        'output_shape': output_flat.shape,
        'explanation': "Matrix multiplication with transposed weight"
    }

    # Step 3: Add bias (if present)
    if bias is not None:
        output_flat = output_flat + bias
        breakdown['step3_bias'] = {
            'bias_shape': bias.shape,
            'explanation': "Add bias (broadcasts across batch*seq dimension)"
        }

    # Step 4: Reshape back to original batch structure
    # [batch * seq_len, out_features] -> [batch, seq_len, out_features]
    output = output_flat.view(batch_size, seq_len, out_features)
    breakdown['step4_reshape_back'] = {
        'flat_shape': output_flat.shape,
        'final_shape': output.shape,
        'explanation': "Reshape back to [batch, seq_len, out_features]"
    }

    breakdown['final_output'] = output

    return breakdown


if __name__ == "__main__":
    # Educational examples and tests
    print("Linear Algebra Foundations for Transformers")
    print("=" * 50)

    # Test basic operations
    batch_size, seq_len, d_model = 2, 10, 512
    n_heads, d_head = 8, 64

    print(f"\n1. Multi-head reshaping example:")
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"Original shape: {x.shape}")

    x_multihead = tensor_reshaping_for_multihead(x, n_heads, d_head)
    print(f"Multi-head shape: {x_multihead.shape}")

    x_back = tensor_reshaping_from_multihead(x_multihead)
    print(f"Reshaped back: {x_back.shape}")
    print(f"Shapes match original: {torch.allclose(x, x_back)}")

    print(f"\n2. Causal mask example:")
    mask = create_causal_mask(5, torch.device('cpu'))
    print("Causal mask (5x5):")
    print(mask)

    print(f"\n3. Broadcasting examples:")
    broadcast_examples = broadcasting_examples()
    for name, example in broadcast_examples.items():
        print(f"{name}: {example['explanation']}")

    print(f"\n4. Einsum examples:")
    einsum_examples = einsum_attention_patterns()
    for name, example in einsum_examples.items():
        print(f"{name}: {example['explanation']}")
        print(f"  Notation: {example['einsum_notation']}")
        print(f"  Results match: {example['are_equal']}")

    print("\n✅ All linear algebra foundations implemented!")