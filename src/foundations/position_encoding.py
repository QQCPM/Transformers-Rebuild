"""
Position Encoding for Transformers

Educational implementation of positional encoding mechanisms used in transformers.
This module explains why transformers need positional information and how different
encoding schemes work.

Key concepts covered:
- Sinusoidal position encoding (original Transformer paper)
- Learned position embeddings
- Relative position encoding
- Why position encoding is necessary for transformers

Following ARENA Chapter 1.1 curriculum.
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt


class SinusoidalPositionEncoding:
    """
    Sinusoidal position encoding from "Attention Is All You Need".

    This implementation creates fixed sinusoidal patterns that encode position
    information in a way that allows the model to learn relative positions.

    Mathematical formula:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Where:
    - pos: position in sequence
    - i: dimension index
    - d_model: model dimension
    """

    def __init__(self, d_model: int, max_seq_len: int = 5000):
        """
        Initialize sinusoidal position encoding.

        Args:
            d_model: Model dimension (must be even)
            max_seq_len: Maximum sequence length to precompute
        """
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")

        # Precompute position encodings
        self.pe = self._create_position_encoding()

    def _create_position_encoding(self) -> torch.Tensor:
        """
        Create the sinusoidal position encoding matrix.

        Returns:
            Position encoding matrix [max_seq_len, d_model]
        """
        # Create position indices [0, 1, 2, ..., max_seq_len-1]
        positions = torch.arange(self.max_seq_len, dtype=torch.float).unsqueeze(1)

        # Create dimension indices [0, 2, 4, ..., d_model-2]
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float) *
            -(math.log(10000.0) / self.d_model)
        )

        # Initialize position encoding matrix
        pe = torch.zeros(self.max_seq_len, self.d_model)

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(positions * div_term)

        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(positions * div_term)

        return pe

    def __call__(self, seq_len: int) -> torch.Tensor:
        """
        Get position encoding for a sequence.

        Args:
            seq_len: Length of sequence

        Returns:
            Position encoding [seq_len, d_model]
        """
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}")

        return self.pe[:seq_len]

    def visualize_encoding(self, seq_len: Optional[int] = None, dims_to_show: int = 8) -> Dict[str, Any]:
        """
        Create visualization of position encoding patterns.

        Args:
            seq_len: Sequence length to visualize (default: min(100, max_seq_len))
            dims_to_show: Number of dimensions to show in visualization

        Returns:
            Dictionary with visualization data and explanations
        """
        if seq_len is None:
            seq_len = min(100, self.max_seq_len)

        pe_subset = self.pe[:seq_len, :dims_to_show]

        visualization_data = {
            'position_encoding': pe_subset,
            'seq_len': seq_len,
            'dims_shown': dims_to_show,
            'explanation': {
                'sine_pattern': "Even dimensions use sine waves with different frequencies",
                'cosine_pattern': "Odd dimensions use cosine waves with different frequencies",
                'frequency_property': "Lower dimensions have lower frequencies (longer wavelengths)",
                'uniqueness': "Each position has a unique encoding vector",
                'relative_position': "The encoding allows learning relative position relationships"
            }
        }

        return visualization_data

    def demonstrate_properties(self, positions: list = [0, 1, 10, 50]) -> Dict[str, Any]:
        """
        Demonstrate key properties of sinusoidal position encoding.

        Args:
            positions: List of positions to analyze

        Returns:
            Dictionary with property demonstrations
        """
        properties = {}

        # Get encodings for specified positions
        encodings = {pos: self.pe[pos] for pos in positions}

        # Property 1: Orthogonality/similarity patterns
        similarities = {}
        for i, pos1 in enumerate(positions):
            for pos2 in positions[i:]:
                similarity = torch.cosine_similarity(
                    encodings[pos1].unsqueeze(0),
                    encodings[pos2].unsqueeze(0)
                ).item()
                similarities[f"pos_{pos1}_vs_pos_{pos2}"] = similarity

        properties['position_similarities'] = {
            'similarities': similarities,
            'explanation': "Cosine similarity between position encodings"
        }

        # Property 2: Linear combination property
        # PE(pos + k) should be expressible as linear combination of PE(pos) and PE(k)
        if len(positions) >= 3:
            pos_a, pos_b = positions[0], positions[1]
            pos_sum = pos_a + pos_b
            if pos_sum < self.max_seq_len:
                pe_a = encodings[pos_a]
                pe_b = encodings[pos_b]
                pe_sum_actual = self.pe[pos_sum]

                # This is an approximation - exact linear combination is complex
                linear_approximation = pe_a + pe_b
                approximation_error = torch.norm(pe_sum_actual - linear_approximation).item()

                properties['linear_combination'] = {
                    'pos_a': pos_a,
                    'pos_b': pos_b,
                    'pos_sum': pos_sum,
                    'approximation_error': approximation_error,
                    'explanation': "PE has linear properties (though not exact linear combination)"
                }

        # Property 3: Frequency analysis of different dimensions
        frequencies = []
        for dim in range(0, min(8, self.d_model), 2):  # Even dimensions only
            # Analyze first 100 positions
            pe_dim = self.pe[:100, dim]
            # Approximate frequency by counting zero crossings
            zero_crossings = torch.sum((pe_dim[:-1] * pe_dim[1:]) < 0).item()
            frequencies.append(zero_crossings / 100)

        properties['frequency_analysis'] = {
            'frequencies': frequencies,
            'explanation': "Approximate frequencies for even dimensions (zero crossings per 100 positions)"
        }

        return properties


class LearnedPositionEmbedding(nn.Module):
    """
    Learned position embeddings as an alternative to sinusoidal encoding.

    This approach learns position representations during training, which can
    potentially capture position-dependent patterns specific to the task.
    """

    def __init__(self, max_seq_len: int, d_model: int):
        """
        Initialize learned position embeddings.

        Args:
            max_seq_len: Maximum sequence length
            d_model: Model dimension
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        # Learnable position embedding lookup table
        self.position_embeddings = nn.Embedding(max_seq_len, d_model)

        # Initialize with small random values
        nn.init.normal_(self.position_embeddings.weight, std=0.02)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Get learned position embeddings.

        Args:
            seq_len: Length of sequence
            device: Device to create tensor on

        Returns:
            Position embeddings [seq_len, d_model]
        """
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}")

        positions = torch.arange(seq_len, device=device)
        return self.position_embeddings(positions)


class RelativePositionEncoding:
    """
    Relative position encoding that focuses on relative distances between positions.

    This approach encodes the relative distance between positions rather than
    absolute positions, which can be more suitable for certain tasks.
    """

    def __init__(self, d_model: int, max_relative_distance: int = 128):
        """
        Initialize relative position encoding.

        Args:
            d_model: Model dimension
            max_relative_distance: Maximum relative distance to encode
        """
        self.d_model = d_model
        self.max_relative_distance = max_relative_distance

        # Create relative position encoding matrix
        self.relative_pe = self._create_relative_encoding()

    def _create_relative_encoding(self) -> torch.Tensor:
        """
        Create relative position encoding matrix.

        Returns:
            Relative encoding matrix [2*max_relative_distance+1, d_model]
        """
        # Create relative distances: [-max_distance, ..., -1, 0, 1, ..., max_distance]
        relative_distances = torch.arange(
            -self.max_relative_distance,
            self.max_relative_distance + 1,
            dtype=torch.float
        ).unsqueeze(1)

        # Use similar sinusoidal pattern as absolute encoding
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float) *
            -(math.log(10000.0) / self.d_model)
        )

        pe = torch.zeros(2 * self.max_relative_distance + 1, self.d_model)
        pe[:, 0::2] = torch.sin(relative_distances * div_term)
        pe[:, 1::2] = torch.cos(relative_distances * div_term)

        return pe

    def get_relative_encoding(self, seq_len: int) -> torch.Tensor:
        """
        Get relative position encoding matrix for a sequence.

        Args:
            seq_len: Length of sequence

        Returns:
            Relative encoding matrix [seq_len, seq_len, d_model]
        """
        # Create relative position matrix
        positions = torch.arange(seq_len)
        relative_positions = positions.unsqueeze(1) - positions.unsqueeze(0)

        # Clip to maximum distance
        relative_positions = torch.clamp(
            relative_positions,
            -self.max_relative_distance,
            self.max_relative_distance
        )

        # Convert to indices (add max_distance to make non-negative)
        indices = relative_positions + self.max_relative_distance

        # Lookup encodings
        return self.relative_pe[indices]


def compare_position_encodings() -> Dict[str, Any]:
    """
    Educational comparison of different position encoding methods.

    Returns:
        Dictionary with comparisons and analysis
    """
    d_model = 64
    seq_len = 20
    max_seq_len = 100

    # Initialize different encoding methods
    sinusoidal = SinusoidalPositionEncoding(d_model, max_seq_len)
    learned = LearnedPositionEmbedding(max_seq_len, d_model)
    relative = RelativePositionEncoding(d_model, max_relative_distance=32)

    comparison = {}

    # Get encodings
    sine_encoding = sinusoidal(seq_len)
    learned_encoding = learned(seq_len, torch.device('cpu'))
    relative_encoding = relative.get_relative_encoding(seq_len)

    comparison['encodings'] = {
        'sinusoidal': {
            'shape': sine_encoding.shape,
            'deterministic': True,
            'parameters': 0,
            'properties': "Fixed, deterministic, allows extrapolation to longer sequences"
        },
        'learned': {
            'shape': learned_encoding.shape,
            'deterministic': False,
            'parameters': max_seq_len * d_model,
            'properties': "Learned during training, task-specific, limited to training length"
        },
        'relative': {
            'shape': relative_encoding.shape,
            'deterministic': True,
            'parameters': 0,
            'properties': "Fixed, focuses on relative distances, translation invariant"
        }
    }

    # Analyze properties
    comparison['analysis'] = {
        'parameter_efficiency': {
            'sinusoidal': "No parameters needed",
            'learned': f"{max_seq_len * d_model:,} parameters",
            'relative': "No parameters needed"
        },
        'extrapolation': {
            'sinusoidal': "Can handle longer sequences than trained on",
            'learned': "Cannot handle longer sequences than trained on",
            'relative': "Can handle longer sequences with proper clipping"
        },
        'training_requirements': {
            'sinusoidal': "No training needed",
            'learned': "Requires training with position supervision",
            'relative': "No training needed"
        }
    }

    return comparison


def educational_position_examples() -> Dict[str, Any]:
    """
    Create educational examples demonstrating position encoding concepts.

    Returns:
        Dictionary with examples and explanations
    """
    examples = {}

    print("Creating position encoding examples...")

    # Example 1: Why position encoding is needed
    d_model = 8
    seq_len = 4

    # Create identical token embeddings
    token_embeddings = torch.tensor([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # "the"
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # "cat"
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # "sat"
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # "the" (repeated)
    ])

    # Without position encoding, transformer can't distinguish token positions
    sine_encoder = SinusoidalPositionEncoding(d_model, 100)
    position_encodings = sine_encoder(seq_len)

    # Add position encodings
    tokens_with_position = token_embeddings + position_encodings

    examples['necessity_demonstration'] = {
        'token_embeddings': token_embeddings,
        'position_encodings': position_encodings,
        'combined': tokens_with_position,
        'explanation': {
            'problem': "Without position encoding, 'the' at position 0 and 3 are identical",
            'solution': "Position encoding makes each position unique",
            'result': "Now position 0 'the' and position 3 'the' have different representations"
        }
    }

    # Example 2: Frequency patterns in sinusoidal encoding
    sine_encoder_big = SinusoidalPositionEncoding(64, 100)
    vis_data = sine_encoder_big.visualize_encoding(seq_len=50, dims_to_show=8)

    examples['frequency_patterns'] = {
        'visualization_data': vis_data,
        'explanation': "Different dimensions have different frequencies for unique position representation"
    }

    # Example 3: Position encoding properties
    properties = sine_encoder_big.demonstrate_properties([0, 1, 5, 10])

    examples['encoding_properties'] = {
        'properties': properties,
        'explanation': "Mathematical properties that make position encoding effective"
    }

    return examples


if __name__ == "__main__":
    print("Position Encoding for Transformers")
    print("=" * 50)

    # Run educational examples
    examples = educational_position_examples()

    print("\n1. Why position encoding is necessary:")
    necessity = examples['necessity_demonstration']
    print("Token 'the' at position 0:", necessity['token_embeddings'][0][:4].tolist())
    print("Token 'the' at position 3:", necessity['token_embeddings'][3][:4].tolist())
    print("These are identical without position encoding!")

    print("\nWith position encoding added:")
    print("Position 0 'the':", necessity['combined'][0][:4].round(decimals=3).tolist())
    print("Position 3 'the':", necessity['combined'][3][:4].round(decimals=3).tolist())
    print("Now they are different!")

    print("\n2. Sinusoidal encoding properties:")
    properties = examples['encoding_properties']['properties']

    if 'position_similarities' in properties:
        print("Position similarities:")
        for pair, similarity in properties['position_similarities']['similarities'].items():
            print(f"  {pair}: {similarity:.3f}")

    if 'frequency_analysis' in properties:
        print("Frequency analysis (zero crossings per 100 positions):")
        for i, freq in enumerate(properties['frequency_analysis']['frequencies']):
            print(f"  Dimension {i*2}: {freq:.3f}")

    print("\n3. Comparison of position encoding methods:")
    comparison = compare_position_encodings()

    for method, info in comparison['encodings'].items():
        print(f"\n{method.capitalize()}:")
        print(f"  Shape: {info['shape']}")
        print(f"  Deterministic: {info['deterministic']}")
        print(f"  Parameters: {info['parameters']}")
        print(f"  Properties: {info['properties']}")

    print("\n4. Analysis summary:")
    analysis = comparison['analysis']
    print("Parameter efficiency:")
    for method, params in analysis['parameter_efficiency'].items():
        print(f"  {method}: {params}")

    print("\nExtrapolation capability:")
    for method, capability in analysis['extrapolation'].items():
        print(f"  {method}: {capability}")

    print("\n✅ All position encoding mechanisms implemented!")