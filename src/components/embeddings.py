"""
Token and Position Embeddings for Educational Transformer

This module implements the embedding layer that converts discrete tokens into
continuous vector representations and adds positional information.

Key concepts covered:
- Token embeddings: Converting discrete tokens to continuous vectors
- Position embeddings: Adding positional information
- Embedding combination strategies
- Vocabulary and embedding space understanding
- Educational analysis and visualization tools

Following ARENA Chapter 1.1 curriculum.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, Union
import math
import numpy as np

try:
    from ..foundations.position_encoding import (
        SinusoidalPositionEncoding,
        LearnedPositionEmbedding
    )
    from ..models.config import TransformerConfig
except ImportError:
    from src.foundations.position_encoding import (
        SinusoidalPositionEncoding,
        LearnedPositionEmbedding
    )
    from src.models.config import TransformerConfig


class TokenEmbedding(nn.Module):
    """
    Educational implementation of token embeddings.

    Token embeddings convert discrete token IDs into continuous vector
    representations that the transformer can process. This is typically
    the first layer of any transformer model.
    """

    def __init__(self, vocab_size: int, d_model: int, padding_idx: Optional[int] = None):
        """
        Initialize token embedding layer.

        Args:
            vocab_size: Size of vocabulary
            d_model: Embedding dimension
            padding_idx: Index of padding token (if any)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.padding_idx = padding_idx

        # The embedding lookup table
        self.embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=padding_idx
        )

        # Initialize embeddings with scaled random values
        # This follows the original Transformer paper initialization
        self._init_embeddings()

    def _init_embeddings(self):
        """Initialize embedding weights following best practices."""
        # Use normal distribution with std = 1/sqrt(d_model)
        # This ensures reasonable initial magnitudes
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0 / math.sqrt(self.d_model))

        # Set padding token embedding to zero if specified
        if self.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.padding_idx].fill_(0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert token IDs to embeddings.

        Args:
            token_ids: Integer token IDs [batch, seq_len]

        Returns:
            Token embeddings [batch, seq_len, d_model]
        """
        # Get embeddings and scale by sqrt(d_model)
        # Scaling is from original Transformer paper to counteract
        # the effect of the attention scaling
        embeddings = self.embedding(token_ids)
        return embeddings * math.sqrt(self.d_model)

    def get_embedding_by_token(self, token_id: int) -> torch.Tensor:
        """
        Get embedding vector for a specific token.

        Args:
            token_id: Token ID

        Returns:
            Embedding vector [d_model]
        """
        if token_id >= self.vocab_size:
            raise ValueError(f"Token ID {token_id} exceeds vocabulary size {self.vocab_size}")

        return self.embedding.weight[token_id] * math.sqrt(self.d_model)

    def find_similar_tokens(self, token_id: int, top_k: int = 5) -> Dict[str, Any]:
        """
        Find tokens with similar embeddings (educational analysis).

        Args:
            token_id: Reference token ID
            top_k: Number of similar tokens to return

        Returns:
            Dictionary with similar tokens and similarities
        """
        if token_id >= self.vocab_size:
            raise ValueError(f"Token ID {token_id} exceeds vocabulary size {self.vocab_size}")

        # Get reference embedding
        ref_embedding = self.get_embedding_by_token(token_id)

        # Compute similarities with all other tokens
        all_embeddings = self.embedding.weight * math.sqrt(self.d_model)
        similarities = F.cosine_similarity(
            ref_embedding.unsqueeze(0), all_embeddings, dim=1
        )

        # Get top-k similar tokens (excluding the token itself)
        similarities[token_id] = -1  # Exclude self
        top_similarities, top_indices = torch.topk(similarities, top_k)

        return {
            'reference_token_id': token_id,
            'similar_tokens': {
                'token_ids': top_indices.tolist(),
                'similarities': top_similarities.tolist()
            },
            'explanation': f"Tokens most similar to token {token_id} in embedding space"
        }


class CombinedEmbedding(nn.Module):
    """
    Combined token and position embeddings for transformers.

    This class combines token embeddings with positional information using
    various strategies. It serves as the complete input embedding layer.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int,
        position_encoding_type: str = "sinusoidal",
        padding_idx: Optional[int] = None,
        dropout: float = 0.1
    ):
        """
        Initialize combined embeddings.

        Args:
            vocab_size: Size of vocabulary
            d_model: Embedding dimension
            max_seq_len: Maximum sequence length
            position_encoding_type: "sinusoidal" or "learned"
            padding_idx: Index of padding token
            dropout: Dropout probability
        """
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.position_encoding_type = position_encoding_type

        # Token embeddings
        self.token_embedding = TokenEmbedding(vocab_size, d_model, padding_idx)

        # Position encodings
        if position_encoding_type == "sinusoidal":
            self.position_encoding = SinusoidalPositionEncoding(d_model, max_seq_len)
            self.learnable_positions = False
        elif position_encoding_type == "learned":
            self.position_embedding = LearnedPositionEmbedding(max_seq_len, d_model)
            self.learnable_positions = True
        else:
            raise ValueError(f"Unknown position encoding type: {position_encoding_type}")

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Layer norm (optional, sometimes used in modern transformers)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        token_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        use_layer_norm: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined token and position embeddings.

        Args:
            token_ids: Token IDs [batch, seq_len]
            position_ids: Optional position IDs [batch, seq_len]
            use_layer_norm: Whether to apply layer norm

        Returns:
            combined_embeddings: [batch, seq_len, d_model]
            debug_info: Dictionary with intermediate tensors
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        # Validate sequence length
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}")

        # Get token embeddings
        token_embeddings = self.token_embedding(token_ids)

        # Get position embeddings
        if self.learnable_positions:
            position_embeddings = self.position_embedding(seq_len, device)
            position_embeddings = position_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            position_encodings = self.position_encoding(seq_len)
            position_embeddings = position_encodings.to(device).unsqueeze(0).expand(batch_size, -1, -1)

        # Combine embeddings (typically addition)
        combined = token_embeddings + position_embeddings

        # Apply layer normalization if requested
        if use_layer_norm:
            combined = self.layer_norm(combined)

        # Apply dropout
        combined = self.dropout(combined)

        # Prepare debug information
        debug_info = {
            'token_embeddings': token_embeddings,
            'position_embeddings': position_embeddings,
            'combined_before_dropout': token_embeddings + position_embeddings,
            'combined_after_dropout': combined
        }

        return combined, debug_info

    def analyze_embedding_components(
        self,
        token_ids: torch.Tensor,
        position_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze the contribution of different embedding components.

        Args:
            token_ids: Token IDs to analyze [batch, seq_len]
            position_analysis: Whether to include position analysis

        Returns:
            Dictionary with analysis results
        """
        with torch.no_grad():
            combined, debug_info = self.forward(token_ids, use_layer_norm=False)

            analysis = {}

            # Component magnitudes
            token_emb = debug_info['token_embeddings']
            pos_emb = debug_info['position_embeddings']

            analysis['component_magnitudes'] = {
                'token_embeddings': {
                    'mean_norm': torch.norm(token_emb, dim=-1).mean().item(),
                    'std_norm': torch.norm(token_emb, dim=-1).std().item(),
                    'explanation': "Average L2 norm of token embeddings"
                },
                'position_embeddings': {
                    'mean_norm': torch.norm(pos_emb, dim=-1).mean().item(),
                    'std_norm': torch.norm(pos_emb, dim=-1).std().item(),
                    'explanation': "Average L2 norm of position embeddings"
                },
                'combined': {
                    'mean_norm': torch.norm(combined, dim=-1).mean().item(),
                    'std_norm': torch.norm(combined, dim=-1).std().item(),
                    'explanation': "Average L2 norm of combined embeddings"
                }
            }

            # Cosine similarity between token and position embeddings
            cos_sim = F.cosine_similarity(token_emb, pos_emb, dim=-1)
            analysis['token_position_interaction'] = {
                'mean_cosine_similarity': cos_sim.mean().item(),
                'std_cosine_similarity': cos_sim.std().item(),
                'explanation': "How aligned token and position embeddings are"
            }

            # Variance explained by each component
            total_var = torch.var(combined, dim=-1).mean().item()
            token_var = torch.var(token_emb, dim=-1).mean().item()
            pos_var = torch.var(pos_emb, dim=-1).mean().item()

            analysis['variance_analysis'] = {
                'token_variance_contribution': token_var / total_var,
                'position_variance_contribution': pos_var / total_var,
                'total_variance': total_var,
                'explanation': "How much variance each component contributes"
            }

            if position_analysis:
                # Position-specific analysis
                seq_len = token_ids.shape[1]
                position_norms = torch.norm(pos_emb, dim=-1)[0]  # First batch item

                analysis['position_patterns'] = {
                    'position_norms': position_norms.tolist(),
                    'norm_trend': 'increasing' if position_norms[-1] > position_norms[0] else 'decreasing',
                    'explanation': "How position embedding magnitude changes across sequence"
                }

        return analysis


class EmbeddingSpace:
    """
    Educational tools for analyzing embedding spaces.

    This class provides utilities for understanding the geometry and
    properties of the learned embedding space.
    """

    def __init__(self, embedding_layer: TokenEmbedding):
        """
        Initialize embedding space analyzer.

        Args:
            embedding_layer: Token embedding layer to analyze
        """
        self.embedding_layer = embedding_layer
        self.vocab_size = embedding_layer.vocab_size
        self.d_model = embedding_layer.d_model

    def compute_embedding_statistics(self) -> Dict[str, Any]:
        """
        Compute various statistics about the embedding space.

        Returns:
            Dictionary with embedding statistics
        """
        with torch.no_grad():
            # Get all embeddings (without scaling)
            all_embeddings = self.embedding_layer.embedding.weight

            stats = {}

            # Basic statistics
            stats['basic'] = {
                'mean': all_embeddings.mean().item(),
                'std': all_embeddings.std().item(),
                'min': all_embeddings.min().item(),
                'max': all_embeddings.max().item()
            }

            # Embedding norms
            embedding_norms = torch.norm(all_embeddings, dim=1)
            stats['norms'] = {
                'mean_norm': embedding_norms.mean().item(),
                'std_norm': embedding_norms.std().item(),
                'min_norm': embedding_norms.min().item(),
                'max_norm': embedding_norms.max().item()
            }

            # Pairwise similarities
            normalized_embeddings = F.normalize(all_embeddings, dim=1)
            similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())

            # Remove diagonal (self-similarities)
            mask = ~torch.eye(self.vocab_size, dtype=torch.bool)
            off_diagonal_similarities = similarity_matrix[mask]

            stats['similarities'] = {
                'mean_similarity': off_diagonal_similarities.mean().item(),
                'std_similarity': off_diagonal_similarities.std().item(),
                'max_similarity': off_diagonal_similarities.max().item(),
                'min_similarity': off_diagonal_similarities.min().item()
            }

            # Embedding space dimensionality analysis
            # Compute effective rank of embedding matrix
            U, S, V = torch.svd(all_embeddings)
            explained_variance_ratio = S / S.sum()
            cumulative_variance = torch.cumsum(explained_variance_ratio, dim=0)

            # Find dimensions needed for 90% and 99% variance
            dims_90 = (cumulative_variance < 0.9).sum().item() + 1
            dims_99 = (cumulative_variance < 0.99).sum().item() + 1

            stats['dimensionality'] = {
                'effective_rank_90': dims_90,
                'effective_rank_99': dims_99,
                'total_dimensions': self.d_model,
                'top_10_singular_values': S[:10].tolist(),
                'explanation': "Number of dimensions needed to capture variance"
            }

            return stats

    def find_embedding_clusters(self, n_clusters: int = 5) -> Dict[str, Any]:
        """
        Find clusters in embedding space using k-means.

        Args:
            n_clusters: Number of clusters to find

        Returns:
            Dictionary with clustering results
        """
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            return {'error': 'sklearn not available for clustering'}

        with torch.no_grad():
            embeddings = self.embedding_layer.embedding.weight.cpu().numpy()

            # Perform k-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)

            # Analyze clusters
            clusters = {}
            for i in range(n_clusters):
                cluster_tokens = np.where(cluster_labels == i)[0]
                cluster_center = kmeans.cluster_centers_[i]

                # Find tokens closest to cluster center
                distances = np.linalg.norm(embeddings[cluster_tokens] - cluster_center, axis=1)
                closest_indices = cluster_tokens[np.argsort(distances)[:5]]

                clusters[f'cluster_{i}'] = {
                    'size': len(cluster_tokens),
                    'center': cluster_center.tolist(),
                    'closest_tokens': closest_indices.tolist(),
                    'all_tokens': cluster_tokens.tolist()
                }

            return {
                'clusters': clusters,
                'n_clusters': n_clusters,
                'inertia': kmeans.inertia_,
                'explanation': "K-means clustering of embedding space"
            }


def create_educational_examples() -> Dict[str, Any]:
    """
    Create educational examples for understanding embeddings.

    Returns:
        Dictionary with examples and explanations
    """
    examples = {}

    # Example 1: Small vocabulary embedding
    print("Creating small vocabulary example...")
    vocab_size, d_model, seq_len = 10, 8, 5

    # Create simple embedding layer
    token_emb = TokenEmbedding(vocab_size, d_model)

    # Create sample tokens
    sample_tokens = torch.randint(0, vocab_size, (2, seq_len))

    # Get embeddings
    embeddings = token_emb(sample_tokens)

    examples['small_vocabulary'] = {
        'vocab_size': vocab_size,
        'd_model': d_model,
        'sample_tokens': sample_tokens,
        'embeddings': embeddings,
        'embedding_shape': embeddings.shape,
        'explanation': "Simple example with small vocabulary"
    }

    # Example 2: Position encoding effects
    print("Creating position encoding effects example...")
    config = TransformerConfig.educational_config()
    combined_emb = CombinedEmbedding(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        max_seq_len=config.max_position_embeddings,
        position_encoding_type="sinusoidal"
    )

    # Test with repeated tokens
    repeated_tokens = torch.tensor([[1, 2, 1, 2, 1]])  # Repeated pattern
    combined, debug_info = combined_emb(repeated_tokens)

    examples['position_effects'] = {
        'repeated_tokens': repeated_tokens,
        'token_embeddings': debug_info['token_embeddings'],
        'position_embeddings': debug_info['position_embeddings'],
        'combined_embeddings': combined,
        'explanation': "How position encoding affects repeated tokens"
    }

    # Example 3: Embedding similarity analysis
    print("Creating embedding similarity analysis...")
    analyzer = EmbeddingSpace(token_emb)
    stats = analyzer.compute_embedding_statistics()

    examples['embedding_analysis'] = {
        'statistics': stats,
        'explanation': "Statistical analysis of embedding space properties"
    }

    return examples


if __name__ == "__main__":
    print("Token and Position Embeddings")
    print("=" * 50)

    # Run educational examples
    examples = create_educational_examples()

    print("\n1. Small vocabulary embedding example:")
    small_example = examples['small_vocabulary']
    print(f"Vocabulary size: {small_example['vocab_size']}")
    print(f"Embedding dimension: {small_example['d_model']}")
    print(f"Sample tokens: {small_example['sample_tokens'][0].tolist()}")
    print(f"Embedding shape: {small_example['embedding_shape']}")

    print("\n2. Position encoding effects:")
    pos_example = examples['position_effects']
    print(f"Repeated tokens: {pos_example['repeated_tokens'][0].tolist()}")
    print("Token embeddings for first two positions are identical:")
    print(f"Position 0: {pos_example['token_embeddings'][0, 0, :3].round(decimals=3).tolist()}")
    print(f"Position 2: {pos_example['token_embeddings'][0, 2, :3].round(decimals=3).tolist()}")
    print("But combined embeddings are different due to position encoding:")
    print(f"Position 0: {pos_example['combined_embeddings'][0, 0, :3].round(decimals=3).tolist()}")
    print(f"Position 2: {pos_example['combined_embeddings'][0, 2, :3].round(decimals=3).tolist()}")

    print("\n3. Embedding space analysis:")
    analysis = examples['embedding_analysis']['statistics']
    print("Basic statistics:")
    print(f"  Mean: {analysis['basic']['mean']:.4f}")
    print(f"  Std: {analysis['basic']['std']:.4f}")
    print("Embedding norms:")
    print(f"  Mean norm: {analysis['norms']['mean_norm']:.4f}")
    print(f"  Std norm: {analysis['norms']['std_norm']:.4f}")
    print("Pairwise similarities:")
    print(f"  Mean similarity: {analysis['similarities']['mean_similarity']:.4f}")
    print(f"  Std similarity: {analysis['similarities']['std_similarity']:.4f}")
    print("Dimensionality:")
    print(f"  Effective rank (90%): {analysis['dimensionality']['effective_rank_90']}")
    print(f"  Effective rank (99%): {analysis['dimensionality']['effective_rank_99']}")

    print("\n4. Testing combined embeddings:")
    config = TransformerConfig.educational_config()

    # Test both sinusoidal and learned position encodings
    for pos_type in ["sinusoidal", "learned"]:
        print(f"\n{pos_type.capitalize()} position encoding:")
        combined_emb = CombinedEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_seq_len=config.max_position_embeddings,
            position_encoding_type=pos_type,
            dropout=0.0  # No dropout for testing
        )

        test_tokens = torch.randint(0, 100, (1, 10))  # Random tokens
        output, debug = combined_emb(test_tokens)

        print(f"  Input shape: {test_tokens.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Learnable positions: {combined_emb.learnable_positions}")

        # Analyze component contributions
        analysis = combined_emb.analyze_embedding_components(test_tokens)
        print(f"  Token embedding norm: {analysis['component_magnitudes']['token_embeddings']['mean_norm']:.4f}")
        print(f"  Position embedding norm: {analysis['component_magnitudes']['position_embeddings']['mean_norm']:.4f}")
        print(f"  Token-position similarity: {analysis['token_position_interaction']['mean_cosine_similarity']:.4f}")

    print("\n✅ All embedding components implemented!")