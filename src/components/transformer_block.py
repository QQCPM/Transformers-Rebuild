"""
Complete Transformer Block

Educational implementation of a complete transformer block that combines
multi-head attention, MLP, layer normalization, and residual connections.
This is the fundamental building block of transformer architectures.

Key concepts covered:
- Residual connections and their importance
- Pre-norm vs post-norm architectures
- Layer normalization placement
- Residual stream analysis
- Component contribution analysis
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
    from .attention import MultiHeadAttention
    from .mlp import TransformerMLP, LayerNorm
    from ..models.config import TransformerConfig
except ImportError:
    from src.components.attention import MultiHeadAttention
    from src.components.mlp import TransformerMLP, LayerNorm
    from src.models.config import TransformerConfig


class TransformerBlock(nn.Module):
    """
    Educational implementation of a complete transformer block.

    This implementation uses pre-norm architecture for better training stability
    and includes comprehensive educational analysis tools.

    Architecture (Pre-norm):
    1. x_attn = x + Attention(LayerNorm(x))
    2. x_out = x_attn + MLP(LayerNorm(x_attn))

    The residual stream (x -> x_attn -> x_out) is the core concept that
    allows information to flow directly through the network while allowing
    each component to make targeted modifications.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_idx: int = 0,
        use_pre_norm: bool = True
    ):
        """
        Initialize transformer block.

        Args:
            config: Transformer configuration
            layer_idx: Layer index (for debugging and analysis)
            use_pre_norm: Whether to use pre-norm (True) or post-norm (False) architecture
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.use_pre_norm = use_pre_norm
        self.d_model = config.d_model

        # Multi-head attention
        self.attention = MultiHeadAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.attention_dropout,
            is_causal=True  # Assuming causal/autoregressive model
        )

        # MLP (feed-forward network)
        self.mlp = TransformerMLP(
            d_model=config.d_model,
            d_mlp=config.d_mlp,
            dropout=config.dropout,
            activation="gelu"
        )

        # Layer normalization
        self.ln1 = LayerNorm(config.d_model, eps=config.layer_norm_eps)  # Pre-attention norm
        self.ln2 = LayerNorm(config.d_model, eps=config.layer_norm_eps)  # Pre-MLP norm

        # Post-norm layers (only used if use_pre_norm=False)
        if not use_pre_norm:
            self.ln1_post = LayerNorm(config.d_model, eps=config.layer_norm_eps)
            self.ln2_post = LayerNorm(config.d_model, eps=config.layer_norm_eps)

        # Hook points for TransformerLens compatibility and educational analysis
        self.hook_residual_pre = None      # Before any processing
        self.hook_attn_pre = None          # Before attention (after first layer norm)
        self.hook_attn_out = None          # Attention output (before residual addition)
        self.hook_residual_mid = None      # After attention residual addition
        self.hook_mlp_pre = None           # Before MLP (after second layer norm)
        self.hook_mlp_out = None           # MLP output (before residual addition)
        self.hook_residual_post = None     # After final residual addition

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
        return_debug_info: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict]]:
        """
        Forward pass through transformer block.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            attention_mask: Optional attention mask
            return_attention_weights: Whether to return attention weights
            return_debug_info: Whether to return detailed debug information

        Returns:
            output: Block output [batch, seq_len, d_model]
            attention_weights: Optional attention weights
            debug_info: Optional debug information
        """
        # Store original input for residual connections
        residual_input = x

        # Apply hook if present
        if self.hook_residual_pre is not None:
            x = self.hook_residual_pre(x)

        debug_info = {}
        if return_debug_info:
            debug_info['input_stats'] = self._compute_tensor_stats(x, 'input')

        # === ATTENTION SUB-BLOCK ===
        if self.use_pre_norm:
            # Pre-norm: LayerNorm -> Attention -> Residual
            x_normed = self.ln1(x)
        else:
            x_normed = x

        # Apply hook if present
        if self.hook_attn_pre is not None:
            x_normed = self.hook_attn_pre(x_normed)

        if return_debug_info:
            debug_info['attn_pre_norm_stats'] = self._compute_tensor_stats(x_normed, 'attn_pre_norm')

        # Multi-head attention
        if return_attention_weights:
            attn_output, attention_weights = self.attention(
                x_normed, attention_mask=attention_mask, return_attention_weights=True
            )
        else:
            attn_output = self.attention(
                x_normed, attention_mask=attention_mask, return_attention_weights=False
            )
            attention_weights = None

        # Apply hook if present
        if self.hook_attn_out is not None:
            attn_output = self.hook_attn_out(attn_output)

        if return_debug_info:
            debug_info['attn_output_stats'] = self._compute_tensor_stats(attn_output, 'attn_output')

        # Residual connection for attention
        if self.use_pre_norm:
            x = x + attn_output  # Pre-norm: residual addition
        else:
            x = self.ln1_post(x + attn_output)  # Post-norm: residual then norm

        # Apply hook if present
        if self.hook_residual_mid is not None:
            x = self.hook_residual_mid(x)

        if return_debug_info:
            debug_info['residual_mid_stats'] = self._compute_tensor_stats(x, 'residual_mid')

        # === MLP SUB-BLOCK ===
        residual_mid = x  # Store for MLP residual connection

        if self.use_pre_norm:
            # Pre-norm: LayerNorm -> MLP -> Residual
            x_normed = self.ln2(x)
        else:
            x_normed = x

        # Apply hook if present
        if self.hook_mlp_pre is not None:
            x_normed = self.hook_mlp_pre(x_normed)

        if return_debug_info:
            debug_info['mlp_pre_norm_stats'] = self._compute_tensor_stats(x_normed, 'mlp_pre_norm')

        # MLP forward pass
        mlp_output = self.mlp(x_normed)

        # Apply hook if present
        if self.hook_mlp_out is not None:
            mlp_output = self.hook_mlp_out(mlp_output)

        if return_debug_info:
            debug_info['mlp_output_stats'] = self._compute_tensor_stats(mlp_output, 'mlp_output')

        # Residual connection for MLP
        if self.use_pre_norm:
            x = residual_mid + mlp_output  # Pre-norm: residual addition
        else:
            x = self.ln2_post(residual_mid + mlp_output)  # Post-norm: residual then norm

        # Apply hook if present
        if self.hook_residual_post is not None:
            x = self.hook_residual_post(x)

        if return_debug_info:
            debug_info['output_stats'] = self._compute_tensor_stats(x, 'output')
            debug_info.update(self._analyze_residual_stream(residual_input, attn_output, mlp_output, x))

        # Return based on requested outputs
        if return_debug_info:
            if return_attention_weights:
                return x, attention_weights, debug_info
            else:
                return x, debug_info
        elif return_attention_weights:
            return x, attention_weights
        else:
            return x

    def _compute_tensor_stats(self, tensor: torch.Tensor, name: str) -> Dict[str, Any]:
        """Compute comprehensive statistics for a tensor."""
        with torch.no_grad():
            return {
                'name': name,
                'shape': list(tensor.shape),
                'mean': tensor.mean().item(),
                'std': tensor.std().item(),
                'min': tensor.min().item(),
                'max': tensor.max().item(),
                'norm': torch.norm(tensor, dim=-1).mean().item(),
                'abs_max': tensor.abs().max().item()
            }

    def _analyze_residual_stream(
        self,
        input_tensor: torch.Tensor,
        attn_output: torch.Tensor,
        mlp_output: torch.Tensor,
        final_output: torch.Tensor
    ) -> Dict[str, Any]:
        """Analyze the residual stream and component contributions."""
        with torch.no_grad():
            analysis = {}

            # Component magnitudes
            input_norm = torch.norm(input_tensor, dim=-1).mean().item()
            attn_norm = torch.norm(attn_output, dim=-1).mean().item()
            mlp_norm = torch.norm(mlp_output, dim=-1).mean().item()
            output_norm = torch.norm(final_output, dim=-1).mean().item()

            analysis['component_magnitudes'] = {
                'input_norm': input_norm,
                'attention_contribution_norm': attn_norm,
                'mlp_contribution_norm': mlp_norm,
                'output_norm': output_norm,
                'relative_attention_contribution': attn_norm / input_norm if input_norm > 0 else 0,
                'relative_mlp_contribution': mlp_norm / input_norm if input_norm > 0 else 0
            }

            # Residual stream flow analysis
            intermediate = input_tensor + attn_output  # After attention residual

            # How much each component changes the representation
            attn_change = torch.norm(intermediate - input_tensor, dim=-1).mean().item()
            mlp_change = torch.norm(final_output - intermediate, dim=-1).mean().item()
            total_change = torch.norm(final_output - input_tensor, dim=-1).mean().item()

            analysis['representation_changes'] = {
                'attention_change': attn_change,
                'mlp_change': mlp_change,
                'total_change': total_change,
                'attention_change_ratio': attn_change / total_change if total_change > 0 else 0,
                'mlp_change_ratio': mlp_change / total_change if total_change > 0 else 0
            }

            # Directional analysis (how much components align with input)
            input_flat = input_tensor.view(-1)
            attn_flat = attn_output.view(-1)
            mlp_flat = mlp_output.view(-1)
            output_flat = final_output.view(-1)

            analysis['directional_analysis'] = {
                'input_attn_similarity': F.cosine_similarity(input_flat, attn_flat, dim=0).item(),
                'input_mlp_similarity': F.cosine_similarity(input_flat, mlp_flat, dim=0).item(),
                'input_output_similarity': F.cosine_similarity(input_flat, output_flat, dim=0).item(),
                'attn_mlp_similarity': F.cosine_similarity(attn_flat, mlp_flat, dim=0).item()
            }

            # Information preservation analysis
            analysis['information_preservation'] = {
                'magnitude_preservation': output_norm / input_norm if input_norm > 0 else 0,
                'direction_preservation': analysis['directional_analysis']['input_output_similarity'],
                'explanation': "How well the block preserves vs transforms the input information"
            }

            return {'residual_stream_analysis': analysis}

    def analyze_layer_behavior(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        n_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of transformer block behavior.

        Args:
            x: Input tensor for analysis
            attention_mask: Optional attention mask
            n_samples: Number of samples to analyze

        Returns:
            Dictionary with detailed behavioral analysis
        """
        self.eval()
        with torch.no_grad():
            # Forward pass with full debug info
            output, attention_weights, debug_info = self.forward(
                x, attention_mask=attention_mask,
                return_attention_weights=True, return_debug_info=True
            )

            analysis = {
                'layer_index': self.layer_idx,
                'architecture': 'pre-norm' if self.use_pre_norm else 'post-norm',
                'debug_info': debug_info
            }

            # Attention pattern analysis
            if attention_weights is not None:
                analysis['attention_analysis'] = self.attention.analyze_attention_patterns(x)

            # MLP behavior analysis
            # Get the input to MLP (after attention and layer norm)
            if self.use_pre_norm:
                mlp_input = self.ln2(x + self.attention(self.ln1(x), attention_mask=attention_mask, return_attention_weights=False))
            else:
                attn_out = self.attention(x, attention_mask=attention_mask, return_attention_weights=False)
                mlp_input = self.ln1_post(x + attn_out) if hasattr(self, 'ln1_post') else x + attn_out

            analysis['mlp_analysis'] = self.mlp.analyze_mlp_behavior(mlp_input, n_samples=n_samples)

            # Layer normalization effects
            if self.use_pre_norm:
                ln1_analysis = self.ln1.analyze_normalization_effects(x)
                ln2_input = x + self.attention(self.ln1(x), attention_mask=attention_mask, return_attention_weights=False)
                ln2_analysis = self.ln2.analyze_normalization_effects(ln2_input)
            else:
                # For post-norm, analyze the pre-norm inputs
                attn_out = self.attention(x, attention_mask=attention_mask, return_attention_weights=False)
                ln1_analysis = self.ln1_post.analyze_normalization_effects(x + attn_out) if hasattr(self, 'ln1_post') else None

                mlp_out = self.mlp(mlp_input)
                ln2_analysis = self.ln2_post.analyze_normalization_effects(mlp_input + mlp_out) if hasattr(self, 'ln2_post') else None

            analysis['layer_norm_analysis'] = {
                'ln1_effects': ln1_analysis,
                'ln2_effects': ln2_analysis
            }

            # Component interaction analysis
            analysis['component_interactions'] = self._analyze_component_interactions(x, attention_mask)

            return analysis

    def _analyze_component_interactions(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Analyze how attention and MLP components interact."""
        with torch.no_grad():
            # Get individual component outputs
            if self.use_pre_norm:
                x_ln1 = self.ln1(x)
                attn_out = self.attention(x_ln1, attention_mask=attention_mask, return_attention_weights=False)
                x_mid = x + attn_out
                x_ln2 = self.ln2(x_mid)
                mlp_out = self.mlp(x_ln2)
            else:
                attn_out = self.attention(x, attention_mask=attention_mask, return_attention_weights=False)
                x_mid = self.ln1_post(x + attn_out) if hasattr(self, 'ln1_post') else x + attn_out
                mlp_out = self.mlp(x_mid)

            interactions = {}

            # Measure how much each component depends on the previous
            interactions['attention_dependency'] = {
                'input_sensitivity': self._measure_input_sensitivity(x, 'attention'),
                'explanation': "How sensitive attention is to input variations"
            }

            interactions['mlp_dependency'] = {
                'attention_sensitivity': self._measure_attention_dependency(x, attention_mask),
                'explanation': "How much MLP behavior depends on attention output"
            }

            # Measure component orthogonality
            attn_flat = attn_out.view(-1)
            mlp_flat = mlp_out.view(-1)

            interactions['component_orthogonality'] = {
                'attn_mlp_cosine_similarity': F.cosine_similarity(attn_flat, mlp_flat, dim=0).item(),
                'explanation': "How orthogonal (independent) attention and MLP contributions are"
            }

            # Analyze component dominance
            attn_norm = torch.norm(attn_out, dim=-1).mean().item()
            mlp_norm = torch.norm(mlp_out, dim=-1).mean().item()

            interactions['component_dominance'] = {
                'attention_dominance': attn_norm / (attn_norm + mlp_norm),
                'mlp_dominance': mlp_norm / (attn_norm + mlp_norm),
                'explanation': "Which component contributes more to the output"
            }

            return interactions

    def _measure_input_sensitivity(self, x: torch.Tensor, component: str) -> float:
        """Measure how sensitive a component is to input perturbations."""
        # Add small random perturbation
        perturbation = torch.randn_like(x) * 0.01
        x_perturbed = x + perturbation

        if component == 'attention':
            if self.use_pre_norm:
                output_original = self.attention(self.ln1(x), return_attention_weights=False)
                output_perturbed = self.attention(self.ln1(x_perturbed), return_attention_weights=False)
            else:
                output_original = self.attention(x, return_attention_weights=False)
                output_perturbed = self.attention(x_perturbed, return_attention_weights=False)
        else:  # MLP
            if self.use_pre_norm:
                output_original = self.mlp(self.ln2(x))
                output_perturbed = self.mlp(self.ln2(x_perturbed))
            else:
                output_original = self.mlp(x)
                output_perturbed = self.mlp(x_perturbed)

        # Compute sensitivity as ratio of output change to input change
        output_change = torch.norm(output_perturbed - output_original).item()
        input_change = torch.norm(perturbation).item()

        return output_change / input_change if input_change > 0 else 0

    def _measure_attention_dependency(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> float:
        """Measure how much MLP output depends on attention output."""
        # Get MLP output with normal attention
        if self.use_pre_norm:
            attn_out = self.attention(self.ln1(x), attention_mask=attention_mask, return_attention_weights=False)
            mlp_input_normal = self.ln2(x + attn_out)
        else:
            attn_out = self.attention(x, attention_mask=attention_mask, return_attention_weights=False)
            mlp_input_normal = self.ln1_post(x + attn_out) if hasattr(self, 'ln1_post') else x + attn_out

        mlp_output_normal = self.mlp(mlp_input_normal)

        # Get MLP output with zero attention (attention disabled)
        if self.use_pre_norm:
            mlp_input_no_attn = self.ln2(x)
        else:
            mlp_input_no_attn = self.ln1_post(x) if hasattr(self, 'ln1_post') else x

        mlp_output_no_attn = self.mlp(mlp_input_no_attn)

        # Compute dependency as normalized difference
        output_diff = torch.norm(mlp_output_normal - mlp_output_no_attn).item()
        output_norm = torch.norm(mlp_output_normal).item()

        return output_diff / output_norm if output_norm > 0 else 0


def create_educational_transformer_block_examples() -> Dict[str, Any]:
    """
    Create educational examples for understanding transformer blocks.

    Returns:
        Dictionary with examples and explanations
    """
    examples = {}

    print("Creating transformer block educational examples...")

    # Example 1: Basic transformer block functionality
    config = TransformerConfig.educational_config()
    transformer_block = TransformerBlock(config, layer_idx=0, use_pre_norm=True)

    # Create sample input
    batch_size, seq_len = 2, 8
    input_tensor = torch.randn(batch_size, seq_len, config.d_model) * 0.1

    # Forward pass with full analysis
    output, attention_weights, debug_info = transformer_block(
        input_tensor,
        return_attention_weights=True,
        return_debug_info=True
    )

    examples['basic_transformer_block'] = {
        'input_shape': input_tensor.shape,
        'output_shape': output.shape,
        'attention_weights_shape': attention_weights.shape,
        'debug_info': debug_info,
        'explanation': "Complete transformer block forward pass with residual stream analysis"
    }

    # Example 2: Pre-norm vs Post-norm comparison
    pre_norm_block = TransformerBlock(config, layer_idx=0, use_pre_norm=True)
    post_norm_block = TransformerBlock(config, layer_idx=0, use_pre_norm=False)

    output_pre, debug_pre = pre_norm_block(input_tensor, return_debug_info=True)
    output_post, debug_post = post_norm_block(input_tensor, return_debug_info=True)

    examples['norm_architecture_comparison'] = {
        'pre_norm_output': output_pre,
        'post_norm_output': output_post,
        'output_difference_norm': torch.norm(output_pre - output_post).item(),
        'pre_norm_debug': debug_pre,
        'post_norm_debug': debug_post,
        'explanation': "Comparison between pre-norm and post-norm architectures"
    }

    # Example 3: Comprehensive behavior analysis
    behavior_analysis = transformer_block.analyze_layer_behavior(input_tensor)

    examples['behavior_analysis'] = {
        'analysis': behavior_analysis,
        'explanation': "Detailed analysis of transformer block behavior and component interactions"
    }

    # Example 4: Residual stream visualization
    residual_analysis = debug_info['residual_stream_analysis']

    examples['residual_stream_analysis'] = {
        'analysis': residual_analysis,
        'explanation': "Analysis of how information flows through the residual stream"
    }

    return examples


if __name__ == "__main__":
    print("Complete Transformer Block")
    print("=" * 50)

    # Run educational examples
    examples = create_educational_transformer_block_examples()

    print("\n1. Basic transformer block functionality:")
    basic = examples['basic_transformer_block']
    print(f"Input shape: {basic['input_shape']}")
    print(f"Output shape: {basic['output_shape']}")
    print(f"Attention weights shape: {basic['attention_weights_shape']}")

    # Print residual stream analysis
    residual = basic['debug_info']['residual_stream_analysis']
    print(f"\nResidual stream analysis:")
    magnitudes = residual['component_magnitudes']
    print(f"  Input norm: {magnitudes['input_norm']:.4f}")
    print(f"  Attention contribution: {magnitudes['relative_attention_contribution']:.4f}")
    print(f"  MLP contribution: {magnitudes['relative_mlp_contribution']:.4f}")

    changes = residual['representation_changes']
    print(f"  Attention change ratio: {changes['attention_change_ratio']:.4f}")
    print(f"  MLP change ratio: {changes['mlp_change_ratio']:.4f}")

    print("\n2. Pre-norm vs Post-norm comparison:")
    comparison = examples['norm_architecture_comparison']
    print(f"Output difference norm: {comparison['output_difference_norm']:.6f}")

    pre_residual = comparison['pre_norm_debug']['residual_stream_analysis']
    post_residual = comparison['post_norm_debug']['residual_stream_analysis']

    print(f"Pre-norm attention contribution: {pre_residual['component_magnitudes']['relative_attention_contribution']:.4f}")
    print(f"Post-norm attention contribution: {post_residual['component_magnitudes']['relative_attention_contribution']:.4f}")

    print("\n3. Behavior analysis:")
    behavior = examples['behavior_analysis']['analysis']

    if 'attention_analysis' in behavior:
        attn_analysis = behavior['attention_analysis']
        if 'head_diversity' in attn_analysis:
            print(f"Head diversity (mean similarity): {attn_analysis['head_diversity']['mean_similarity']:.4f}")

    if 'mlp_analysis' in behavior:
        mlp_analysis = behavior['mlp_analysis']
        print(f"MLP expansion rank increase: {mlp_analysis['expansion_phase']['rank_increase']:.2f}")
        print(f"MLP activation sparsity: {mlp_analysis['activation_effects']['activation_sparsity']:.4f}")

    if 'component_interactions' in behavior:
        interactions = behavior['component_interactions']
        print(f"Component orthogonality: {interactions['component_orthogonality']['attn_mlp_cosine_similarity']:.4f}")
        print(f"Attention dominance: {interactions['component_dominance']['attention_dominance']:.4f}")
        print(f"MLP dominance: {interactions['component_dominance']['mlp_dominance']:.4f}")

    print("\n4. Residual stream flow:")
    residual_stream = examples['residual_stream_analysis']['analysis']
    directional = residual_stream['directional_analysis']
    print(f"Input-output similarity: {directional['input_output_similarity']:.4f}")
    print(f"Attention-MLP similarity: {directional['attn_mlp_similarity']:.4f}")

    preservation = residual_stream['information_preservation']
    print(f"Magnitude preservation: {preservation['magnitude_preservation']:.4f}")
    print(f"Direction preservation: {preservation['direction_preservation']:.4f}")

    print("\n5. Configuration validation:")
    test_configs = [
        TransformerConfig.small_config(),
        TransformerConfig.educational_config(),
        TransformerConfig.gpt2_small_config()
    ]

    for i, config in enumerate(test_configs):
        test_block = TransformerBlock(config, layer_idx=i)
        test_input = torch.randn(1, 10, config.d_model) * 0.1
        test_output = test_block(test_input)

        print(f"✓ Config {i+1}: d_model={config.d_model}, n_heads={config.n_heads}")
        print(f"  Input: {test_input.shape} -> Output: {test_output.shape}")

    print("\n✅ Complete transformer block implemented and tested!")