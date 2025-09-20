"""
Complete Educational Transformer Model

This module implements a complete transformer model that is both educational
and compatible with TransformerLens for mechanistic interpretability research.
The implementation follows ARENA Chapter 1.1 curriculum while providing
extensive analysis and debugging capabilities.

Key features:
- Full transformer architecture with configurable components
- TransformerLens compatibility for mechanistic interpretability
- Comprehensive educational analysis tools
- Hook system for activation caching and intervention
- Residual stream analysis across all layers
- Model loading/saving functionality

Following ARENA Chapter 1.1 curriculum.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, Union, List
import math
import numpy as np
from collections import OrderedDict

try:
    from .config import TransformerConfig
    from ..components.embeddings import CombinedEmbedding
    from ..components.transformer_block import TransformerBlock
    from ..components.mlp import LayerNorm
except ImportError:
    from src.models.config import TransformerConfig
    from src.components.embeddings import CombinedEmbedding
    from src.components.transformer_block import TransformerBlock
    from src.components.mlp import LayerNorm


class EducationalTransformer(nn.Module):
    """
    Educational transformer model with TransformerLens compatibility.

    This implementation provides a complete transformer model that can be used
    for both educational purposes and serious mechanistic interpretability research.

    Architecture:
    1. Token + Position Embeddings
    2. Stack of Transformer Blocks
    3. Final Layer Normalization
    4. Language Modeling Head (optional)

    The model supports comprehensive activation caching, intervention through hooks,
    and detailed analysis of the residual stream across all layers.
    """

    def __init__(
        self,
        config: TransformerConfig,
        use_language_head: bool = True,
        tie_embeddings: bool = True
    ):
        """
        Initialize the educational transformer model.

        Args:
            config: Transformer configuration
            use_language_head: Whether to include language modeling head
            tie_embeddings: Whether to tie input and output embeddings
        """
        super().__init__()
        self.config = config
        self.use_language_head = use_language_head
        self.tie_embeddings = tie_embeddings

        # Core model components
        self.embed = CombinedEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_seq_len=config.max_position_embeddings,
            position_encoding_type="sinusoidal",  # Can be made configurable
            dropout=config.dropout
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config, layer_idx=i)
            for i in range(config.n_layers)
        ])

        # Final layer normalization
        self.ln_f = LayerNorm(config.d_model, eps=config.layer_norm_eps)

        # Language modeling head (optional)
        if use_language_head:
            if tie_embeddings:
                # Share weights with token embeddings
                self.lm_head = None  # Will use embed.token_embedding.embedding.weight
            else:
                self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights
        self._init_weights()

        # Hook system for TransformerLens compatibility
        self.hook_dict = {}
        self._setup_hooks()

        # Cache for storing activations (TransformerLens style)
        self.cache = {}

    def _init_weights(self):
        """Initialize model weights following best practices."""
        # Initialize language modeling head if present and not tied
        if self.use_language_head and not self.tie_embeddings and self.lm_head is not None:
            nn.init.normal_(self.lm_head.weight, std=0.02)

    def _setup_hooks(self):
        """Setup hook points for TransformerLens compatibility."""
        # Embedding hooks
        self.hook_dict['embed'] = None

        # Block-level hooks
        for i in range(self.config.n_layers):
            layer_prefix = f'blocks.{i}'

            # Residual stream hooks
            self.hook_dict[f'{layer_prefix}.hook_resid_pre'] = None
            self.hook_dict[f'{layer_prefix}.hook_resid_mid'] = None
            self.hook_dict[f'{layer_prefix}.hook_resid_post'] = None

            # Attention hooks
            self.hook_dict[f'{layer_prefix}.attn.hook_q'] = None
            self.hook_dict[f'{layer_prefix}.attn.hook_k'] = None
            self.hook_dict[f'{layer_prefix}.attn.hook_v'] = None
            self.hook_dict[f'{layer_prefix}.attn.hook_attn_scores'] = None
            self.hook_dict[f'{layer_prefix}.attn.hook_pattern'] = None
            self.hook_dict[f'{layer_prefix}.attn.hook_z'] = None

            # MLP hooks
            self.hook_dict[f'{layer_prefix}.mlp.hook_pre'] = None
            self.hook_dict[f'{layer_prefix}.mlp.hook_mid'] = None
            self.hook_dict[f'{layer_prefix}.mlp.hook_post'] = None

        # Final hooks
        self.hook_dict['ln_final.hook_normalized'] = None
        if self.use_language_head:
            self.hook_dict['lm_head.hook_logits'] = None

    def add_hook(self, hook_name: str, hook_fn):
        """Add a hook function to the specified hook point."""
        if hook_name in self.hook_dict:
            self.hook_dict[hook_name] = hook_fn
        else:
            raise ValueError(f"Unknown hook point: {hook_name}")

    def remove_hook(self, hook_name: str):
        """Remove a hook function from the specified hook point."""
        if hook_name in self.hook_dict:
            self.hook_dict[hook_name] = None
        else:
            raise ValueError(f"Unknown hook point: {hook_name}")

    def clear_hooks(self):
        """Clear all hooks."""
        for hook_name in self.hook_dict:
            self.hook_dict[hook_name] = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        return_logits: bool = True,
        return_debug_info: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Forward pass through the transformer model.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Optional attention mask
            use_cache: Whether to cache activations
            return_logits: Whether to return logits (vs final hidden states)
            return_debug_info: Whether to return debug information

        Returns:
            output: Model output (logits or hidden states)
            debug_info: Optional debug information
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if use_cache:
            self.cache.clear()

        debug_info = {} if return_debug_info else None

        # === EMBEDDING LAYER ===
        x, embed_debug = self.embed(input_ids)

        # Apply embedding hook
        if self.hook_dict['embed'] is not None:
            x = self.hook_dict['embed'](x)

        if use_cache:
            self.cache['embed'] = x.clone()

        if return_debug_info:
            debug_info['embedding'] = embed_debug

        # === TRANSFORMER BLOCKS ===
        layer_outputs = []
        attention_weights_all = []

        for i, block in enumerate(self.blocks):
            layer_prefix = f'blocks.{i}'

            # Setup block hooks
            self._setup_block_hooks(block, layer_prefix)

            # Forward through block
            if return_debug_info:
                x, attn_weights, layer_debug = block(
                    x, attention_mask=attention_mask,
                    return_attention_weights=True,
                    return_debug_info=True
                )
                debug_info[f'layer_{i}'] = layer_debug
            else:
                x, attn_weights = block(
                    x, attention_mask=attention_mask,
                    return_attention_weights=True
                )

            layer_outputs.append(x.clone())
            attention_weights_all.append(attn_weights)

            if use_cache:
                self.cache[f'layer_{i}_output'] = x.clone()
                self.cache[f'layer_{i}_attn_weights'] = attn_weights.clone()

        # === FINAL LAYER NORMALIZATION ===
        x = self.ln_f(x)

        # Apply final layer norm hook
        if self.hook_dict['ln_final.hook_normalized'] is not None:
            x = self.hook_dict['ln_final.hook_normalized'](x)

        if use_cache:
            self.cache['ln_final'] = x.clone()

        # === LANGUAGE MODELING HEAD ===
        if return_logits and self.use_language_head:
            if self.tie_embeddings:
                # Use tied embeddings
                logits = F.linear(x, self.embed.token_embedding.embedding.weight)
            else:
                logits = self.lm_head(x)

            # Apply logits hook
            if self.hook_dict['lm_head.hook_logits'] is not None:
                logits = self.hook_dict['lm_head.hook_logits'](logits)

            if use_cache:
                self.cache['logits'] = logits.clone()

            output = logits
        else:
            output = x

        if return_debug_info:
            debug_info['layer_outputs'] = layer_outputs
            debug_info['attention_weights'] = attention_weights_all
            debug_info['final_hidden_states'] = x
            if return_logits and self.use_language_head:
                debug_info['logits'] = output

            return output, debug_info
        else:
            return output

    def _setup_block_hooks(self, block: TransformerBlock, layer_prefix: str):
        """Setup hooks for a specific transformer block."""
        # Residual stream hooks
        block.hook_residual_pre = self.hook_dict[f'{layer_prefix}.hook_resid_pre']
        block.hook_residual_mid = self.hook_dict[f'{layer_prefix}.hook_resid_mid']
        block.hook_residual_post = self.hook_dict[f'{layer_prefix}.hook_resid_post']

        # Attention hooks
        block.attention.hook_q = self.hook_dict[f'{layer_prefix}.attn.hook_q']
        block.attention.hook_k = self.hook_dict[f'{layer_prefix}.attn.hook_k']
        block.attention.hook_v = self.hook_dict[f'{layer_prefix}.attn.hook_v']
        block.attention.hook_attn_scores = self.hook_dict[f'{layer_prefix}.attn.hook_attn_scores']
        block.attention.hook_attn_weights = self.hook_dict[f'{layer_prefix}.attn.hook_pattern']
        block.attention.hook_attn_out = self.hook_dict[f'{layer_prefix}.attn.hook_z']

        # MLP hooks
        block.mlp.hook_pre = self.hook_dict[f'{layer_prefix}.mlp.hook_pre']
        block.mlp.hook_mid = self.hook_dict[f'{layer_prefix}.mlp.hook_mid']
        block.mlp.hook_post = self.hook_dict[f'{layer_prefix}.mlp.hook_post']

    def run_with_cache(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_logits: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Run model with activation caching (TransformerLens style).

        This method is designed to be compatible with TransformerLens workflows
        for mechanistic interpretability research.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Optional attention mask
            return_logits: Whether to return logits vs hidden states

        Returns:
            output: Model output (logits or hidden states)
            cache: Dictionary of cached activations
        """
        output = self.forward(
            input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_logits=return_logits
        )

        return output, self.cache.copy()

    def analyze_model_behavior(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of model behavior across all layers.

        Args:
            input_ids: Token IDs for analysis
            attention_mask: Optional attention mask
            token_labels: Optional token labels for visualization

        Returns:
            Dictionary with comprehensive model analysis
        """
        self.eval()
        with torch.no_grad():
            # Forward pass with full debug info
            output, debug_info = self.forward(
                input_ids,
                attention_mask=attention_mask,
                return_debug_info=True
            )

            analysis = {
                'model_config': self.config.to_dict(),
                'input_info': {
                    'input_shape': list(input_ids.shape),
                    'vocab_size': self.config.vocab_size,
                    'token_labels': token_labels
                },
                'layer_analysis': {},
                'global_analysis': {}
            }

            # Analyze each layer
            for i in range(self.config.n_layers):
                layer_debug = debug_info[f'layer_{i}']
                layer_analysis = {
                    'layer_index': i,
                    'residual_stream': layer_debug['residual_stream_analysis'],
                    'component_stats': {
                        'input': layer_debug['input_stats'],
                        'output': layer_debug['output_stats']
                    }
                }
                analysis['layer_analysis'][f'layer_{i}'] = layer_analysis

            # Global residual stream analysis
            layer_outputs = debug_info['layer_outputs']
            analysis['global_analysis'] = self._analyze_global_residual_stream(layer_outputs)

            # Attention pattern analysis across layers
            attention_weights = debug_info['attention_weights']
            analysis['global_analysis']['attention_evolution'] = self._analyze_attention_evolution(attention_weights)

            # Model capacity analysis
            analysis['global_analysis']['model_capacity'] = self._analyze_model_capacity()

            return analysis

    def _analyze_global_residual_stream(self, layer_outputs: List[torch.Tensor]) -> Dict[str, Any]:
        """Analyze how representations evolve through the residual stream."""
        analysis = {}

        # Compute layer-to-layer changes
        layer_changes = []
        layer_similarities = []

        for i in range(1, len(layer_outputs)):
            prev_output = layer_outputs[i-1]
            curr_output = layer_outputs[i]

            # Change magnitude
            change = torch.norm(curr_output - prev_output, dim=-1).mean().item()
            layer_changes.append(change)

            # Similarity
            prev_flat = prev_output.view(-1)
            curr_flat = curr_output.view(-1)
            similarity = F.cosine_similarity(prev_flat, curr_flat, dim=0).item()
            layer_similarities.append(similarity)

        analysis['layer_evolution'] = {
            'layer_changes': layer_changes,
            'layer_similarities': layer_similarities,
            'total_change': sum(layer_changes),
            'explanation': "How much each layer changes the representation"
        }

        # Analyze representation dimensionality evolution
        effective_ranks = []
        for output in layer_outputs:
            # Compute effective rank
            output_2d = output.view(-1, output.shape[-1])
            try:
                U, S, V = torch.svd(output_2d)
                total_var = S.sum()
                cumsum_var = torch.cumsum(S, dim=0) / total_var
                effective_rank = (cumsum_var < 0.99).sum().item() + 1
                effective_ranks.append(min(effective_rank, output.shape[-1]))
            except:
                effective_ranks.append(output.shape[-1])

        analysis['dimensionality_evolution'] = {
            'effective_ranks': effective_ranks,
            'rank_changes': [effective_ranks[i] - effective_ranks[i-1] for i in range(1, len(effective_ranks))],
            'explanation': "How the effective dimensionality of representations changes"
        }

        return analysis

    def _analyze_attention_evolution(self, attention_weights_all: List[torch.Tensor]) -> Dict[str, Any]:
        """Analyze how attention patterns evolve across layers."""
        analysis = {}

        # Compute attention pattern similarities between layers
        layer_attention_similarities = []

        for i in range(1, len(attention_weights_all)):
            prev_attn = attention_weights_all[i-1].mean(dim=1)  # Average across heads
            curr_attn = attention_weights_all[i].mean(dim=1)

            # Flatten and compute similarity
            prev_flat = prev_attn.view(-1)
            curr_flat = curr_attn.view(-1)
            similarity = F.cosine_similarity(prev_flat, curr_flat, dim=0).item()
            layer_attention_similarities.append(similarity)

        analysis['layer_attention_similarities'] = layer_attention_similarities

        # Analyze attention concentration evolution
        attention_entropies = []
        for attn_weights in attention_weights_all:
            # Compute entropy for each layer
            log_attn = torch.log(attn_weights + 1e-12)
            entropy = -(attn_weights * log_attn).sum(dim=-1).mean().item()
            attention_entropies.append(entropy)

        analysis['attention_entropy_evolution'] = {
            'entropies': attention_entropies,
            'entropy_changes': [attention_entropies[i] - attention_entropies[i-1] for i in range(1, len(attention_entropies))],
            'explanation': "How attention concentration changes across layers"
        }

        # Analyze head specialization across layers
        head_diversities = []
        for attn_weights in attention_weights_all:
            batch_size, n_heads, seq_len, _ = attn_weights.shape
            if n_heads > 1:
                # Compute pairwise correlations between heads
                correlations = []
                attn_flat = attn_weights.view(batch_size, n_heads, -1)
                for h1 in range(n_heads):
                    for h2 in range(h1 + 1, n_heads):
                        corr = F.cosine_similarity(
                            attn_flat[:, h1, :], attn_flat[:, h2, :], dim=-1
                        ).mean().item()
                        correlations.append(corr)
                head_diversities.append(np.mean(correlations))
            else:
                head_diversities.append(0.0)

        analysis['head_specialization_evolution'] = {
            'head_diversities': head_diversities,
            'explanation': "How head specialization evolves across layers (lower correlation = more diverse)"
        }

        return analysis

    def _analyze_model_capacity(self) -> Dict[str, Any]:
        """Analyze overall model capacity and parameter distribution."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Parameter distribution analysis
        component_params = {
            'embeddings': sum(p.numel() for p in self.embed.parameters()),
            'transformer_blocks': sum(p.numel() for p in self.blocks.parameters()),
            'final_ln': sum(p.numel() for p in self.ln_f.parameters()),
        }

        if self.use_language_head and not self.tie_embeddings and self.lm_head is not None:
            component_params['lm_head'] = sum(p.numel() for p in self.lm_head.parameters())

        # Per-layer analysis
        layer_params = []
        for i, block in enumerate(self.blocks):
            layer_params.append({
                'layer_index': i,
                'total_params': sum(p.numel() for p in block.parameters()),
                'attention_params': sum(p.numel() for p in block.attention.parameters()),
                'mlp_params': sum(p.numel() for p in block.mlp.parameters()),
                'layernorm_params': sum(p.numel() for p in block.ln1.parameters()) +
                                  sum(p.numel() for p in block.ln2.parameters())
            })

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'component_distribution': component_params,
            'layer_parameters': layer_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'parameters_per_layer': total_params // self.config.n_layers if self.config.n_layers > 0 else 0
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information for educational purposes."""
        return {
            'architecture': 'Educational Transformer (ARENA Chapter 1.1)',
            'config': self.config.to_dict(),
            'model_capacity': self._analyze_model_capacity(),
            'hook_points': list(self.hook_dict.keys()),
            'components': {
                'embedding_type': type(self.embed).__name__,
                'num_layers': len(self.blocks),
                'attention_type': type(self.blocks[0].attention).__name__ if self.blocks else None,
                'mlp_type': type(self.blocks[0].mlp).__name__ if self.blocks else None,
                'layer_norm_type': type(self.ln_f).__name__
            },
            'features': {
                'transformer_lens_compatible': True,
                'educational_analysis': True,
                'hook_system': True,
                'activation_caching': True,
                'tied_embeddings': self.tie_embeddings,
                'language_modeling_head': self.use_language_head
            }
        }

    def save_model(self, path: str, include_config: bool = True):
        """Save model state with optional configuration."""
        save_dict = {
            'model_state_dict': self.state_dict(),
            'model_info': self.get_model_info()
        }

        if include_config:
            save_dict['config'] = self.config.to_dict()

        torch.save(save_dict, path)

    @classmethod
    def load_model(cls, path: str, config: Optional[TransformerConfig] = None):
        """Load model from saved state."""
        checkpoint = torch.load(path, map_location='cpu')

        if config is None:
            if 'config' in checkpoint:
                config = TransformerConfig.from_dict(checkpoint['config'])
            else:
                raise ValueError("No config provided and none found in checkpoint")

        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])

        return model


def create_educational_transformer_examples() -> Dict[str, Any]:
    """
    Create educational examples for the complete transformer model.

    Returns:
        Dictionary with examples and explanations
    """
    examples = {}

    print("Creating complete transformer model examples...")

    # Example 1: Basic model functionality
    config = TransformerConfig.educational_config()
    model = EducationalTransformer(config)

    # Create sample input
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Forward pass
    logits = model(input_ids)

    examples['basic_functionality'] = {
        'config': config.to_dict(),
        'input_shape': input_ids.shape,
        'output_shape': logits.shape,
        'model_info': model.get_model_info(),
        'explanation': "Basic transformer model forward pass"
    }

    # Example 2: TransformerLens-style caching
    output_cached, cache = model.run_with_cache(input_ids)

    examples['activation_caching'] = {
        'cached_activations': list(cache.keys()),
        'cache_sizes': {k: list(v.shape) for k, v in cache.items()},
        'output_matches': torch.allclose(logits, output_cached),
        'explanation': "TransformerLens-style activation caching"
    }

    # Example 3: Comprehensive model analysis
    token_labels = [f"token_{i}" for i in range(seq_len)]
    analysis = model.analyze_model_behavior(input_ids, token_labels=token_labels)

    examples['model_analysis'] = {
        'analysis': analysis,
        'explanation': "Comprehensive analysis of model behavior across all layers"
    }

    # Example 4: Hook system demonstration
    activations_collected = {}

    def collect_activation(name):
        def hook_fn(activation):
            activations_collected[name] = activation.clone()
            return activation
        return hook_fn

    # Add hooks to collect specific activations
    model.add_hook('embed', collect_activation('embed'))
    model.add_hook('blocks.0.hook_resid_post', collect_activation('layer_0_output'))
    model.add_hook('ln_final.hook_normalized', collect_activation('final_hidden'))

    # Run model with hooks
    output_with_hooks = model(input_ids)

    examples['hook_system'] = {
        'collected_activations': list(activations_collected.keys()),
        'activation_shapes': {k: list(v.shape) for k, v in activations_collected.items()},
        'output_unchanged': torch.allclose(logits, output_with_hooks),
        'explanation': "Hook system for activation collection and intervention"
    }

    # Clear hooks
    model.clear_hooks()

    return examples


if __name__ == "__main__":
    print("Complete Educational Transformer Model")
    print("=" * 50)

    # Run educational examples
    examples = create_educational_transformer_examples()

    print("\n1. Basic model functionality:")
    basic = examples['basic_functionality']
    print(f"Input shape: {basic['input_shape']}")
    print(f"Output shape: {basic['output_shape']}")
    print(f"Model parameters: {basic['model_info']['model_capacity']['total_parameters']:,}")
    print(f"Model size: {basic['model_info']['model_capacity']['model_size_mb']:.2f} MB")

    print("\n2. Model configuration:")
    config_info = basic['config']
    print(f"d_model: {config_info['d_model']}")
    print(f"n_layers: {config_info['n_layers']}")
    print(f"n_heads: {config_info['n_heads']}")
    print(f"vocab_size: {config_info['vocab_size']}")

    print("\n3. TransformerLens-style caching:")
    caching = examples['activation_caching']
    print(f"Cached activations: {len(caching['cached_activations'])}")
    print(f"Key activations: {caching['cached_activations'][:5]}...")
    print(f"Output matches non-cached: {caching['output_matches']}")

    print("\n4. Model analysis highlights:")
    analysis = examples['model_analysis']['analysis']

    # Global analysis
    global_analysis = analysis['global_analysis']
    if 'layer_evolution' in global_analysis:
        evolution = global_analysis['layer_evolution']
        print(f"Total representation change: {evolution['total_change']:.4f}")
        print(f"Layer similarities: {[f'{sim:.3f}' for sim in evolution['layer_similarities'][:3]]}...")

    if 'attention_evolution' in global_analysis:
        attn_evolution = global_analysis['attention_evolution']
        if 'attention_entropy_evolution' in attn_evolution:
            entropies = attn_evolution['attention_entropy_evolution']['entropies']
            print(f"Attention entropies across layers: {[f'{ent:.2f}' for ent in entropies]}")

    print("\n5. Hook system demonstration:")
    hooks = examples['hook_system']
    print(f"Collected activations: {hooks['collected_activations']}")
    print(f"Output unchanged by hooks: {hooks['output_unchanged']}")

    for name, shape in hooks['activation_shapes'].items():
        print(f"  {name}: {shape}")

    print("\n6. Parameter distribution:")
    capacity = examples['basic_functionality']['model_info']['model_capacity']
    components = capacity['component_distribution']

    total_params = capacity['total_parameters']
    for component, params in components.items():
        percentage = (params / total_params) * 100
        print(f"  {component}: {params:,} ({percentage:.1f}%)")

    print("\n7. Model features:")
    features = examples['basic_functionality']['model_info']['features']
    for feature, enabled in features.items():
        status = "✓" if enabled else "✗"
        print(f"  {status} {feature.replace('_', ' ').title()}")

    print("\n8. Testing different configurations:")
    test_configs = [
        TransformerConfig.small_config(),
        TransformerConfig.educational_config(),
    ]

    for i, config in enumerate(test_configs):
        test_model = EducationalTransformer(config)
        test_input = torch.randint(0, 100, (1, 10))
        test_output = test_model(test_input)

        params = sum(p.numel() for p in test_model.parameters())
        print(f"✓ Config {i+1}: {params:,} parameters")
        print(f"  d_model={config.d_model}, n_layers={config.n_layers}, n_heads={config.n_heads}")
        print(f"  Input: {test_input.shape} -> Output: {test_output.shape}")

    print("\n✅ Complete transformer model implemented and tested!")