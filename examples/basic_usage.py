"""
Basic Usage Examples for Educational Transformer

This module provides comprehensive examples demonstrating how to use the
educational transformer implementation for both learning and research.
These examples showcase the key features and educational insights available.

Following ARENA Chapter 1.1 curriculum.
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.config import TransformerConfig
from src.models.transformer import EducationalTransformer
from src.interpretability.hooks import HookManager, ActivationPatcher, AblationAnalyzer


def basic_model_usage():
    """Demonstrate basic model creation and usage."""
    print("Basic Model Usage")
    print("=" * 50)

    # Create model with educational configuration
    config = TransformerConfig.educational_config()
    print(f"Configuration: {config.d_model}D model, {config.n_layers} layers, {config.n_heads} heads")

    model = EducationalTransformer(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create sample input
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))  # Use subset of vocab for demo
    print(f"Input shape: {input_ids.shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)
        print(f"Output logits shape: {logits.shape}")

        # Get probabilities
        probs = F.softmax(logits, dim=-1)
        print(f"Output probabilities shape: {probs.shape}")

        # Most likely next token for each position
        top_tokens = torch.argmax(probs, dim=-1)
        print(f"Top predicted tokens: {top_tokens[0].tolist()[:8]}...")

    return model, input_ids


def demonstrate_activation_caching():
    """Demonstrate TransformerLens-style activation caching."""
    print("\nActivation Caching (TransformerLens Style)")
    print("=" * 50)

    config = TransformerConfig.small_config()  # Smaller for faster demo
    model = EducationalTransformer(config)

    # Sample input
    input_ids = torch.randint(0, 100, (1, 8))
    print(f"Input: {input_ids.tolist()}")

    # Run with caching
    model.eval()
    with torch.no_grad():
        logits, cache = model.run_with_cache(input_ids)

    print(f"Cached {len(cache)} activations:")
    for key, activation in cache.items():
        print(f"  {key}: {list(activation.shape)}")

    # Analyze cached activations
    print("\nActivation Analysis:")
    if 'embed' in cache:
        embed_norm = torch.norm(cache['embed'], dim=-1).mean()
        print(f"  Embedding norm: {embed_norm:.4f}")

    if 'layer_0_output' in cache:
        layer_0_norm = torch.norm(cache['layer_0_output'], dim=-1).mean()
        print(f"  Layer 0 output norm: {layer_0_norm:.4f}")

    if 'ln_final' in cache:
        final_norm = torch.norm(cache['ln_final'], dim=-1).mean()
        print(f"  Final hidden state norm: {final_norm:.4f}")

    return model, input_ids, cache


def demonstrate_attention_analysis():
    """Demonstrate attention pattern analysis."""
    print("\nAttention Pattern Analysis")
    print("=" * 50)

    config = TransformerConfig.educational_config()
    model = EducationalTransformer(config)

    # Create interpretable input sequence
    seq_len = 10
    input_ids = torch.arange(seq_len).unsqueeze(0)  # [0, 1, 2, ..., 9]
    print(f"Sequential input: {input_ids.tolist()[0]}")

    # Forward pass with attention analysis
    model.eval()
    with torch.no_grad():
        logits, debug_info = model(input_ids, return_debug_info=True)

    # Analyze attention patterns
    attention_weights = debug_info['attention_weights']
    print(f"Attention weights for {len(attention_weights)} layers")

    for layer_idx, attn_weights in enumerate(attention_weights):
        if layer_idx >= 2:  # Show first 2 layers only
            break

        print(f"\nLayer {layer_idx} attention patterns:")
        batch_size, n_heads, seq_len, _ = attn_weights.shape

        # Average across batch and heads for visualization
        avg_attention = attn_weights.mean(dim=(0, 1))

        # Show attention pattern for first few positions
        for pos in range(min(3, seq_len)):
            attention_dist = avg_attention[pos]
            top_3_attended = torch.topk(attention_dist, min(3, seq_len))

            print(f"  Position {pos} attends most to positions: {top_3_attended.indices.tolist()}")
            print(f"    with weights: {top_3_attended.values.tolist()}")

    return attention_weights


def demonstrate_residual_stream_analysis():
    """Demonstrate residual stream analysis."""
    print("\nResidual Stream Analysis")
    print("=" * 50)

    config = TransformerConfig.small_config()
    model = EducationalTransformer(config)

    input_ids = torch.randint(0, 100, (1, 6))

    # Get comprehensive model analysis
    model.eval()
    analysis = model.analyze_model_behavior(input_ids)

    # Global residual stream analysis
    global_analysis = analysis['global_analysis']

    if 'layer_evolution' in global_analysis:
        evolution = global_analysis['layer_evolution']
        print("Layer-by-layer representation changes:")
        for i, change in enumerate(evolution['layer_changes']):
            print(f"  Layer {i} -> {i+1}: {change:.4f} change magnitude")

        print(f"Total change through model: {evolution['total_change']:.4f}")

    # Component analysis per layer
    print("\nPer-layer component analysis:")
    for layer_key, layer_info in analysis['layer_analysis'].items():
        if 'layer_0' not in layer_key:  # Show only first layer for brevity
            continue

        residual_analysis = layer_info['residual_stream']
        component_mag = residual_analysis['component_magnitudes']

        print(f"{layer_key}:")
        print(f"  Attention contribution: {component_mag['relative_attention_contribution']:.3f}")
        print(f"  MLP contribution: {component_mag['relative_mlp_contribution']:.3f}")

    return analysis


def demonstrate_interpretability_tools():
    """Demonstrate interpretability tools and interventions."""
    print("\nInterpretability Tools")
    print("=" * 50)

    config = TransformerConfig.small_config()
    model = EducationalTransformer(config)

    # Create hook manager
    hook_manager = HookManager(model)

    # Example 1: Simple activation intervention
    print("1. Simple activation scaling intervention:")

    input_ids = torch.randint(0, 50, (1, 5))

    # Baseline forward pass
    model.eval()
    with torch.no_grad():
        baseline_logits = model(input_ids)
        baseline_probs = F.softmax(baseline_logits[0, -1], dim=-1)
        baseline_top_token = torch.argmax(baseline_probs).item()

    print(f"Baseline top token: {baseline_top_token}")

    # Add scaling intervention
    def scale_mlp_output(activation):
        return activation * 0.5  # Scale down MLP output

    hook_manager.add_intervention_hook(
        'blocks.0.mlp.hook_post',
        scale_mlp_output,
        "Scale MLP output by 0.5"
    )

    # Forward pass with intervention
    with torch.no_grad():
        intervention_logits = model(input_ids)
        intervention_probs = F.softmax(intervention_logits[0, -1], dim=-1)
        intervention_top_token = torch.argmax(intervention_probs).item()

    print(f"After MLP scaling: {intervention_top_token}")
    print(f"Token changed: {baseline_top_token != intervention_top_token}")

    hook_manager.clear_hooks()

    # Example 2: Activation patching
    print("\n2. Activation patching experiment:")

    patcher = ActivationPatcher(model)

    # Create clean and corrupted inputs
    clean_input = torch.tensor([[1, 2, 3, 4, 5]])
    corrupted_input = torch.tensor([[1, 2, 3, 4, 99]])  # Last token corrupted

    # Define metric (logit of clean last token)
    def clean_token_metric(logits):
        return logits[0, -1, 5].item()  # Logit for token 5

    # Perform patching
    patch_result = patcher.patch_activation(
        clean_input,
        corrupted_input,
        'blocks.0.hook_resid_post',
        clean_token_metric
    )

    print(f"Clean metric: {patch_result['clean_metric']:.4f}")
    print(f"Corrupted metric: {patch_result['corrupted_metric']:.4f}")
    print(f"Patched metric: {patch_result['patched_metric']:.4f}")
    print(f"Recovery ratio: {patch_result['recovery_ratio']:.4f}")
    print(f"Interpretation: {patch_result['analysis']['interpretation']}")

    # Example 3: Ablation study
    print("\n3. Component ablation study:")

    ablator = AblationAnalyzer(model)

    # Test importance of different components
    test_input = torch.randint(0, 30, (1, 4))

    def output_norm_metric(logits):
        return torch.norm(logits).item()

    components_to_test = [
        'blocks.0.attn.hook_z',
        'blocks.0.mlp.hook_post',
        'blocks.1.attn.hook_z' if config.n_layers > 1 else None
    ]

    for component in components_to_test:
        if component is None:
            continue

        result = ablator.zero_ablation(test_input, component, output_norm_metric)
        print(f"{component}:")
        print(f"  Importance score: {result['importance_score']:.4f}")
        print(f"  Interpretation: {result['interpretation']}")

    return hook_manager, patcher, ablator


def demonstrate_educational_insights():
    """Demonstrate key educational insights about transformers."""
    print("\nEducational Insights")
    print("=" * 50)

    config = TransformerConfig.educational_config()
    model = EducationalTransformer(config)

    # Insight 1: Effect of position encoding
    print("1. Position encoding effects:")

    # Create repeated token sequence
    repeated_tokens = torch.tensor([[5, 3, 5, 3, 5, 3]])

    model.eval()
    with torch.no_grad():
        output, debug_info = model(repeated_tokens, return_debug_info=True)

    embed_debug = debug_info['embedding']
    token_embeddings = embed_debug['token_embeddings']
    position_embeddings = embed_debug['position_embeddings']
    combined = embed_debug['combined_before_dropout']

    # Show that identical tokens become different due to position encoding
    print(f"Token 5 at position 0: {token_embeddings[0, 0, :3].tolist()}")
    print(f"Token 5 at position 2: {token_embeddings[0, 2, :3].tolist()}")
    print(f"Tokens are identical: {torch.allclose(token_embeddings[0, 0], token_embeddings[0, 2])}")

    print(f"Combined at position 0: {combined[0, 0, :3].tolist()}")
    print(f"Combined at position 2: {combined[0, 2, :3].tolist()}")
    print(f"Combined are different: {not torch.allclose(combined[0, 0], combined[0, 2])}")

    # Insight 2: Residual stream information flow
    print("\n2. Residual stream information flow:")

    layer_outputs = debug_info['layer_outputs']

    # Compute how much information is preserved vs transformed
    input_hidden = debug_info['embedding']['combined_after_dropout']

    for i, layer_output in enumerate(layer_outputs[:2]):  # First 2 layers
        # Similarity with input
        input_flat = input_hidden.view(-1)
        output_flat = layer_output.view(-1)
        similarity = F.cosine_similarity(input_flat, output_flat, dim=0).item()

        # Change magnitude
        change_norm = torch.norm(layer_output - input_hidden).item()
        input_norm = torch.norm(input_hidden).item()
        relative_change = change_norm / input_norm

        print(f"Layer {i}:")
        print(f"  Similarity to input: {similarity:.4f}")
        print(f"  Relative change: {relative_change:.4f}")

    # Insight 3: Attention pattern interpretations
    print("\n3. Attention pattern insights:")

    attention_weights = debug_info['attention_weights'][0]  # First layer
    batch_size, n_heads, seq_len, _ = attention_weights.shape

    # Analyze different types of attention patterns
    for head in range(min(2, n_heads)):
        head_attention = attention_weights[0, head]  # [seq_len, seq_len]

        # Check for diagonal pattern (self-attention)
        diagonal_strength = torch.diagonal(head_attention).mean().item()

        # Check for local pattern (attending to nearby positions)
        local_mask = torch.abs(torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)) <= 1
        local_strength = (head_attention * local_mask.float()).sum().item() / local_mask.sum().item()

        print(f"Head {head}:")
        print(f"  Self-attention strength: {diagonal_strength:.4f}")
        print(f"  Local attention strength: {local_strength:.4f}")

        if diagonal_strength > 0.3:
            print(f"  → Primarily self-attending head")
        elif local_strength > 0.5:
            print(f"  → Local attention head")
        else:
            print(f"  → Global/mixed attention head")

    return model


def run_all_examples():
    """Run all basic usage examples."""
    print("Educational Transformer - Complete Examples")
    print("=" * 60)

    # Run all demonstrations
    try:
        model, input_ids = basic_model_usage()
        print("✅ Basic usage demonstrated")

        model_cached, input_cached, cache = demonstrate_activation_caching()
        print("✅ Activation caching demonstrated")

        attention_weights = demonstrate_attention_analysis()
        print("✅ Attention analysis demonstrated")

        analysis = demonstrate_residual_stream_analysis()
        print("✅ Residual stream analysis demonstrated")

        hook_manager, patcher, ablator = demonstrate_interpretability_tools()
        print("✅ Interpretability tools demonstrated")

        model_insights = demonstrate_educational_insights()
        print("✅ Educational insights demonstrated")

        print("\nAll examples completed successfully!")
        print("\nKey takeaways:")
        print("- Transformers use position encoding to distinguish token positions")
        print("- Residual stream allows information to flow while enabling modifications")
        print("- Attention patterns vary across heads and serve different functions")
        print("- Components can be analyzed and intervened upon for interpretability")
        print("- Hook system enables deep inspection of model internals")

        return True

    except Exception as e:
        print(f"Error in examples: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_examples()
    if success:
        print("\nReady for ARENA Chapter 1.1 curriculum!")
    else:
        print("\nSome examples failed - check implementation")