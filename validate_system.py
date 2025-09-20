"""
Quick System Validation Script

This script validates that the educational transformer system works correctly
by running basic functionality tests.
"""

import sys
import os
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import using package imports
from src.models.config import TransformerConfig
from src.models.transformer import EducationalTransformer
from src.foundations.attention_math import AttentionMathematics
from src.components.mlp import GELU
from src.interpretability.hooks import HookManager


def validate_foundations():
    """Test mathematical foundations."""
    print("Testing Mathematical Foundations...")

    # Test GELU
    gelu = GELU()
    x = torch.tensor([-1.0, 0.0, 1.0])
    y = gelu(x)
    assert y.shape == x.shape
    print("  ✅ GELU activation works")

    # Test attention math
    query = torch.randn(1, 4, 8)  # [batch, seq_len, d_head]
    key = torch.randn(1, 4, 8)
    value = torch.randn(1, 4, 8)

    output, attn_weights, debug_info = AttentionMathematics.single_head_attention(query, key, value)
    assert output.shape == (1, 4, 8)
    assert attn_weights.shape == (1, 4, 4)
    print("  ✅ Attention mathematics works")


def validate_model():
    """Test complete model functionality."""
    print("\nTesting Complete Model...")

    config = TransformerConfig.small_config()
    model = EducationalTransformer(config)

    # Test forward pass
    input_ids = torch.randint(0, 100, (1, 5))

    model.eval()
    with torch.no_grad():
        logits = model(input_ids)
        assert logits.shape == (1, 5, config.vocab_size)
        print("  ✅ Basic forward pass works")

        # Test caching
        logits_cached, cache = model.run_with_cache(input_ids)
        assert torch.allclose(logits, logits_cached)
        assert len(cache) > 0
        print("  ✅ Activation caching works")

        # Test analysis
        analysis = model.analyze_model_behavior(input_ids)
        assert 'global_analysis' in analysis
        assert 'layer_analysis' in analysis
        print("  ✅ Model analysis works")


def validate_interpretability():
    """Test interpretability tools."""
    print("\nTesting Interpretability Tools...")

    config = TransformerConfig.small_config()
    model = EducationalTransformer(config)

    # Test hook manager
    hook_manager = HookManager(model)

    input_ids = torch.randint(0, 30, (1, 3))

    # Add caching hook
    hook_manager.add_caching_hook('embed')

    model.eval()
    with torch.no_grad():
        model(input_ids)

    cached_embed = hook_manager.get_cached_activation('embed')
    assert cached_embed is not None
    assert cached_embed.shape == (1, 3, config.d_model)
    print("  ✅ Hook system works")

    hook_manager.clear_hooks()


def validate_system():
    """Run complete system validation."""
    print("ARENA Chapter 1.1 Educational Transformer Validation")
    print("=" * 60)

    try:
        validate_foundations()
        validate_model()
        validate_interpretability()

        print("\nSystem Validation Complete!")
        print("\nAll core components working correctly:")
        print("  - Mathematical foundations implemented")
        print("  - Complete transformer model functional")
        print("  - TransformerLens-compatible caching system")
        print("  - Comprehensive analysis tools")
        print("  - Hook-based interpretability system")
        print("  - Educational debugging capabilities")

        print("\nSystem ready for ARENA Chapter 1.1 curriculum!")
        return True

    except Exception as e:
        print(f"\nValidation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = validate_system()
    exit(0 if success else 1)