"""
Comprehensive Test Suite for Educational Transformer

This test suite validates the complete educational transformer implementation,
ensuring all components work correctly individually and together as a system.

Following ARENA Chapter 1.1 curriculum requirements.
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.config import TransformerConfig
from src.models.transformer import EducationalTransformer
from src.components.embeddings import TokenEmbedding, CombinedEmbedding
from src.components.attention import MultiHeadAttention
from src.components.mlp import TransformerMLP, LayerNorm, GELU
from src.components.transformer_block import TransformerBlock
from src.foundations.linear_algebra import (
    scaled_dot_product_math,
    tensor_reshaping_for_multihead,
    tensor_reshaping_from_multihead,
    create_causal_mask
)
from src.foundations.attention_math import AttentionMathematics
from src.foundations.position_encoding import SinusoidalPositionEncoding
from src.interpretability.hooks import HookManager, ActivationPatcher, AblationAnalyzer


class TestFoundations:
    """Test the mathematical foundations."""

    def test_scaled_dot_product_attention(self):
        """Test scaled dot-product attention mathematics."""
        batch_size, n_heads, seq_len, d_head = 2, 4, 8, 16

        query = torch.randn(batch_size, n_heads, seq_len, d_head)
        key = torch.randn(batch_size, n_heads, seq_len, d_head)
        value = torch.randn(batch_size, n_heads, seq_len, d_head)

        output, attention_weights = scaled_dot_product_math(query, key, value)

        # Check output shape
        assert output.shape == (batch_size, n_heads, seq_len, d_head)
        assert attention_weights.shape == (batch_size, n_heads, seq_len, seq_len)

        # Check attention weights sum to 1
        assert torch.allclose(attention_weights.sum(dim=-1), torch.ones(batch_size, n_heads, seq_len))

        # Check attention weights are non-negative
        assert torch.all(attention_weights >= 0)

    def test_tensor_reshaping(self):
        """Test tensor reshaping for multi-head attention."""
        batch_size, seq_len, d_model = 2, 10, 64
        n_heads, d_head = 8, 8

        tensor = torch.randn(batch_size, seq_len, d_model)

        # Reshape for multi-head
        reshaped = tensor_reshaping_for_multihead(tensor, n_heads, d_head)
        assert reshaped.shape == (batch_size, n_heads, seq_len, d_head)

        # Reshape back
        original = tensor_reshaping_from_multihead(reshaped)
        assert original.shape == tensor.shape
        assert torch.allclose(original, tensor)

    def test_causal_mask(self):
        """Test causal mask creation."""
        seq_len = 5
        mask = create_causal_mask(seq_len, torch.device('cpu'))

        assert mask.shape == (seq_len, seq_len)

        # Check lower triangular structure
        for i in range(seq_len):
            for j in range(seq_len):
                if j > i:
                    assert mask[i, j] == 0
                else:
                    assert mask[i, j] == 1

    def test_position_encoding(self):
        """Test sinusoidal position encoding."""
        d_model = 64
        max_len = 100

        pos_encoder = SinusoidalPositionEncoding(d_model, max_len)

        # Test basic properties
        encoding = pos_encoder(10)
        assert encoding.shape == (10, d_model)

        # Test uniqueness - different positions should have different encodings
        pos_0 = encoding[0]
        pos_1 = encoding[1]
        assert not torch.allclose(pos_0, pos_1)


class TestComponents:
    """Test individual transformer components."""

    def test_token_embedding(self):
        """Test token embedding layer."""
        vocab_size, d_model = 1000, 256
        embedding = TokenEmbedding(vocab_size, d_model)

        batch_size, seq_len = 2, 10
        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        output = embedding(token_ids)
        assert output.shape == (batch_size, seq_len, d_model)

        # Test scaling - embeddings should be scaled by sqrt(d_model)
        raw_embedding = embedding.embedding(token_ids)
        expected_output = raw_embedding * (d_model ** 0.5)
        assert torch.allclose(output, expected_output)

    def test_combined_embedding(self):
        """Test combined token and position embeddings."""
        vocab_size, d_model, max_seq_len = 100, 64, 50

        embedding = CombinedEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
            position_encoding_type="sinusoidal",
            dropout=0.0  # No dropout for testing
        )

        batch_size, seq_len = 2, 8
        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        output, debug_info = embedding(token_ids)

        assert output.shape == (batch_size, seq_len, d_model)
        assert 'token_embeddings' in debug_info
        assert 'position_embeddings' in debug_info
        assert debug_info['token_embeddings'].shape == (batch_size, seq_len, d_model)
        assert debug_info['position_embeddings'].shape == (batch_size, seq_len, d_model)

    def test_gelu_activation(self):
        """Test GELU activation function."""
        gelu = GELU(approximation="tanh")

        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        y = gelu(x)

        # GELU should be smooth and approximately linear around 0
        assert y.shape == x.shape

        # GELU(0) should be 0
        assert torch.allclose(gelu(torch.tensor([0.0])), torch.tensor([0.0]), atol=1e-6)

        # GELU should be approximately x for large positive x
        large_x = torch.tensor([10.0])
        large_y = gelu(large_x)
        assert torch.allclose(large_y, large_x, rtol=0.1)

    def test_layer_norm(self):
        """Test layer normalization."""
        d_model = 64
        layer_norm = LayerNorm(d_model)

        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, d_model) * 5  # Large variance

        output = layer_norm(x)

        assert output.shape == x.shape

        # Check normalization properties (approximately)
        output_mean = output.mean(dim=-1)
        output_std = output.std(dim=-1)

        # Mean should be close to learned bias (initially 0)
        assert torch.allclose(output_mean, torch.zeros_like(output_mean), atol=1e-6)

        # Std should be close to learned scale (initially 1)
        assert torch.allclose(output_std, torch.ones_like(output_std), atol=1e-2)

    def test_mlp_block(self):
        """Test MLP block."""
        d_model, d_mlp = 128, 512

        mlp = TransformerMLP(
            d_model=d_model,
            d_mlp=d_mlp,
            dropout=0.0,  # No dropout for testing
            activation="gelu"
        )

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, d_model)

        output = mlp(x)
        assert output.shape == x.shape

        # Test with debug info
        output_debug, debug_info = mlp(x, return_debug_info=True)
        assert torch.allclose(output, output_debug)
        assert 'statistics' in debug_info
        assert debug_info['configuration']['d_model'] == d_model
        assert debug_info['configuration']['d_mlp'] == d_mlp

    def test_multi_head_attention(self):
        """Test multi-head attention mechanism."""
        d_model, n_heads = 128, 8

        attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=0.0,  # No dropout for testing
            is_causal=True
        )

        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, d_model)

        # Test basic forward pass
        output = attention(x, return_attention_weights=False)
        assert output.shape == x.shape

        # Test with attention weights
        output_with_attn, attn_weights = attention(x, return_attention_weights=True)
        assert torch.allclose(output, output_with_attn)
        assert attn_weights.shape == (batch_size, n_heads, seq_len, seq_len)

        # Test causal attention - upper triangular should be zero (after softmax, very small)
        # Check that attention to future positions is minimal
        for head in range(n_heads):
            for i in range(seq_len):
                for j in range(i + 1, seq_len):
                    assert attn_weights[0, head, i, j] < 1e-6

    def test_transformer_block(self):
        """Test complete transformer block."""
        config = TransformerConfig.small_config()
        block = TransformerBlock(config, layer_idx=0)

        batch_size, seq_len = 1, 8
        x = torch.randn(batch_size, seq_len, config.d_model)

        # Set to eval mode for deterministic behavior
        block.eval()

        # Test basic forward pass
        with torch.no_grad():
            output = block(x)
            assert output.shape == x.shape

            # Test with attention weights and debug info
            output_debug, attn_weights, debug_info = block(
                x, return_attention_weights=True, return_debug_info=True
            )

            assert torch.allclose(output, output_debug, atol=1e-5)
        assert attn_weights.shape == (batch_size, config.n_heads, seq_len, seq_len)
        assert 'residual_stream_analysis' in debug_info


class TestFullModel:
    """Test the complete transformer model."""

    def test_model_creation(self):
        """Test model creation with different configurations."""
        configs = [
            TransformerConfig.small_config(),
            TransformerConfig.educational_config(),
        ]

        for config in configs:
            model = EducationalTransformer(config)

            # Check model structure
            assert len(model.blocks) == config.n_layers
            assert model.config.d_model == config.d_model
            assert model.config.n_heads == config.n_heads

            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            assert total_params > 0

    def test_forward_pass(self):
        """Test model forward pass."""
        config = TransformerConfig.small_config()
        model = EducationalTransformer(config)

        batch_size, seq_len = 2, 6
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        model.eval()
        with torch.no_grad():
            # Basic forward pass
            logits = model(input_ids)
            assert logits.shape == (batch_size, seq_len, config.vocab_size)

            # Forward pass with debug info
            logits_debug, debug_info = model(input_ids, return_debug_info=True)
            assert torch.allclose(logits, logits_debug)

            # Check debug info structure
            assert 'embedding' in debug_info
            assert 'layer_outputs' in debug_info
            assert 'attention_weights' in debug_info
            assert len(debug_info['layer_outputs']) == config.n_layers

    def test_activation_caching(self):
        """Test TransformerLens-style activation caching."""
        config = TransformerConfig.small_config()
        model = EducationalTransformer(config)

        input_ids = torch.randint(0, 100, (1, 5))

        model.eval()
        with torch.no_grad():
            # Regular forward pass
            logits_regular = model(input_ids)

            # Cached forward pass
            logits_cached, cache = model.run_with_cache(input_ids)

            # Results should be identical
            assert torch.allclose(logits_regular, logits_cached)

            # Cache should contain expected keys
            expected_keys = ['embed', 'layer_0_output', 'ln_final', 'logits']
            for key in expected_keys:
                assert key in cache

    def test_hook_system(self):
        """Test the hook system functionality."""
        config = TransformerConfig.small_config()
        model = EducationalTransformer(config)

        input_ids = torch.randint(0, 50, (1, 4))

        # Test hook registration and execution
        collected_activations = {}

        def collect_hook(name):
            def hook_fn(activation):
                collected_activations[name] = activation.clone()
                return activation
            return hook_fn

        # Add hooks
        model.add_hook('embed', collect_hook('embed'))
        model.add_hook('blocks.0.hook_resid_post', collect_hook('layer_0'))

        model.eval()
        with torch.no_grad():
            output = model(input_ids)

        # Check that hooks were executed
        assert 'embed' in collected_activations
        assert 'layer_0' in collected_activations

        # Clear hooks
        model.clear_hooks()

    def test_model_analysis(self):
        """Test comprehensive model analysis."""
        config = TransformerConfig.small_config()
        model = EducationalTransformer(config)

        input_ids = torch.randint(0, 30, (1, 6))

        model.eval()
        analysis = model.analyze_model_behavior(input_ids)

        # Check analysis structure
        assert 'model_config' in analysis
        assert 'layer_analysis' in analysis
        assert 'global_analysis' in analysis

        # Check layer analysis
        for i in range(config.n_layers):
            layer_key = f'layer_{i}'
            assert layer_key in analysis['layer_analysis']

        # Check global analysis
        global_analysis = analysis['global_analysis']
        assert 'model_capacity' in global_analysis


class TestInterpretabilityTools:
    """Test interpretability and intervention tools."""

    def test_hook_manager(self):
        """Test hook manager functionality."""
        config = TransformerConfig.small_config()
        model = EducationalTransformer(config)
        hook_manager = HookManager(model)

        input_ids = torch.randint(0, 30, (1, 4))

        # Test caching hook
        hook_manager.add_caching_hook('embed')

        model.eval()
        with torch.no_grad():
            model(input_ids)

        # Check that activation was cached
        cached_embed = hook_manager.get_cached_activation('embed')
        assert cached_embed is not None
        assert cached_embed.shape == (1, 4, config.d_model)

        hook_manager.clear_hooks()

    def test_activation_patcher(self):
        """Test activation patching functionality."""
        config = TransformerConfig.small_config()
        model = EducationalTransformer(config)
        patcher = ActivationPatcher(model)

        # Create clean and corrupted inputs
        clean_input = torch.tensor([[1, 2, 3, 4]])
        corrupted_input = torch.tensor([[1, 2, 3, 99]])  # Last token different

        # Simple metric function
        def metric_fn(logits):
            return logits[0, -1, 4].item()  # Logit for token 4

        model.eval()
        result = patcher.patch_activation(
            clean_input,
            corrupted_input,
            'blocks.0.hook_resid_post',
            metric_fn
        )

        # Check result structure
        assert 'clean_metric' in result
        assert 'corrupted_metric' in result
        assert 'patched_metric' in result
        assert 'recovery_ratio' in result
        assert 'analysis' in result

    def test_ablation_analyzer(self):
        """Test ablation analysis functionality."""
        config = TransformerConfig.small_config()
        model = EducationalTransformer(config)
        ablator = AblationAnalyzer(model)

        input_ids = torch.randint(0, 20, (1, 3))

        def metric_fn(logits):
            return torch.norm(logits).item()

        model.eval()
        result = ablator.zero_ablation(
            input_ids,
            'blocks.0.attn.hook_z',
            metric_fn
        )

        # Check result structure
        assert 'baseline_metric' in result
        assert 'ablated_metric' in result
        assert 'importance_score' in result
        assert 'interpretation' in result


class TestEndToEndFunctionality:
    """Test end-to-end functionality and integration."""

    def test_educational_workflow(self):
        """Test a complete educational workflow."""
        # Create model
        config = TransformerConfig.educational_config()
        model = EducationalTransformer(config)

        # Sample educational sequence
        seq_len = 8
        input_ids = torch.arange(seq_len).unsqueeze(0)  # [0, 1, 2, ..., 7]

        model.eval()

        # Step 1: Basic forward pass
        with torch.no_grad():
            logits = model(input_ids)
            assert logits.shape == (1, seq_len, config.vocab_size)

        # Step 2: Activation caching
        with torch.no_grad():
            logits_cached, cache = model.run_with_cache(input_ids)
            assert torch.allclose(logits, logits_cached)
            assert len(cache) > 0

        # Step 3: Comprehensive analysis
        analysis = model.analyze_model_behavior(input_ids)
        assert 'global_analysis' in analysis
        assert 'layer_analysis' in analysis

        # Step 4: Hook intervention
        hook_manager = HookManager(model)

        def scale_intervention(activation):
            return activation * 0.8

        hook_manager.add_intervention_hook(
            'blocks.0.mlp.hook_post',
            scale_intervention,
            "Scale MLP output"
        )

        with torch.no_grad():
            modified_logits = model(input_ids)

        # Intervention should change output
        assert not torch.allclose(logits, modified_logits)

        hook_manager.clear_hooks()

    def test_model_serialization(self):
        """Test model saving and loading."""
        config = TransformerConfig.small_config()
        model1 = EducationalTransformer(config)

        # Test input
        input_ids = torch.randint(0, 50, (1, 5))

        model1.eval()
        with torch.no_grad():
            output1 = model1(input_ids)

        # Save model
        save_path = "/tmp/test_model.pt"
        model1.save_model(save_path)

        # Load model
        model2 = EducationalTransformer.load_model(save_path)
        model2.eval()

        with torch.no_grad():
            output2 = model2(input_ids)

        # Outputs should be identical
        assert torch.allclose(output1, output2)

        # Clean up
        os.remove(save_path)

    def test_numerical_stability(self):
        """Test numerical stability under various conditions."""
        config = TransformerConfig.small_config()
        model = EducationalTransformer(config)

        model.eval()

        # Test with various input conditions
        test_cases = [
            torch.randint(0, 10, (1, 3)),  # Small vocab
            torch.randint(0, config.vocab_size, (1, config.max_position_embeddings // 2)),  # Long sequence
            torch.zeros(1, 4, dtype=torch.long),  # All zeros
            torch.full((1, 4), config.vocab_size - 1, dtype=torch.long),  # High token IDs
        ]

        for input_ids in test_cases:
            with torch.no_grad():
                try:
                    output = model(input_ids)
                    # Check for NaN or inf
                    assert torch.isfinite(output).all(), f"Non-finite output for input {input_ids}"
                    # Check reasonable magnitude
                    assert output.abs().max() < 1000, f"Extremely large output for input {input_ids}"
                except Exception as e:
                    pytest.fail(f"Model failed on input {input_ids}: {e}")


def run_all_tests():
    """Run all tests and report results."""
    print("Running Comprehensive Test Suite")
    print("=" * 50)

    test_classes = [
        TestFoundations,
        TestComponents,
        TestFullModel,
        TestInterpretabilityTools,
        TestEndToEndFunctionality
    ]

    total_tests = 0
    passed_tests = 0

    for test_class in test_classes:
        print(f"\nTesting {test_class.__name__}")
        test_instance = test_class()

        # Get all test methods
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]

        for test_method in test_methods:
            total_tests += 1
            try:
                getattr(test_instance, test_method)()
                print(f"  ✅ {test_method}")
                passed_tests += 1
            except Exception as e:
                print(f"  X {test_method}: {e}")

    print(f"\nTest Results: {passed_tests}/{total_tests} passed")

    if passed_tests == total_tests:
        print("All tests passed! System is ready for use.")
        return True
    else:
        print(f"{total_tests - passed_tests} tests failed. Please review implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)