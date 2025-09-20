"""
Hook System and Intervention Tools

Educational implementation of hooks and intervention tools for mechanistic
interpretability. This module provides comprehensive tools for activation
caching, patching, and intervention experiments.

Key concepts covered:
- Hook-based activation caching
- Activation patching for causal analysis
- Ablation studies
- Circuit analysis utilities
- Educational intervention demonstrations

Following ARENA Chapter 1.1 curriculum and TransformerLens patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, Union, List, Callable
import numpy as np
from collections import defaultdict
import copy

try:
    from ..models.transformer import EducationalTransformer
    from ..models.config import TransformerConfig
except ImportError:
    from src.models.transformer import EducationalTransformer
    from src.models.config import TransformerConfig


class HookManager:
    """
    Educational hook manager for comprehensive activation analysis and intervention.

    This class provides a unified interface for managing hooks across the transformer
    model, enabling sophisticated mechanistic interpretability experiments.
    """

    def __init__(self, model: EducationalTransformer):
        """
        Initialize hook manager.

        Args:
            model: Educational transformer model to manage hooks for
        """
        self.model = model
        self.active_hooks = {}
        self.cached_activations = {}
        self.intervention_history = []

    def add_caching_hook(self, hook_name: str, cache_key: Optional[str] = None):
        """
        Add a hook that caches activations.

        Args:
            hook_name: Name of hook point
            cache_key: Key to store activation under (defaults to hook_name)
        """
        if cache_key is None:
            cache_key = hook_name

        def caching_hook(activation):
            self.cached_activations[cache_key] = activation.clone().detach()
            return activation

        self.add_hook(hook_name, caching_hook)

    def add_intervention_hook(
        self,
        hook_name: str,
        intervention_fn: Callable[[torch.Tensor], torch.Tensor],
        description: str = ""
    ):
        """
        Add a hook that performs intervention on activations.

        Args:
            hook_name: Name of hook point
            intervention_fn: Function to apply to activations
            description: Description of the intervention
        """
        def intervention_hook(activation):
            original_activation = activation.clone()
            modified_activation = intervention_fn(activation)

            # Record intervention
            self.intervention_history.append({
                'hook_name': hook_name,
                'description': description,
                'original_norm': torch.norm(original_activation).item(),
                'modified_norm': torch.norm(modified_activation).item(),
                'change_norm': torch.norm(modified_activation - original_activation).item()
            })

            return modified_activation

        self.add_hook(hook_name, intervention_hook)

    def add_hook(self, hook_name: str, hook_fn: Callable):
        """Add a hook function to the model."""
        self.active_hooks[hook_name] = hook_fn
        self.model.add_hook(hook_name, hook_fn)

    def remove_hook(self, hook_name: str):
        """Remove a specific hook."""
        if hook_name in self.active_hooks:
            del self.active_hooks[hook_name]
            self.model.remove_hook(hook_name)

    def clear_hooks(self):
        """Clear all hooks."""
        self.active_hooks.clear()
        self.cached_activations.clear()
        self.intervention_history.clear()
        self.model.clear_hooks()

    def get_cached_activation(self, key: str) -> Optional[torch.Tensor]:
        """Get a cached activation."""
        return self.cached_activations.get(key)

    def get_intervention_summary(self) -> Dict[str, Any]:
        """Get summary of all interventions performed."""
        if not self.intervention_history:
            return {'total_interventions': 0}

        return {
            'total_interventions': len(self.intervention_history),
            'interventions': self.intervention_history,
            'average_change_norm': np.mean([h['change_norm'] for h in self.intervention_history]),
            'hooks_used': list(set(h['hook_name'] for h in self.intervention_history))
        }


class ActivationPatcher:
    """
    Educational implementation of activation patching for causal analysis.

    Activation patching is a key technique in mechanistic interpretability that
    allows us to test causal hypotheses about model computations by replacing
    activations from one context with those from another.
    """

    def __init__(self, model: EducationalTransformer):
        """
        Initialize activation patcher.

        Args:
            model: Educational transformer model
        """
        self.model = model
        self.hook_manager = HookManager(model)

    def patch_activation(
        self,
        clean_input: torch.Tensor,
        corrupted_input: torch.Tensor,
        patch_hook: str,
        metric_fn: Callable[[torch.Tensor], float],
        patch_positions: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Perform activation patching experiment.

        This method runs the classic activation patching experiment:
        1. Run clean input and cache activations
        2. Run corrupted input and patch in clean activations
        3. Measure the effect on the output metric

        Args:
            clean_input: Input that produces desired behavior
            corrupted_input: Input that produces undesired behavior
            patch_hook: Hook point to patch
            metric_fn: Function to compute metric from model output
            patch_positions: Specific positions to patch (None = all positions)

        Returns:
            Dictionary with patching results and analysis
        """
        results = {
            'clean_input': clean_input,
            'corrupted_input': corrupted_input,
            'patch_hook': patch_hook,
            'patch_positions': patch_positions
        }

        # Step 1: Get baseline metrics
        with torch.no_grad():
            clean_output = self.model(clean_input)
            clean_metric = metric_fn(clean_output)

            corrupted_output = self.model(corrupted_input)
            corrupted_metric = metric_fn(corrupted_output)

        results['clean_metric'] = clean_metric
        results['corrupted_metric'] = corrupted_metric
        results['baseline_difference'] = clean_metric - corrupted_metric

        # Step 2: Cache clean activation
        clean_activation = None

        def cache_clean_activation(activation):
            nonlocal clean_activation
            clean_activation = activation.clone()
            return activation

        self.hook_manager.add_hook(patch_hook, cache_clean_activation)

        # Run clean input to cache activation
        with torch.no_grad():
            _ = self.model(clean_input)

        self.hook_manager.remove_hook(patch_hook)

        # Step 3: Patch in clean activation during corrupted run
        def patch_intervention(activation):
            if clean_activation is None:
                return activation

            if patch_positions is None:
                # Patch all positions
                return clean_activation.clone()
            else:
                # Patch specific positions
                patched_activation = activation.clone()
                for pos in patch_positions:
                    if pos < activation.shape[1]:  # seq_len dimension
                        patched_activation[:, pos, :] = clean_activation[:, pos, :]
                return patched_activation

        self.hook_manager.add_hook(patch_hook, patch_intervention)

        # Run corrupted input with patching
        with torch.no_grad():
            patched_output = self.model(corrupted_input)
            patched_metric = metric_fn(patched_output)

        self.hook_manager.remove_hook(patch_hook)

        # Step 4: Analyze results
        results['patched_metric'] = patched_metric
        results['patch_effect'] = patched_metric - corrupted_metric
        results['recovery_ratio'] = (patched_metric - corrupted_metric) / (clean_metric - corrupted_metric) if (clean_metric - corrupted_metric) != 0 else 0

        results['analysis'] = {
            'effect_size': abs(results['patch_effect']),
            'effect_direction': 'positive' if results['patch_effect'] > 0 else 'negative',
            'recovery_percentage': results['recovery_ratio'] * 100,
            'interpretation': self._interpret_patch_result(results['recovery_ratio'])
        }

        return results

    def _interpret_patch_result(self, recovery_ratio: float) -> str:
        """Interpret the patching result based on recovery ratio."""
        if recovery_ratio > 0.8:
            return "Strong causal effect - this component is likely critical for the behavior"
        elif recovery_ratio > 0.5:
            return "Moderate causal effect - this component contributes to the behavior"
        elif recovery_ratio > 0.2:
            return "Weak causal effect - this component has some influence"
        elif recovery_ratio > -0.2:
            return "Minimal effect - this component doesn't significantly affect the behavior"
        else:
            return "Negative effect - patching this component makes the behavior worse"

    def batch_patch_experiment(
        self,
        clean_input: torch.Tensor,
        corrupted_input: torch.Tensor,
        hook_names: List[str],
        metric_fn: Callable[[torch.Tensor], float]
    ) -> Dict[str, Any]:
        """
        Perform patching experiments across multiple hook points.

        Args:
            clean_input: Clean input
            corrupted_input: Corrupted input
            hook_names: List of hook points to test
            metric_fn: Metric function

        Returns:
            Dictionary with results for all hook points
        """
        results = {}

        for hook_name in hook_names:
            try:
                result = self.patch_activation(
                    clean_input, corrupted_input, hook_name, metric_fn
                )
                results[hook_name] = result
            except Exception as e:
                results[hook_name] = {'error': str(e)}

        # Add summary analysis
        valid_results = {k: v for k, v in results.items() if 'error' not in v}

        if valid_results:
            recovery_ratios = [r['recovery_ratio'] for r in valid_results.values()]
            results['summary'] = {
                'num_hooks_tested': len(hook_names),
                'num_successful': len(valid_results),
                'mean_recovery_ratio': np.mean(recovery_ratios),
                'std_recovery_ratio': np.std(recovery_ratios),
                'best_hook': max(valid_results.keys(), key=lambda k: valid_results[k]['recovery_ratio']),
                'worst_hook': min(valid_results.keys(), key=lambda k: valid_results[k]['recovery_ratio'])
            }

        return results


class AblationAnalyzer:
    """
    Educational implementation of ablation analysis for understanding component importance.

    Ablation studies systematically remove or modify components to understand
    their importance for model behavior.
    """

    def __init__(self, model: EducationalTransformer):
        """
        Initialize ablation analyzer.

        Args:
            model: Educational transformer model
        """
        self.model = model
        self.hook_manager = HookManager(model)

    def zero_ablation(
        self,
        input_ids: torch.Tensor,
        hook_name: str,
        metric_fn: Callable[[torch.Tensor], float],
        positions: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Perform zero ablation (set activations to zero).

        Args:
            input_ids: Input token IDs
            hook_name: Hook point to ablate
            metric_fn: Metric function
            positions: Specific positions to ablate (None = all)

        Returns:
            Ablation results and analysis
        """
        # Baseline metric
        with torch.no_grad():
            baseline_output = self.model(input_ids)
            baseline_metric = metric_fn(baseline_output)

        # Define ablation intervention
        def zero_intervention(activation):
            ablated = activation.clone()
            if positions is None:
                ablated.zero_()
            else:
                for pos in positions:
                    if pos < activation.shape[1]:
                        ablated[:, pos, :] = 0
            return ablated

        # Apply ablation
        self.hook_manager.add_intervention_hook(
            hook_name, zero_intervention, f"Zero ablation at {hook_name}"
        )

        with torch.no_grad():
            ablated_output = self.model(input_ids)
            ablated_metric = metric_fn(ablated_output)

        self.hook_manager.clear_hooks()

        # Analyze results
        effect = baseline_metric - ablated_metric
        relative_effect = effect / baseline_metric if baseline_metric != 0 else 0

        return {
            'hook_name': hook_name,
            'positions': positions,
            'baseline_metric': baseline_metric,
            'ablated_metric': ablated_metric,
            'absolute_effect': effect,
            'relative_effect': relative_effect,
            'importance_score': abs(relative_effect),
            'interpretation': self._interpret_ablation_result(relative_effect)
        }

    def mean_ablation(
        self,
        input_ids: torch.Tensor,
        hook_name: str,
        metric_fn: Callable[[torch.Tensor], float],
        dataset_mean: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Perform mean ablation (replace with mean activation).

        Args:
            input_ids: Input token IDs
            hook_name: Hook point to ablate
            metric_fn: Metric function
            dataset_mean: Pre-computed dataset mean (computed if None)

        Returns:
            Ablation results and analysis
        """
        # Get baseline
        with torch.no_grad():
            baseline_output = self.model(input_ids)
            baseline_metric = metric_fn(baseline_output)

        # If no dataset mean provided, use activation mean
        if dataset_mean is None:
            activation_for_mean = None

            def cache_for_mean(activation):
                nonlocal activation_for_mean
                activation_for_mean = activation.clone()
                return activation

            self.hook_manager.add_hook(hook_name, cache_for_mean)
            with torch.no_grad():
                _ = self.model(input_ids)
            self.hook_manager.clear_hooks()

            dataset_mean = activation_for_mean.mean(dim=(0, 1), keepdim=True)

        # Define mean ablation intervention
        def mean_intervention(activation):
            return dataset_mean.expand_as(activation)

        # Apply ablation
        self.hook_manager.add_intervention_hook(
            hook_name, mean_intervention, f"Mean ablation at {hook_name}"
        )

        with torch.no_grad():
            ablated_output = self.model(input_ids)
            ablated_metric = metric_fn(ablated_output)

        self.hook_manager.clear_hooks()

        # Analyze results
        effect = baseline_metric - ablated_metric
        relative_effect = effect / baseline_metric if baseline_metric != 0 else 0

        return {
            'hook_name': hook_name,
            'baseline_metric': baseline_metric,
            'ablated_metric': ablated_metric,
            'absolute_effect': effect,
            'relative_effect': relative_effect,
            'importance_score': abs(relative_effect),
            'dataset_mean_shape': list(dataset_mean.shape),
            'interpretation': self._interpret_ablation_result(relative_effect)
        }

    def _interpret_ablation_result(self, relative_effect: float) -> str:
        """Interpret ablation results based on relative effect size."""
        abs_effect = abs(relative_effect)

        if abs_effect > 0.5:
            return "Critical component - large impact on behavior"
        elif abs_effect > 0.2:
            return "Important component - moderate impact"
        elif abs_effect > 0.05:
            return "Minor component - small but measurable impact"
        else:
            return "Negligible component - minimal impact on behavior"

    def systematic_ablation_study(
        self,
        input_ids: torch.Tensor,
        metric_fn: Callable[[torch.Tensor], float],
        component_types: List[str] = None
    ) -> Dict[str, Any]:
        """
        Perform systematic ablation across multiple components.

        Args:
            input_ids: Input token IDs
            metric_fn: Metric function
            component_types: Types of components to ablate

        Returns:
            Comprehensive ablation study results
        """
        if component_types is None:
            component_types = ['attention', 'mlp', 'residual']

        # Generate hook names for systematic study
        hook_names = []
        n_layers = self.model.config.n_layers

        for layer_idx in range(n_layers):
            if 'attention' in component_types:
                hook_names.append(f'blocks.{layer_idx}.attn.hook_z')
            if 'mlp' in component_types:
                hook_names.append(f'blocks.{layer_idx}.mlp.hook_post')
            if 'residual' in component_types:
                hook_names.append(f'blocks.{layer_idx}.hook_resid_post')

        # Perform ablations
        results = {}
        for hook_name in hook_names:
            try:
                result = self.zero_ablation(input_ids, hook_name, metric_fn)
                results[hook_name] = result
            except Exception as e:
                results[hook_name] = {'error': str(e)}

        # Analysis and ranking
        valid_results = {k: v for k, v in results.items() if 'error' not in v}

        if valid_results:
            # Rank by importance
            ranked_components = sorted(
                valid_results.items(),
                key=lambda x: x[1]['importance_score'],
                reverse=True
            )

            results['analysis'] = {
                'total_components': len(hook_names),
                'successful_ablations': len(valid_results),
                'most_important': ranked_components[0][0] if ranked_components else None,
                'least_important': ranked_components[-1][0] if ranked_components else None,
                'importance_scores': [r[1]['importance_score'] for r in ranked_components],
                'mean_importance': np.mean([r['importance_score'] for r in valid_results.values()]),
                'ranking': [(name, result['importance_score']) for name, result in ranked_components]
            }

        return results


def create_educational_interpretability_examples(model: EducationalTransformer) -> Dict[str, Any]:
    """
    Create educational examples demonstrating interpretability techniques.

    Args:
        model: Educational transformer model

    Returns:
        Dictionary with interpretability examples and explanations
    """
    examples = {}

    print("Creating interpretability examples...")

    # Example input for demonstrations
    batch_size, seq_len = 1, 12
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))

    # Example 1: Basic hook management
    hook_manager = HookManager(model)

    # Add caching hooks for different components
    hook_manager.add_caching_hook('embed')
    hook_manager.add_caching_hook('blocks.0.hook_resid_post', 'layer_0_output')
    hook_manager.add_caching_hook('ln_final.hook_normalized', 'final_hidden')

    # Run model to populate cache
    with torch.no_grad():
        output = model(input_ids)

    examples['hook_management'] = {
        'cached_activations': list(hook_manager.cached_activations.keys()),
        'activation_shapes': {k: list(v.shape) for k, v in hook_manager.cached_activations.items()},
        'explanation': "Basic hook system for caching activations"
    }

    hook_manager.clear_hooks()

    # Example 2: Simple intervention
    def scale_intervention(activation):
        return activation * 0.5  # Scale down activations

    hook_manager.add_intervention_hook(
        'blocks.0.attn.hook_z',
        scale_intervention,
        "Scale attention output by 0.5"
    )

    with torch.no_grad():
        intervention_output = model(input_ids)

    intervention_summary = hook_manager.get_intervention_summary()

    examples['simple_intervention'] = {
        'intervention_summary': intervention_summary,
        'output_change': torch.norm(output - intervention_output).item(),
        'explanation': "Simple intervention scaling attention output"
    }

    hook_manager.clear_hooks()

    # Example 3: Activation patching
    patcher = ActivationPatcher(model)

    # Create "clean" and "corrupted" inputs (for demonstration)
    clean_input = input_ids.clone()
    corrupted_input = input_ids.clone()
    corrupted_input[:, -1] = torch.randint(0, model.config.vocab_size, (1,))  # Change last token

    # Define simple metric (logit of last token at last position)
    def simple_metric(logits):
        return logits[0, -1, input_ids[0, -1]].item()

    # Perform patching experiment
    patch_result = patcher.patch_activation(
        clean_input,
        corrupted_input,
        'blocks.0.hook_resid_post',
        simple_metric
    )

    examples['activation_patching'] = {
        'patch_result': {
            'clean_metric': patch_result['clean_metric'],
            'corrupted_metric': patch_result['corrupted_metric'],
            'patched_metric': patch_result['patched_metric'],
            'recovery_ratio': patch_result['recovery_ratio'],
            'interpretation': patch_result['analysis']['interpretation']
        },
        'explanation': "Activation patching to test causal effects"
    }

    # Example 4: Ablation analysis
    ablator = AblationAnalyzer(model)

    # Perform zero ablation on attention
    ablation_result = ablator.zero_ablation(
        input_ids,
        'blocks.0.attn.hook_z',
        simple_metric
    )

    examples['ablation_analysis'] = {
        'ablation_result': {
            'baseline_metric': ablation_result['baseline_metric'],
            'ablated_metric': ablation_result['ablated_metric'],
            'relative_effect': ablation_result['relative_effect'],
            'importance_score': ablation_result['importance_score'],
            'interpretation': ablation_result['interpretation']
        },
        'explanation': "Zero ablation to measure component importance"
    }

    # Example 5: Systematic ablation study (small scale for demo)
    systematic_results = ablator.systematic_ablation_study(
        input_ids,
        simple_metric,
        component_types=['attention', 'mlp']
    )

    if 'analysis' in systematic_results:
        examples['systematic_ablation'] = {
            'analysis': systematic_results['analysis'],
            'top_3_components': systematic_results['analysis']['ranking'][:3],
            'explanation': "Systematic ablation study across multiple components"
        }

    return examples


if __name__ == "__main__":
    print("Interpretability Tools and Hook System")
    print("=" * 50)

    # Create a small model for testing
    config = TransformerConfig.small_config()
    model = EducationalTransformer(config)

    # Run examples
    examples = create_educational_interpretability_examples(model)

    print("\n1. Hook management example:")
    hook_mgmt = examples['hook_management']
    print(f"Cached activations: {hook_mgmt['cached_activations']}")
    for name, shape in hook_mgmt['activation_shapes'].items():
        print(f"  {name}: {shape}")

    print("\n2. Simple intervention example:")
    intervention = examples['simple_intervention']
    print(f"Interventions performed: {intervention['intervention_summary']['total_interventions']}")
    print(f"Output change magnitude: {intervention['output_change']:.6f}")

    print("\n3. Activation patching example:")
    patching = examples['activation_patching']['patch_result']
    print(f"Clean metric: {patching['clean_metric']:.4f}")
    print(f"Corrupted metric: {patching['corrupted_metric']:.4f}")
    print(f"Patched metric: {patching['patched_metric']:.4f}")
    print(f"Recovery ratio: {patching['recovery_ratio']:.4f}")
    print(f"Interpretation: {patching['interpretation']}")

    print("\n4. Ablation analysis example:")
    ablation = examples['ablation_analysis']['ablation_result']
    print(f"Baseline metric: {ablation['baseline_metric']:.4f}")
    print(f"Ablated metric: {ablation['ablated_metric']:.4f}")
    print(f"Relative effect: {ablation['relative_effect']:.4f}")
    print(f"Importance score: {ablation['importance_score']:.4f}")
    print(f"Interpretation: {ablation['interpretation']}")

    if 'systematic_ablation' in examples:
        print("\n5. Systematic ablation study:")
        systematic = examples['systematic_ablation']
        print(f"Components tested: {systematic['analysis']['total_components']}")
        print(f"Successful ablations: {systematic['analysis']['successful_ablations']}")
        print(f"Most important: {systematic['analysis']['most_important']}")
        print(f"Mean importance: {systematic['analysis']['mean_importance']:.4f}")

        print("\nTop 3 most important components:")
        for i, (component, score) in enumerate(systematic['top_3_components'], 1):
            print(f"  {i}. {component}: {score:.4f}")

    print("\n6. Educational insights:")
    print("✓ Hooks enable non-invasive activation monitoring")
    print("✓ Interventions test causal hypotheses")
    print("✓ Activation patching reveals component importance")
    print("✓ Ablation studies quantify component contributions")
    print("✓ Systematic studies identify critical circuits")

    print("\n✅ Interpretability tools implemented and demonstrated!")