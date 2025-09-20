# ARENA Chapter 1.1: Transformer from Scratch

**Philosophy**: Educational Understanding - Build conceptual understanding through guided exercises

This project implements a complete Transformer model following the ARENA (Alignment Research Engineer Accelerator) Chapter 1.1 curriculum, designed for educational understanding and mechanistic interpretability research.

## Project Structure

```
LLM Builder/
├── src/
│   ├── foundations/             # Mathematical foundations
│   │   ├── linear_algebra.py    # Matrix operations, tensor reshaping
│   │   ├── attention_math.py    # QKV calculations, scaled dot-product
│   │   └── position_encoding.py # Positional information understanding
│   ├── components/              # Transformer components
│   │   ├── embeddings.py        # Token & position embeddings
│   │   ├── attention.py         # Multi-head attention mechanism
│   │   ├── mlp.py              # Feed-forward networks with GELU
│   │   └── transformer_block.py # Complete transformer block
│   ├── models/                 # Full model implementation
│   │   ├── transformer.py      # Main transformer model
│   │   └── config.py           # Model configuration
│   └── interpretability/       # Mechanistic interpretability tools
│       ├── hooks.py            # TransformerLens integration
│       └── residual_stream.py  # Residual stream analysis
├── notebooks/                  # Educational notebooks
│   └── 01_complete_transformer_demo.ipynb
├── tests/                      # Test suite
│   └── test_complete_system.py
├── examples/                   # Usage examples
│   └── basic_usage.py
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Learning Objectives

### Part 1: Mathematical Foundations
- Linear algebra operations: Matrix multiplications, tensor reshaping, broadcasting
- Attention mechanism math: QKV calculations, scaled dot-product attention
- Position encoding: Understanding why transformers need positional information
- Residual stream concept: How information flows through transformer layers

### Part 2: Component Implementation
- Token & Position Embeddings: Converting tokens to vectors, adding positional information
- Multi-Head Attention: Query, Key, Value projections, scaled dot-product attention, multiple heads
- MLP Blocks: Feed-forward networks, GELU activation, layer normalization
- Transformer Block: Residual connections, layer normalization, attention + MLP combination

### Part 3: Model Integration
- Full transformer assembly: Complete model architecture
- Parameter initialization: Proper weight initialization strategies
- Forward pass implementation: End-to-end model execution
- TransformerLens integration: Mechanistic interpretability capabilities

## Key Technologies

- PyTorch: Primary deep learning framework
- TransformerLens: Mechanistic interpretability library
- NumPy: Numerical computations
- Jupyter: Interactive development and education

## Educational Approach

- Step-by-step implementation: Each component built incrementally
- Mathematical intuition: Deep dive into the math behind each operation
- Visualization: Attention patterns, residual stream analysis
- Debugging tools: Understanding transformer internals
- Interpretability focus: Connect implementation to research

## Prerequisites

- Comfortable with Python and PyTorch basics
- Understanding of basic linear algebra (matrices, vectors)
- Familiarity with deep learning concepts
- Experience with vectorized operations (NumPy-style)

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the complete demo:
   ```bash
   python -m src.models.transformer
   ```

3. Basic usage:
   ```python
   from src.models.transformer import EducationalTransformer
   from src.models.config import TransformerConfig

   # Create and use model
   config = TransformerConfig.educational_config()
   model = EducationalTransformer(config)

   # Run with caching for analysis
   input_ids = torch.randint(0, 100, (1, 10))
   logits, cache = model.run_with_cache(input_ids)
   ```

## Learning Path

1. Mathematical Foundations - Understand the core math
2. Component Building - Implement each piece
3. Model Assembly - Put it all together
4. Interpretability - Understand what the model learned
5. Advanced Analysis - Dive deep into mechanistic interpretability

## Resources

- [ARENA 3.0 Official Repository](https://github.com/callummcdougall/ARENA_3.0)
- [TransformerLens Documentation](https://transformerlensorg.github.io/TransformerLens/)
- [Attention Is All You Need Paper](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

This implementation is designed for educational understanding and mechanistic interpretability research, following the ARENA curriculum philosophy of learning through guided implementation.