# Neuroscience-Inspired AI Safety Experiment

## 🧠 Overview

This repository implements a framework for **reverse-engineering neural circuits of misalignment** in AI systems, inspired by neuroscience techniques. The experiment analyzes how neural networks develop deceptive behaviors and identifies specific circuits responsible for these behaviors.

## 🎯 Experiment Framework

The framework follows a 5-step neuroscience-inspired methodology:

1. **Choose a Model**: Toy transformer architecture for controlled experimentation
2. **Define Target Behavior**: Deceptive vs. truthful response patterns  
3. **Analyze Activations**: Compare neural activation patterns between behaviors
4. **Perform Lesion Studies**: Ablate specific neurons to test causal relationships
5. **Visualize & Document**: Generate plots and research memo

## 🔬 Key Components

### ToyTransformer
- Simplified transformer model with activation tracking
- Configurable architecture (layers, dimensions, attention heads)
- Built-in hooks for capturing intermediate activations

### DeceptionDataset
- Generates controlled scenarios for truthful vs. deceptive behavior
- Simple token-based patterns for reliable analysis
- Scalable to more complex behavioral definitions

### NeuralCircuitAnalyzer
- Main analysis engine for circuit identification
- Activation pattern comparison algorithms
- Targeted lesion study implementation
- Automated visualization and reporting

## 🚀 Usage

### Full Experiment
```bash
python3 neural_circuit_analysis.py
```

### Quick Demo
```bash
python3 demo_usage.py
```

## 📊 Results

The experiment generates:

- **Activation difference heatmaps**: Visual comparison of neural patterns
- **Important neuron distribution**: Count of significant neurons per layer  
- **Lesion study results**: Effect of targeted ablations on behavior
- **Research memo**: Automated summary of findings and implications

### Sample Findings

From our toy experiment:
- **787 important neurons** identified across 9 layers
- **Differential lesion effects** on truthful vs. deceptive behavior
- **Layer-specific patterns** suggesting hierarchical processing of deception

## 🔧 Technical Details

### Model Architecture
- 3-layer transformer with 128-dimensional embeddings
- 4 attention heads per layer
- 858,216 total parameters

### Analysis Parameters
- Activation difference threshold: 0.5
- Sample sizes: 50 (analysis), 20 (lesion study)
- Visualization: Multi-panel plots with statistical summaries

## 🛡️ AI Safety Implications

This framework enables:

1. **Mechanistic Understanding**: Identify specific circuits causing misaligned behavior
2. **Early Detection**: Monitor activation patterns as warning signals
3. **Targeted Intervention**: Develop precise correction techniques
4. **Safety Verification**: Test robustness of safety measures

## 📚 Future Directions

- Scale to larger, production models (GPT-style architectures)
- Investigate complex deceptive behaviors (adversarial examples, manipulation)
- Develop real-time monitoring systems
- Create intervention techniques based on circuit analysis

## 🔬 Scientific Basis

This work draws inspiration from:
- **Neuroscience lesion studies**: Classic techniques for understanding brain function
- **Mechanistic interpretability**: Recent advances in understanding neural networks
- **AI safety research**: Focus on identifying and preventing harmful behaviors

## 📁 File Structure

```
/workspace/
├── neural_circuit_analysis.py  # Main experiment framework
├── demo_usage.py              # Quick demonstration script  
├── README.md                  # This documentation
└── results/                   # Generated outputs
    ├── research_memo.md       # Automated research summary
    ├── activation_differences.png
    ├── important_neurons.png
    └── lesion_results.png
```

## 🎓 Educational Value

This experiment serves as:
- **Hands-on introduction** to mechanistic interpretability
- **Practical demonstration** of neuroscience-inspired AI safety
- **Template framework** for studying other AI behaviors
- **Bridge** between neuroscience and AI safety research

---

*30-minute implementation demonstrating the power of neuroscience-inspired approaches to AI safety.*

