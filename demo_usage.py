#!/usr/bin/env python3
"""
Demo: Quick Usage of Neural Circuit Analysis Framework

This demonstrates how to use the neuroscience-inspired AI safety framework
for analyzing deceptive behavior in neural networks.
"""

import sys
sys.path.append('/workspace')

from neural_circuit_analysis import ToyTransformer, NeuralCircuitAnalyzer

def quick_demo():
    """Quick demonstration of the framework"""
    
    print("🔬 Neural Circuit Analysis Demo")
    print("=" * 40)
    
    # 1. Create a smaller model for quick demo
    print("Creating toy model...")
    model = ToyTransformer(vocab_size=100, d_model=64, n_heads=2, n_layers=2)
    
    # 2. Initialize analyzer with matching vocab size
    analyzer = NeuralCircuitAnalyzer(model)
    analyzer.dataset.vocab_size = 100  # Match the model vocab size
    
    # 3. Quick analysis with fewer samples
    print("Analyzing activation patterns...")
    analyzer.analyze_activations(n_samples=10)
    
    # 4. Identify important neurons
    important_neurons = analyzer.identify_important_neurons(threshold=0.3)
    total_important = sum(len(neurons) for neurons in important_neurons.values())
    
    print(f"Found {total_important} important neurons")
    
    # 5. Quick lesion test
    print("Running lesion study...")
    lesion_results = analyzer.perform_lesion_study(n_samples=5)
    
    print(f"Tested lesions on {len(lesion_results)} layers")
    
    # 6. Show summary
    print("\n📊 Quick Analysis Summary:")
    for layer, neurons in important_neurons.items():
        if neurons:
            print(f"  • {layer}: {len(neurons)} important neurons")
    
    print("\n✨ Demo complete! For full analysis, run neural_circuit_analysis.py")

if __name__ == "__main__":
    quick_demo()