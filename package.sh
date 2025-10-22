#!/bin/bash

# Create directories for models and visualizations
mkdir -p models/cartoon
mkdir -p models/ghibli
mkdir -p models/sketch
mkdir -p visualizations/architecture
mkdir -p visualizations/training
mkdir -p visualizations/evaluation
mkdir -p sample_images

# Generate sample model architecture diagrams
python3 visualize_performance.py

# Create a README file
echo "# Enhanced Image Transformation System

This project enhances the classical Digital Image Processing (DIP) cartoonization approach with Artificial Neural Network (ANN) techniques to create a more sophisticated image transformation system with multiple style options.

## Features

- Multiple transformation styles:
  - Classical cartoonization
  - Neural network-based cartoonization
  - Ghibli-style animation
  - Sketch rendering

- ANN concepts implemented:
  - Convolutional Neural Networks (CNN)
  - Recurrent Neural Networks (RNN)
  - Long Short-Term Memory (LSTM)
  - Generative Adversarial Networks (GAN)
  - Various activation functions (ReLU, LeakyReLU, Tanh, Sigmoid)
  - Different weight initializers (He, Xavier/Glorot, LeCun)
  - Multiple optimizers (SGD, Adam, RMSprop)

- Interactive GUI with:
  - Real-time parameter adjustment
  - Style selection
  - Visualization tools
  - Batch processing

## Installation

\`\`\`bash
# Install required packages
pip install tensorflow torch torchvision matplotlib scikit-learn opencv-python pillow pandas seaborn
\`\`\`

## Usage

1. Run the GUI application:
   \`\`\`bash
   python enhanced_gui.py
   \`\`\`

2. Load an image using the 'Open Image' button
3. Select a transformation style
4. Adjust parameters as desired
5. Save the transformed image

## Files

- \`enhanced_transformer.py\`: Main transformation framework
- \`neural_style_models.py\`: Neural network model implementations
- \`style_processor.py\`: Unified interface for applying styles
- \`model_trainer.py\`: Training and evaluation framework
- \`enhanced_gui.py\`: Graphical user interface
- \`visualize_performance.py\`: Visualization utilities
- \`ieee_paper.tex\`: IEEE-style conference paper

## IEEE Paper

The included IEEE-style paper documents the research methodology, implementation details, and experimental results. To compile the paper:

\`\`\`bash
pdflatex ieee_paper.tex
bibtex ieee_paper
pdflatex ieee_paper.tex
pdflatex ieee_paper.tex
\`\`\`

## Authors

- Aditya Sharma
- Priya Patel

" > README.md

# Create a presentation file
echo "# Enhanced Image Transformation Using Artificial Neural Networks

## Overview
- Integration of classical DIP with modern ANN techniques
- Multiple transformation styles: cartoon, Ghibli, sketch
- Comprehensive evaluation and comparison

## Limitations of Classical DIP
- Edge detection limitations (noise sensitivity, lack of semantic understanding)
- Color quantization limitations (simplistic reduction, lack of style-specific coloring)
- Bilateral filtering limitations (computational intensity, limited texture handling)
- Overall artistic style limitations (generic look, no learning capability)

## ANN-Based Enhancements
- CNN-based cartoonization with U-Net architecture
- GAN-based Ghibli-style transformation
- RNN-LSTM-based sketch generation
- Integration with classical techniques

## Neural Network Components
- Activation functions: ReLU, LeakyReLU, Tanh, Sigmoid
- Weight initializers: He, Xavier/Glorot, LeCun
- Optimizers: SGD, Adam, RMSprop

## Results
- Quantitative evaluation: MSE, PSNR, SSIM
- Processing time comparison
- User preference study
- Ablation study

## Conclusion
- Neural approaches outperform classical DIP
- GAN-based models achieve highest quality
- Hybrid approaches leverage strengths of both paradigms
- Future work: video processing, additional styles, mobile deployment

" > presentation.md

# Create a demo script
echo "#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from style_processor import StyleProcessor

def main():
    # Check if image path is provided
    if len(sys.argv) < 2:
        print('Usage: python demo.py <image_path>')
        return
    
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f'Error: Image file {image_path} not found')
        return
    
    # Create output directory
    output_dir = 'demo_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize style processor
    processor = StyleProcessor()
    
    # Apply all styles
    print('Applying transformations...')
    results = processor.apply_all_styles(img_path=image_path)
    
    # Save results
    processor.save_styled_images(results, output_dir)
    
    # Create comparison visualization
    fig = processor.visualize_styles(results)
    plt.savefig(os.path.join(output_dir, 'comparison.png'), dpi=300, bbox_inches='tight')
    
    print(f'Transformation complete. Results saved to {output_dir}/')

if __name__ == '__main__':
    main()
" > demo.py
chmod +x demo.py

# Package everything
echo "Packaging final deliverables..."
