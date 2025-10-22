#!/usr/bin/env python3
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
