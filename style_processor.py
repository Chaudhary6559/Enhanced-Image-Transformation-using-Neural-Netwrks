import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

class StyleProcessor:
    """
    A class that processes images with multiple style transformations.
    Integrates different neural network models for cartoonization, Ghibli-style, and sketch effects.
    Includes fallback mechanisms if models are not loaded.
    """
    
    def __init__(self):
        """
        Initialize the StyleProcessor.
        Models are loaded explicitly via load_models method.
        """
        self.img_size = (256, 256)
        self.models_loaded = False # True if any neural model is loaded
        self.cartoon_model_loaded = False
        self.ghibli_model_loaded = False
        self.sketch_model_loaded = False
        
        self.cartoon_model = None
        self.ghibli_model = None
        self.sketch_model = None
        self.original_size = None # Store original image size for postprocessing
    
    def load_models(self, models_dir):
        """
        Load pre-trained models from directory.
        Checks for specific model files and updates loading status for each.
        
        Args:
            models_dir: Directory containing model files (e.g., cartoon_generator.h5)
            
        Returns:
            A dictionary indicating the loading status of each model.
        """
        self.models_loaded = False
        self.cartoon_model_loaded = False
        self.ghibli_model_loaded = False
        self.sketch_model_loaded = False
        status = {"cartoon": False, "ghibli": False, "sketch": False, "message": ""}
        messages = []

        # Try loading Cartoon model
        cartoon_path = os.path.join(models_dir, 'cartoon_generator.h5')
        if os.path.exists(cartoon_path):
            try:
                self.cartoon_model = load_model(cartoon_path)
                self.cartoon_model_loaded = True
                status["cartoon"] = True
                messages.append("Cartoon model loaded successfully.")
                print("Cartoon model loaded successfully.")
            except Exception as e:
                messages.append(f"Error loading cartoon model: {e}")
                print(f"Error loading cartoon model: {e}")
        else:
            messages.append("Cartoon model file (cartoon_generator.h5) not found.")
            print("Cartoon model file (cartoon_generator.h5) not found.")

        # Try loading Ghibli model
        ghibli_path = os.path.join(models_dir, 'ghibli_generator.h5')
        if os.path.exists(ghibli_path):
            try:
                self.ghibli_model = load_model(ghibli_path)
                self.ghibli_model_loaded = True
                status["ghibli"] = True
                messages.append("Ghibli model loaded successfully.")
                print("Ghibli model loaded successfully.")
            except Exception as e:
                messages.append(f"Error loading Ghibli model: {e}")
                print(f"Error loading Ghibli model: {e}")
        else:
            messages.append("Ghibli model file (ghibli_generator.h5) not found.")
            print("Ghibli model file (ghibli_generator.h5) not found.")

        # Try loading Sketch model
        sketch_path = os.path.join(models_dir, 'sketch_generator.h5')
        if os.path.exists(sketch_path):
            try:
                self.sketch_model = load_model(sketch_path)
                self.sketch_model_loaded = True
                status["sketch"] = True
                messages.append("Sketch model loaded successfully.")
                print("Sketch model loaded successfully.")
            except Exception as e:
                messages.append(f"Error loading sketch model: {e}")
                print(f"Error loading sketch model: {e}")
        else:
            messages.append("Sketch model file (sketch_generator.h5) not found.")
            print("Sketch model file (sketch_generator.h5) not found.")

        # Update overall models_loaded status
        self.models_loaded = self.cartoon_model_loaded or self.ghibli_model_loaded or self.sketch_model_loaded
        
        status["message"] = "\n".join(messages)
        if not self.models_loaded:
             messages.append("No neural models loaded. Neural styles will use fallbacks.")
             print("No neural models loaded. Neural styles will use fallbacks.")
        
        status["message"] = "\n".join(messages)
        return status

    def preprocess_image(self, img_path=None, img=None, target_size=None):
        """
        Preprocess image for neural network input.
        
        Args:
            img_path: Path to image file (optional)
            img: Image array (optional, alternative to img_path)
            target_size: Target size for resizing (default: self.img_size)
            
        Returns:
            Preprocessed image tensor
        """
        if target_size is None:
            target_size = self.img_size
        
        # Load image from path if provided
        if img_path is not None:
            img = cv2.imread(img_path)
            if img is None:
                 raise ValueError(f"Could not read image from path: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img is not None:
            # Ensure image is in RGB format if it has 3 channels
            if len(img.shape) == 3 and img.shape[2] == 3:
                 # Simple check: if blue channel mean is higher, assume BGR
                 if np.mean(img[:,:,0]) > np.mean(img[:,:,2]):
                      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif len(img.shape) == 2: # Grayscale image
                 img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            raise ValueError("Either img_path or img must be provided")
        
        # Store original size for later
        self.original_size = img.shape[:2] # height, width
        
        # Resize
        img_resized = cv2.resize(img, target_size)
        
        # Normalize to [-1, 1]
        img_normalized = img_resized.astype(np.float32) / 127.5 - 1
        
        # Add batch dimension
        img_tensor = np.expand_dims(img_normalized, axis=0)
        
        return img_tensor
    
    def postprocess_image(self, img_tensor, resize_to_original=True):
        """
        Convert output tensor to displayable image.
        
        Args:
            img_tensor: Output tensor from model
            resize_to_original: Whether to resize back to original dimensions
            
        Returns:
            Processed image in RGB format (uint8)
        """
        # Remove batch dimension
        img = np.squeeze(img_tensor, axis=0)
        
        # Denormalize from [-1, 1] to [0, 255]
        img = (img + 1) * 127.5
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        # Resize to original dimensions if requested and original_size is known
        if resize_to_original and self.original_size is not None:
            # cv2.resize expects (width, height)
            img = cv2.resize(img, (self.original_size[1], self.original_size[0]))
        
        # Ensure output is RGB
        if len(img.shape) == 2: # If model output is grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 1: # If model output is single channel
             img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
             
        return img

    def _apply_style_with_fallback(self, model, model_loaded_flag, fallback_func, img_path, img):
        """Helper function to apply a style or its fallback."""
        if model_loaded_flag:
            try:
                img_tensor = self.preprocess_image(img_path, img)
                output_tensor = model.predict(img_tensor)
                output_img = self.postprocess_image(output_tensor)
                return output_img
            except Exception as e:
                print(f"Error during model prediction: {e}. Using fallback.")
                # Fall through to fallback if prediction fails
        
        # Use fallback if model not loaded or prediction failed
        print("Applying fallback transformation.")
        if img is None and img_path is not None:
             img = cv2.imread(img_path)
             if img is None:
                  raise ValueError(f"Could not read image from path: {img_path}")
             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img is None:
             raise ValueError("Image input required for fallback.")
             
        # Ensure img is RGB for fallback functions
        if len(img.shape) == 3 and img.shape[2] == 3:
             if np.mean(img[:,:,0]) > np.mean(img[:,:,2]): # Basic BGR check
                  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif len(img.shape) == 2:
             img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
             
        return fallback_func(img)

    # --- Fallback Implementations ---
    def _fallback_cartoon(self, img):
        """Fallback: Apply classical cartoon effect."""
        print("Applying classical cartoon as fallback for neural cartoon.")
        return self.apply_classical_cartoon(img=img)

    def _fallback_ghibli(self, img):
        """Fallback: Apply bilateral filter and adjust colors for a softer, painterly look."""
        print("Applying bilateral filter and color adjustment as fallback for Ghibli style.")
        # 1. Apply bilateral filter for smoothing while preserving edges
        # Parameters: d=diameter, sigmaColor=color similarity, sigmaSpace=spatial proximity
        # Larger sigma values mean more smoothing.
        d = 7 # Diameter of each pixel neighborhood
        sigma_color = 50
        sigma_space = 50
        img_filtered = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
        
        # 2. Convert to HSV color space
        img_hsv = cv2.cvtColor(img_filtered, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # 3. Adjust saturation and value for pastel tones
        # Slightly decrease saturation (e.g., multiply by 0.8-0.9)
        # Slightly increase value (e.g., multiply by 1.1-1.2)
        saturation_factor = 0.85
        value_factor = 1.1
        
        img_hsv[:, :, 1] *= saturation_factor # Decrease Saturation
        img_hsv[:, :, 2] *= value_factor    # Increase Value (Brightness)
        
        # Clip values to valid range [0, 255] for S and V
        img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1], 0, 255)
        img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2], 0, 255)
        
        # 4. Convert back to RGB
        img_adjusted = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # Optional: Add a very subtle sharpening or detail enhancement if needed
        # For now, let's return the color-adjusted filtered image
        return img_adjusted

    def _fallback_sketch(self, img):
        """Fallback: Convert to grayscale sketch using dodge blend."""
        print("Applying grayscale sketch as fallback for neural sketch.")
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_gray_inv = 255 - img_gray
        img_blur = cv2.GaussianBlur(img_gray_inv, ksize=(21, 21), sigmaX=0, sigmaY=0)
        img_blend = cv2.divide(img_gray, 255 - img_blur, scale=256)
        # Convert back to RGB for consistency
        img_sketch_rgb = cv2.cvtColor(img_blend, cv2.COLOR_GRAY2RGB)
        return img_sketch_rgb
    # --------------------------------
    
    def apply_cartoon_style(self, img_path=None, img=None):
        """
        Apply cartoon style to image using model or fallback.
        """
        return self._apply_style_with_fallback(
            self.cartoon_model, self.cartoon_model_loaded, self._fallback_cartoon, img_path, img
        )
    
    def apply_ghibli_style(self, img_path=None, img=None):
        """
        Apply Ghibli style to image using model or fallback.
        """
        return self._apply_style_with_fallback(
            self.ghibli_model, self.ghibli_model_loaded, self._fallback_ghibli, img_path, img
        )
    
    def apply_sketch_style(self, img_path=None, img=None):
        """
        Apply sketch style to image using model or fallback.
        """
        return self._apply_style_with_fallback(
            self.sketch_model, self.sketch_model_loaded, self._fallback_sketch, img_path, img
        )
    
    def apply_classical_cartoon(self, img_path=None, img=None, line_size=7, blur_value=7, 
                               bilateral_d=9, bilateral_sigma_color=75, bilateral_sigma_space=75,
                               edge_threshold1=50, edge_threshold2=150, total_color_levels=8):
        """
        Apply classical DIP-based cartoonization for comparison.
        
        Args:
            img_path: Path to image file (optional)
            img: Image array (optional, alternative to img_path)
            Various parameters for classical cartoonization
            
        Returns:
            Cartoonized image using classical methods (RGB format)
        """
        # Load image
        if img_path is not None:
            img = cv2.imread(img_path)
            if img is None:
                 raise ValueError(f"Could not read image from path: {img_path}")
            # Assume BGR, convert to RGB for internal processing
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img is not None:
            # Make a copy to avoid modifying the original
            img = img.copy()
            # Ensure image is RGB
            if len(img.shape) == 3 and img.shape[2] == 3:
                 if np.mean(img[:,:,0]) > np.mean(img[:,:,2]): # Basic BGR check
                      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif len(img.shape) == 2:
                 img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            raise ValueError("Either img_path or img must be provided")
        
        # Store original size if not already set (might be called directly)
        if self.original_size is None:
             self.original_size = img.shape[:2]
        
        # Convert to BGR for OpenCV processing steps
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Apply bilateral filter
        filtered = cv2.bilateralFilter(img_bgr, bilateral_d, bilateral_sigma_color, bilateral_sigma_space)
        
        # Apply color quantization
        data = np.float32(filtered).reshape((-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
        ret, label, center = cv2.kmeans(data, total_color_levels, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        color_quantized = center[label.flatten()].reshape(img_bgr.shape)
        
        # Apply edge detection
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, line_size)
        edges = cv2.Canny(gray_blur, edge_threshold1, edge_threshold2)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        edges = cv2.bitwise_not(edges)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Combine edges with the smoothed (bilaterally filtered) image to preserve colors
        cartoon = cv2.bitwise_and(filtered, edges)
        
        # Convert final result back to RGB for consistent output
        cartoon_rgb = cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)
        
        # Resize to original dimensions if needed (using stored original_size)
        if self.original_size is not None:
             cartoon_rgb = cv2.resize(cartoon_rgb, (self.original_size[1], self.original_size[0]))
             
        return cartoon_rgb
    
    def apply_all_styles(self, img_path=None, img=None):
        """
        Apply all available styles to an image, using models or fallbacks.
        
        Args:
            img_path: Path to image file (optional)
            img: Image array (optional, alternative to img_path)
            
        Returns:
            Dictionary of styled images (all in RGB format)
        """
        # Load image and ensure it's RGB
        if img_path is not None:
            original_img = cv2.imread(img_path)
            if original_img is None:
                 raise ValueError(f"Could not read image from path: {img_path}")
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        elif img is not None:
            original_img = img.copy()
            # Ensure image is RGB
            if len(original_img.shape) == 3 and original_img.shape[2] == 3:
                 if np.mean(original_img[:,:,0]) > np.mean(original_img[:,:,2]): # Basic BGR check
                      original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            elif len(original_img.shape) == 2:
                 original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
        else:
            raise ValueError("Either img_path or img must be provided")
        
        # Set original size for potential resizing in postprocessing/fallbacks
        self.original_size = original_img.shape[:2]
        
        # Apply all styles
        results = {
            'original': original_img,
            'classical_cartoon': self.apply_classical_cartoon(img=original_img),
            'neural_cartoon': self.apply_cartoon_style(img=original_img),
            'ghibli': self.apply_ghibli_style(img=original_img),
            'sketch': self.apply_sketch_style(img=original_img)
        }
        
        return results
    
    def visualize_styles(self, results, figsize=(15, 10)):
        """
        Visualize all styles side by side.
        
        Args:
            results: Dictionary of styled images from apply_all_styles
            figsize: Figure size for the plot
            
        Returns:
            Matplotlib figure
        """
        n_styles = len(results)
        # Adjust layout based on number of styles (e.g., 2 rows if > 3 styles)
        cols = min(n_styles, 3)
        rows = (n_styles + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() # Flatten axes array for easy iteration
        
        for i, (style_name, img) in enumerate(results.items()):
            if i < len(axes):
                axes[i].imshow(img)
                axes[i].set_title(style_name.replace('_', ' ').title())
                axes[i].axis('off')
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
             axes[j].axis('off')
             
        plt.tight_layout()
        return fig
    
    def save_styled_images(self, results, output_dir):
        """
        Save all styled images to directory.
        
        Args:
            results: Dictionary of styled images from apply_all_styles (expected in RGB)
            output_dir: Directory to save images
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for style_name, img in results.items():
            output_path = os.path.join(output_dir, f"{style_name}.jpg")
            
            # Convert RGB to BGR for OpenCV imwrite
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
            cv2.imwrite(output_path, img_bgr)
            print(f"Saved {style_name} image to {output_path}")

