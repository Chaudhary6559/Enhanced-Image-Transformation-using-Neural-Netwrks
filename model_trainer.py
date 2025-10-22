import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns

class ModelTrainer:
    """
    Class for training and evaluating the neural network models for image transformation.
    """
    
    def __init__(self, models_dir='models'):
        """
        Initialize the ModelTrainer.
        
        Args:
            models_dir: Directory to save trained models
        """
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize metrics tracking
        self.history = {}
        self.evaluation_results = {}
        
    def prepare_dataset(self, photo_dir, cartoon_dir, ghibli_dir, sketch_dir, img_size=(256, 256), batch_size=8):
        """
        Prepare datasets for training different style models.
        
        Args:
            photo_dir: Directory containing real photos
            cartoon_dir: Directory containing cartoon images
            ghibli_dir: Directory containing Ghibli-style images
            sketch_dir: Directory containing sketch images
            img_size: Target image size
            batch_size: Batch size for training
            
        Returns:
            Dictionary of data generators for different styles
        """
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2
        )
        
        # Validation data generator
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        
        # Create generators for photos (input)
        train_photo_generator = train_datagen.flow_from_directory(
            photo_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode=None,
            subset='training',
            shuffle=True
        )
        
        val_photo_generator = val_datagen.flow_from_directory(
            photo_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode=None,
            subset='validation',
            shuffle=True
        )
        
        # Create generators for cartoon style (output)
        if os.path.exists(cartoon_dir):
            train_cartoon_generator = train_datagen.flow_from_directory(
                cartoon_dir,
                target_size=img_size,
                batch_size=batch_size,
                class_mode=None,
                subset='training',
                shuffle=True
            )
            
            val_cartoon_generator = val_datagen.flow_from_directory(
                cartoon_dir,
                target_size=img_size,
                batch_size=batch_size,
                class_mode=None,
                subset='validation',
                shuffle=True
            )
        else:
            train_cartoon_generator = None
            val_cartoon_generator = None
        
        # Create generators for Ghibli style (output)
        if os.path.exists(ghibli_dir):
            train_ghibli_generator = train_datagen.flow_from_directory(
                ghibli_dir,
                target_size=img_size,
                batch_size=batch_size,
                class_mode=None,
                subset='training',
                shuffle=True
            )
            
            val_ghibli_generator = val_datagen.flow_from_directory(
                ghibli_dir,
                target_size=img_size,
                batch_size=batch_size,
                class_mode=None,
                subset='validation',
                shuffle=True
            )
        else:
            train_ghibli_generator = None
            val_ghibli_generator = None
        
        # Create generators for sketch style (output)
        if os.path.exists(sketch_dir):
            train_sketch_generator = train_datagen.flow_from_directory(
                sketch_dir,
                target_size=img_size,
                batch_size=batch_size,
                class_mode=None,
                subset='training',
                shuffle=True,
                color_mode='grayscale'
            )
            
            val_sketch_generator = val_datagen.flow_from_directory(
                sketch_dir,
                target_size=img_size,
                batch_size=batch_size,
                class_mode=None,
                subset='validation',
                shuffle=True,
                color_mode='grayscale'
            )
        else:
            train_sketch_generator = None
            val_sketch_generator = None
        
        # Create combined generators
        def combine_generators(gen1, gen2):
            while True:
                x = next(gen1)
                y = next(gen2)
                yield (x, y)
        
        generators = {}
        
        if train_cartoon_generator is not None:
            generators['cartoon'] = {
                'train': combine_generators(train_photo_generator, train_cartoon_generator),
                'val': combine_generators(val_photo_generator, val_cartoon_generator)
            }
        
        if train_ghibli_generator is not None:
            generators['ghibli'] = {
                'train': combine_generators(train_photo_generator, train_ghibli_generator),
                'val': combine_generators(val_photo_generator, val_ghibli_generator)
            }
        
        if train_sketch_generator is not None:
            generators['sketch'] = {
                'train': combine_generators(train_photo_generator, train_sketch_generator),
                'val': combine_generators(val_photo_generator, val_sketch_generator)
            }
        
        # Store steps per epoch
        self.steps_per_epoch = {
            'cartoon': len(train_photo_generator) if train_cartoon_generator is not None else 0,
            'ghibli': len(train_photo_generator) if train_ghibli_generator is not None else 0,
            'sketch': len(train_photo_generator) if train_sketch_generator is not None else 0
        }
        
        self.validation_steps = {
            'cartoon': len(val_photo_generator) if val_cartoon_generator is not None else 0,
            'ghibli': len(val_photo_generator) if val_ghibli_generator is not None else 0,
            'sketch': len(val_photo_generator) if val_sketch_generator is not None else 0
        }
        
        return generators
    
    def train_cartoon_model(self, model, generators, epochs=50):
        """
        Train the cartoon style model.
        
        Args:
            model: Cartoon generator model
            generators: Data generators dictionary
            epochs: Number of training epochs
            
        Returns:
            Training history
        """
        if 'cartoon' not in generators:
            print("Cartoon generators not available. Skipping training.")
            return None
        
        # Create callbacks
        checkpoint = ModelCheckpoint(
            os.path.join(self.models_dir, 'cartoon_generator.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        tensorboard = TensorBoard(
            log_dir=os.path.join(self.models_dir, 'logs', 'cartoon'),
            histogram_freq=1,
            write_graph=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        # Train the model
        history = model.fit(
            generators['cartoon']['train'],
            steps_per_epoch=self.steps_per_epoch['cartoon'],
            epochs=epochs,
            validation_data=generators['cartoon']['val'],
            validation_steps=self.validation_steps['cartoon'],
            callbacks=[checkpoint, tensorboard, reduce_lr]
        )
        
        # Store history
        self.history['cartoon'] = history.history
        
        return history
    
    def train_ghibli_model(self, model, generators, epochs=50):
        """
        Train the Ghibli style model.
        
        Args:
            model: Ghibli generator model
            generators: Data generators dictionary
            epochs: Number of training epochs
            
        Returns:
            Training history
        """
        if 'ghibli' not in generators:
            print("Ghibli generators not available. Skipping training.")
            return None
        
        # Create callbacks
        checkpoint = ModelCheckpoint(
            os.path.join(self.models_dir, 'ghibli_generator.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        tensorboard = TensorBoard(
            log_dir=os.path.join(self.models_dir, 'logs', 'ghibli'),
            histogram_freq=1,
            write_graph=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        # Train the model
        history = model.fit(
            generators['ghibli']['train'],
            steps_per_epoch=self.steps_per_epoch['ghibli'],
            epochs=epochs,
            validation_data=generators['ghibli']['val'],
            validation_steps=self.validation_steps['ghibli'],
            callbacks=[checkpoint, tensorboard, reduce_lr]
        )
        
        # Store history
        self.history['ghibli'] = history.history
        
        return history
    
    def train_sketch_model(self, model, generators, epochs=50):
        """
        Train the sketch style model.
        
        Args:
            model: Sketch generator model
            generators: Data generators dictionary
            epochs: Number of training epochs
            
        Returns:
            Training history
        """
        if 'sketch' not in generators:
            print("Sketch generators not available. Skipping training.")
            return None
        
        # Create callbacks
        checkpoint = ModelCheckpoint(
            os.path.join(self.models_dir, 'sketch_generator.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        tensorboard = TensorBoard(
            log_dir=os.path.join(self.models_dir, 'logs', 'sketch'),
            histogram_freq=1,
            write_graph=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        # Train the model
        history = model.fit(
            generators['sketch']['train'],
            steps_per_epoch=self.steps_per_epoch['sketch'],
            epochs=epochs,
            validation_data=generators['sketch']['val'],
            validation_steps=self.validation_steps['sketch'],
            callbacks=[checkpoint, tensorboard, reduce_lr]
        )
        
        # Store history
        self.history['sketch'] = history.history
        
        return history
    
    def evaluate_models(self, models, test_photos, test_targets):
        """
        Evaluate trained models on test data.
        
        Args:
            models: Dictionary of trained models
            test_photos: Test input images
            test_targets: Dictionary of test target images for each style
            
        Returns:
            Dictionary of evaluation metrics
        """
        results = {}
        
        for style, model in models.items():
            if style in test_targets:
                print(f"Evaluating {style} model...")
                
                # Generate predictions
                predictions = model.predict(test_photos)
                
                # Calculate MSE
                mse = np.mean(np.square(test_targets[style] - predictions))
                
                # Calculate PSNR
                max_pixel = 1.0
                psnr = 10.0 * np.log10((max_pixel ** 2) / mse)
                
                # Calculate SSIM (structural similarity index)
                ssim_scores = []
                for i in range(len(test_photos)):
                    pred = np.clip(predictions[i], 0, 1)
                    target = test_targets[style][i]
                    
                    # Convert to uint8 for SSIM calculation
                    pred_uint8 = (pred * 255).astype(np.uint8)
                    target_uint8 = (target * 255).astype(np.uint8)
                    
                    # Calculate SSIM
                    if style == 'sketch':
                        # For sketch (grayscale)
                        ssim = cv2.compareSSIM(pred_uint8, target_uint8)
                    else:
                        # For RGB images
                        ssim = cv2.compareSSIM(pred_uint8, target_uint8, multichannel=True)
                    
                    ssim_scores.append(ssim)
                
                avg_ssim = np.mean(ssim_scores)
                
                # Store results
                results[style] = {
                    'mse': mse,
                    'psnr': psnr,
                    'ssim': avg_ssim
                }
                
                print(f"{style} evaluation: MSE={mse:.4f}, PSNR={psnr:.2f}dB, SSIM={avg_ssim:.4f}")
        
        # Store evaluation results
        self.evaluation_results = results
        
        return results
    
    def plot_training_history(self, save_dir=None):
        """
        Plot training history for all models.
        
        Args:
            save_dir: Directory to save plots (optional)
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        for style, history in self.history.items():
            # Plot loss
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(history['loss'], label='Training Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.title(f'{style.capitalize()} Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            # Plot metrics if available
            if 'accuracy' in history:
                plt.subplot(1, 2, 2)
                plt.plot(history['accuracy'], label='Training Accuracy')
                plt.plot(history['val_accuracy'], label='Validation Accuracy')
                plt.title(f'{style.capitalize()} Model Accuracy')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.grid(True)
            
            plt.tight_layout()
            
            if save_dir:
                plt.savefig(os.path.join(save_dir, f'{style}_training_history.png'), dpi=300)
                plt.close()
            else:
                plt.show()
    
    def visualize_results(self, models, test_photos, save_dir=None):
        """
        Visualize transformation results for test images.
        
        Args:
            models: Dictionary of trained models
            test_photos: Test input images
            save_dir: Directory to save visualizations (optional)
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Select a subset of test images for visualization
        num_samples = min(5, len(test_photos))
        indices = np.random.choice(len(test_photos), num_samples, replace=False)
        
        for idx in indices:
            photo = test_photos[idx:idx+1]
            
            # Create figure
            n_cols = len(models) + 1
            plt.figure(figsize=(n_cols * 4, 4))
            
            # Display original photo
            plt.subplot(1, n_cols, 1)
            plt.imshow(np.clip(photo[0], 0, 1))
            plt.title('Original Photo')
            plt.axis('off')
            
            # Display transformed images
            col = 2
            for style, model in models.items():
                # Generate transformation
                transformed = model.predict(photo)
                
                # Display result
                plt.subplot(1, n_cols, col)
                
                if style == 'sketch' and transformed.shape[-1] == 1:
                    # For grayscale sketch
                    plt.imshow(np.clip(transformed[0, :, :, 0], 0, 1), cmap='gray')
                else:
                    # For RGB images
                    plt.imshow(np.clip(transformed[0], 0, 1))
                
                plt.title(f'{style.capitalize()} Style')
                plt.axis('off')
                
                col += 1
            
            plt.tight_layout()
            
            if save_dir:
                plt.savefig(os.path.join(save_dir, f'sample_{idx}_comparison.png'), dpi=300)
                plt.close()
            else:
                plt.show()
    
    def create_confusion_matrix(self, y_true, y_pred, classes, save_path=None):
        """
        Create and plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            classes: List of class names
            save_path: Path to save the plot (optional)
        """
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def calculate_f1_score(self, y_true, y_pred, average='weighted'):
        """
        Calculate F1 score.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: Averaging method for F1 score
            
        Returns:
            F1 score
        """
        return f1_score(y_true, y_pred, average=average)
    
    def calculate_iou(self, y_true, y_pred, threshold=0.5):
        """
        Calculate IoU (Intersection over Union) for segmentation tasks.
        
        Args:
            y_true: True segmentation masks
            y_pred: Predicted segmentation masks
            threshold: Threshold for binarizing predictions
            
        Returns:
            Mean IoU score
        """
        # Binarize predictions
        y_pred_bin = (y_pred > threshold).astype(np.uint8)
        
        # Calculate IoU for each image
        iou_scores = []
        for i in range(len(y_true)):
            intersection = np.logical_and(y_true[i], y_pred_bin[i])
            union = np.logical_or(y_true[i], y_pred_bin[i])
            iou = np.sum(intersection) / np.sum(union)
            iou_scores.append(iou)
        
        # Return mean IoU
        return np.mean(iou_scores)
    
    def save_model_architecture(self, models, save_dir):
        """
        Save model architecture diagrams.
        
        Args:
            models: Dictionary of models
            save_dir: Directory to save architecture diagrams
        """
        os.makedirs(save_dir, exist_ok=True)
        
        for style, model in models.items():
            # Save model summary to text file
            with open(os.path.join(save_dir, f'{style}_model_summary.txt'), 'w') as f:
                # Redirect summary to file
                model.summary(print_fn=lambda x: f.write(x + '\n'))
            
            # Save model architecture diagram
            tf.keras.utils.plot_model(
                model,
                to_file=os.path.join(save_dir, f'{style}_model_architecture.png'),
                show_shapes=True,
                show_layer_names=True,
                rankdir='TB',
                dpi=96
            )
            
            print(f"Saved {style} model architecture to {save_dir}")
