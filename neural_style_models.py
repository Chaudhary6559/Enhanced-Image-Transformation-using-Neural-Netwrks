import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization, ReLU, LeakyReLU, Dropout

class StyleTransferModel:
    """
    Implementation of neural style transfer for cartoonization.
    Uses VGG19 for feature extraction and computes content and style losses.
    """
    
    def __init__(self):
        self.content_layers = ['block4_conv2']
        self.style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
        self.num_content_layers = len(self.content_layers)
        self.num_style_layers = len(self.style_layers)
        self.vgg = self._build_vgg()
        
    def _build_vgg(self):
        """Build and return the VGG19 model for feature extraction."""
        # Load pre-trained VGG19 model
        vgg = VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        
        # Get output layers corresponding to style and content layers
        outputs = [vgg.get_layer(name).output for name in self.style_layers + self.content_layers]
        
        # Create model
        model = Model(inputs=vgg.input, outputs=outputs)
        return model
    
    def _gram_matrix(self, input_tensor):
        """Calculate Gram matrix for style representation."""
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return result / num_locations
    
    def _preprocess_image(self, image):
        """Preprocess image for VGG19."""
        # Convert from RGB to BGR
        image = image[..., ::-1]
        # Normalize with VGG19 mean and std
        mean = [103.939, 116.779, 123.68]
        image = image - mean
        return image
    
    def _get_style_content_features(self, image):
        """Extract style and content features from image."""
        # Preprocess image
        preprocessed = self._preprocess_image(image)
        
        # Get features
        outputs = self.vgg(preprocessed)
        
        # Split style and content features
        style_outputs = outputs[:self.num_style_layers]
        content_outputs = outputs[self.num_style_layers:]
        
        # Calculate Gram matrices for style features
        style_features = [self._gram_matrix(style_output) for style_output in style_outputs]
        
        # Return features
        return {'style': style_features, 'content': content_outputs}
    
    def compute_loss(self, generated_image, content_image, style_image, style_weight=1e-2, content_weight=1e4):
        """Compute total loss for style transfer."""
        # Get features
        generated_features = self._get_style_content_features(generated_image)
        content_features = self._get_style_content_features(content_image)
        style_features = self._get_style_content_features(style_image)
        
        # Initialize loss
        style_loss = 0
        content_loss = 0
        
        # Calculate style loss
        for gen_style, target_style in zip(generated_features['style'], style_features['style']):
            style_loss += tf.reduce_mean(tf.square(gen_style - target_style))
        style_loss *= style_weight / self.num_style_layers
        
        # Calculate content loss
        for gen_content, target_content in zip(generated_features['content'], content_features['content']):
            content_loss += tf.reduce_mean(tf.square(gen_content - target_content))
        content_loss *= content_weight / self.num_content_layers
        
        # Total loss
        total_loss = style_loss + content_loss
        
        return total_loss, style_loss, content_loss
    
    @tf.function
    def train_step(self, generated_image, content_image, style_image, optimizer, style_weight=1e-2, content_weight=1e4):
        """Perform one training step."""
        with tf.GradientTape() as tape:
            total_loss, style_loss, content_loss = self.compute_loss(
                generated_image, content_image, style_image, style_weight, content_weight)
            
        # Compute gradients
        gradients = tape.gradient(total_loss, generated_image)
        
        # Apply gradients
        optimizer.apply_gradients([(gradients, generated_image)])
        
        # Clip pixel values to valid range
        generated_image.assign(tf.clip_by_value(generated_image, 0.0, 255.0))
        
        return total_loss, style_loss, content_loss

class CartoonGANModel:
    """
    Implementation of CartoonGAN for unpaired image-to-cartoon translation.
    Uses a generator and discriminator in an adversarial setup.
    """
    
    def __init__(self, img_size=(256, 256)):
        self.img_size = img_size
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        self.gen_optimizer = Adam(learning_rate=2e-4, beta_1=0.5)
        self.disc_optimizer = Adam(learning_rate=2e-4, beta_1=0.5)
        self.gen_loss_fn = MeanSquaredError()
        self.disc_loss_fn = BinaryCrossentropy(from_logits=True)
        
    def _build_generator(self):
        """Build generator model with U-Net architecture."""
        # Input
        inputs = Input(shape=(self.img_size[0], self.img_size[1], 3))
        
        # Encoder
        # Using He initialization for ReLU activations
        conv1 = Conv2D(64, (7, 7), padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = ReLU()(conv1)
        
        conv2 = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = ReLU()(conv2)
        
        conv3 = Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv2)
        conv3 = BatchNormalization()(conv3)
        conv3 = ReLU()(conv3)
        
        # Residual blocks
        res = conv3
        for i in range(8):
            res_conv1 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(res)
            res_conv1 = BatchNormalization()(res_conv1)
            res_conv1 = ReLU()(res_conv1)
            
            res_conv2 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(res_conv1)
            res_conv2 = BatchNormalization()(res_conv2)
            
            res = res + res_conv2  # Skip connection
            res = ReLU()(res)
        
        # Decoder
        up1 = UpSampling2D(size=(2, 2))(res)
        up1 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(up1)
        up1 = BatchNormalization()(up1)
        up1 = ReLU()(up1)
        
        up2 = UpSampling2D(size=(2, 2))(up1)
        up2 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(up2)
        up2 = BatchNormalization()(up2)
        up2 = ReLU()(up2)
        
        # Output layer
        outputs = Conv2D(3, (7, 7), padding='same', activation='tanh')(up2)
        
        # Create model
        return Model(inputs=inputs, outputs=outputs, name='generator')
    
    def _build_discriminator(self):
        """Build discriminator model."""
        # Input
        inputs = Input(shape=(self.img_size[0], self.img_size[1], 3))
        
        # Using LeakyReLU with He initialization
        conv1 = Conv2D(32, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = LeakyReLU(alpha=0.2)(conv1)
        
        conv2 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = LeakyReLU(alpha=0.2)(conv2)
        
        conv3 = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv2)
        conv3 = BatchNormalization()(conv3)
        conv3 = LeakyReLU(alpha=0.2)(conv3)
        
        conv4 = Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv3)
        conv4 = BatchNormalization()(conv4)
        conv4 = LeakyReLU(alpha=0.2)(conv4)
        
        # Output layer
        outputs = Conv2D(1, (3, 3), padding='same')(conv4)
        
        # Create model
        return Model(inputs=inputs, outputs=outputs, name='discriminator')
    
    @tf.function
    def train_generator_step(self, real_photos, real_cartoons):
        """Train generator for one step."""
        with tf.GradientTape() as tape:
            # Generate fake cartoons
            fake_cartoons = self.generator(real_photos, training=True)
            
            # Discriminator predictions
            fake_preds = self.discriminator(fake_cartoons, training=True)
            
            # Calculate generator loss
            gen_loss = self.gen_loss_fn(tf.ones_like(fake_preds), fake_preds)
            
            # Add content preservation loss
            content_loss = tf.reduce_mean(tf.abs(real_photos - fake_cartoons))
            total_gen_loss = gen_loss + 10 * content_loss
        
        # Calculate gradients and update generator
        gen_gradients = tape.gradient(total_gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        
        return total_gen_loss, gen_loss, content_loss
    
    @tf.function
    def train_discriminator_step(self, real_photos, real_cartoons):
        """Train discriminator for one step."""
        with tf.GradientTape() as tape:
            # Generate fake cartoons
            fake_cartoons = self.generator(real_photos, training=True)
            
            # Discriminator predictions
            real_preds = self.discriminator(real_cartoons, training=True)
            fake_preds = self.discriminator(fake_cartoons, training=True)
            
            # Calculate discriminator loss
            real_loss = self.disc_loss_fn(tf.ones_like(real_preds), real_preds)
            fake_loss = self.disc_loss_fn(tf.zeros_like(fake_preds), fake_preds)
            total_disc_loss = real_loss + fake_loss
        
        # Calculate gradients and update discriminator
        disc_gradients = tape.gradient(total_disc_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))
        
        return total_disc_loss, real_loss, fake_loss

class SketchRNNModel:
    """
    Implementation of a recurrent neural network for sketch generation.
    Uses LSTM/GRU layers for sequence modeling of sketch strokes.
    """
    
    def __init__(self, img_size=(256, 256)):
        self.img_size = img_size
        self.model = self._build_model()
        self.optimizer = RMSprop(learning_rate=0.001)
        
    def _build_model(self):
        """Build sketch generation model with CNN and RNN layers."""
        # Input
        inputs = Input(shape=(self.img_size[0], self.img_size[1], 3))
        
        # CNN feature extraction
        # Using LeCun initialization for sketch network
        conv1 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer='lecun_normal')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = ReLU()(conv1)
        
        conv2 = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer='lecun_normal')(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = ReLU()(conv2)
        
        conv3 = Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer='lecun_normal')(conv2)
        conv3 = BatchNormalization()(conv3)
        conv3 = ReLU()(conv3)
        
        # Reshape for RNN
        features = tf.keras.layers.Reshape((-1, 256))(conv3)
        
        # RNN layers
        lstm1 = tf.keras.layers.LSTM(512, return_sequences=True)(features)
        lstm1 = Dropout(0.3)(lstm1)
        
        lstm2 = tf.keras.layers.LSTM(512, return_sequences=True)(lstm1)
        lstm2 = Dropout(0.3)(lstm2)
        
        # Reshape back to spatial dimensions
        reshaped = tf.keras.layers.Reshape((self.img_size[0]//8, self.img_size[1]//8, 512))(lstm2)
        
        # Decoder
        up1 = UpSampling2D(size=(2, 2))(reshaped)
        up1 = Conv2D(256, (3, 3), padding='same', kernel_initializer='lecun_normal')(up1)
        up1 = BatchNormalization()(up1)
        up1 = ReLU()(up1)
        
        up2 = UpSampling2D(size=(2, 2))(up1)
        up2 = Conv2D(128, (3, 3), padding='same', kernel_initializer='lecun_normal')(up2)
        up2 = BatchNormalization()(up2)
        up2 = ReLU()(up2)
        
        up3 = UpSampling2D(size=(2, 2))(up2)
        up3 = Conv2D(64, (3, 3), padding='same', kernel_initializer='lecun_normal')(up3)
        up3 = BatchNormalization()(up3)
        up3 = ReLU()(up3)
        
        # Output layer - single channel for sketch
        outputs = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(up3)
        
        # Create model
        return Model(inputs=inputs, outputs=outputs, name='sketch_rnn')
    
    @tf.function
    def train_step(self, images, sketches):
        """Train model for one step."""
        with tf.GradientTape() as tape:
            # Generate sketches
            pred_sketches = self.model(images, training=True)
            
            # Calculate loss
            loss = tf.reduce_mean(tf.square(sketches - pred_sketches))
            
        # Calculate gradients and update model
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss

class GhibliStyleGAN:
    """
    Implementation of a GAN for Ghibli-style transfer.
    Uses a generator and discriminator in an adversarial setup with style-specific features.
    """
    
    def __init__(self, img_size=(256, 256)):
        self.img_size = img_size
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        self.gen_optimizer = Adam(learning_rate=1e-4, beta_1=0.5)
        self.disc_optimizer = Adam(learning_rate=1e-4, beta_1=0.5)
        
    def _build_generator(self):
        """Build generator model with residual blocks."""
        # Input
        inputs = Input(shape=(self.img_size[0], self.img_size[1], 3))
        
        # Encoder
        # Using Xavier/Glorot initialization for tanh activations
        conv1 = Conv2D(64, (7, 7), padding='same', kernel_initializer='glorot_normal')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = LeakyReLU(alpha=0.2)(conv1)
        
        conv2 = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer='glorot_normal')(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = LeakyReLU(alpha=0.2)(conv2)
        
        conv3 = Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer='glorot_normal')(conv2)
        conv3 = BatchNormalization()(conv3)
        conv3 = LeakyReLU(alpha=0.2)(conv3)
        
        # Residual blocks
        res = conv3
        for i in range(9):  # More residual blocks for better style transfer
            res_conv1 = Conv2D(256, (3, 3), padding='same', kernel_initializer='glorot_normal')(res)
            res_conv1 = BatchNormalization()(res_conv1)
            res_conv1 = LeakyReLU(alpha=0.2)(res_conv1)
            
            res_conv2 = Conv2D(256, (3, 3), padding='same', kernel_initializer='glorot_normal')(res_conv1)
            res_conv2 = BatchNormalization()(res_conv2)
            
            res = res + res_conv2  # Skip connection
            res = LeakyReLU(alpha=0.2)(res)
        
        # Decoder
        up1 = UpSampling2D(size=(2, 2))(res)
        up1 = Conv2D(128, (3, 3), padding='same', kernel_initializer='glorot_normal')(up1)
        up1 = BatchNormalization()(up1)
        up1 = LeakyReLU(alpha=0.2)(up1)
        
        up2 = UpSampling2D(size=(2, 2))(up1)
        up2 = Conv2D(64, (3, 3), padding='same', kernel_initializer='glorot_normal')(up2)
        up2 = BatchNormalization()(up2)
        up2 = LeakyReLU(alpha=0.2)(up2)
        
        # Output layer
        outputs = Conv2D(3, (7, 7), padding='same', activation='tanh')(up2)
        
        # Create model
        return Model(inputs=inputs, outputs=outputs, name='ghibli_generator')
    
    def _build_discriminator(self):
        """Build discriminator model with PatchGAN architecture."""
        # Input
        inputs = Input(shape=(self.img_size[0], self.img_size[1], 3))
        
        # Using LeakyReLU with He initialization
        conv1 = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = LeakyReLU(alpha=0.2)(conv1)
        
        conv2 = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = LeakyReLU(alpha=0.2)(conv2)
        
        conv3 = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv2)
        conv3 = BatchNormalization()(conv3)
        conv3 = LeakyReLU(alpha=0.2)(conv3)
        
        conv4 = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv3)
        conv4 = BatchNormalization()(conv4)
        conv4 = LeakyReLU(alpha=0.2)(conv4)
        
        # Output layer - PatchGAN
        outputs = Conv2D(1, (4, 4), padding='same')(conv4)
        
        # Create model
        return Model(inputs=inputs, outputs=outputs, name='ghibli_discriminator')
    
    @tf.function
    def train_generator_step(self, real_photos, real_ghibli):
        """Train generator for one step."""
        with tf.GradientTape() as tape:
            # Generate fake Ghibli-style images
            fake_ghibli = self.generator(real_photos, training=True)
            
            # Discriminator predictions
            fake_preds = self.discriminator(fake_ghibli, training=True)
            
            # Calculate generator loss
            gen_loss = tf.reduce_mean(tf.square(fake_preds - 1))
            
            # Add content preservation loss
            content_loss = tf.reduce_mean(tf.abs(real_photos - fake_ghibli))
            
            # Add color palette loss (specific to Ghibli style)
            # This encourages the use of Ghibli-like colors
            color_loss = self._color_palette_loss(fake_ghibli, real_ghibli)
            
            total_gen_loss = gen_loss + 10 * content_loss + 5 * color_loss
        
        # Calculate gradients and update generator
        gen_gradients = tape.gradient(total_gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        
        return total_gen_loss, gen_loss, content_loss, color_loss
    
    def _color_palette_loss(self, generated_images, reference_images):
        """Calculate color palette loss to match Ghibli style colors."""
        # Extract color histograms
        gen_colors = tf.reduce_mean(generated_images, axis=[1, 2])  # Average color per image
        ref_colors = tf.reduce_mean(reference_images, axis=[1, 2])
        
        # Calculate color distribution difference
        color_loss = tf.reduce_mean(tf.square(gen_colors - ref_colors))
        
        return color_loss
    
    @tf.function
    def train_discriminator_step(self, real_photos, real_ghibli):
        """Train discriminator for one step."""
        with tf.GradientTape() as tape:
            # Generate fake Ghibli-style images
            fake_ghibli = self.generator(real_photos, training=True)
            
            # Discriminator predictions
            real_preds = self.discriminator(real_ghibli, training=True)
            fake_preds = self.discriminator(fake_ghibli, training=True)
            
            # Calculate discriminator loss
            real_loss = tf.reduce_mean(tf.square(real_preds - 1))
            fake_loss = tf.reduce_mean(tf.square(fake_preds))
            total_disc_loss = real_loss + fake_loss
        
        # Calculate gradients and update discriminator
        disc_gradients = tape.gradient(total_disc_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))
        
        return total_disc_loss, real_loss, fake_loss
