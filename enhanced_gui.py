import tkinter as tk
from tkinter import filedialog, ttk, Scale, HORIZONTAL, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tensorflow as tf # Keep import for potential future use, though models are loaded in style_processor
from style_processor import StyleProcessor

class EnhancedCartoonGUI:
    """
    Enhanced GUI application for image transformation with multiple styles.
    Fixes issues with style selection, display, and comparison.
    """
    
    def __init__(self, root):
        """
        Initialize the GUI.
        
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("Enhanced Image Transformation (Fixed)")
        self.root.geometry("1280x900")
        self.root.configure(bg="#2c3e50")
        
        # Initialize style processor
        self.style_processor = StyleProcessor()
        
        # Initialize variables
        self.original_image = None
        self.current_image_path = None
        self.transformed_images = {} # Stores processed images {style_name: image_data}
        self.current_style_display_name = tk.StringVar(value="Classical Cartoon") # Variable for Combobox
        self.tk_images = {} # Stores PhotoImage objects for display {name: PhotoImage}
        
        # Create main frames
        self.create_frames()
        
        # Create widgets
        self.create_menu()
        self.create_image_display()
        self.create_controls()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready. Load an image to begin.")
        self.status_bar = tk.Label(self.root, textvariable=self.status_var, 
                                  bd=1, relief=tk.SUNKEN, anchor=tk.W,
                                  bg="#34495e", fg="white")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def create_frames(self):
        """Create the main frames for the GUI."""
        # Top frame for menu and buttons
        self.top_frame = tk.Frame(self.root, bg="#2c3e50")
        self.top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        # Middle frame for images
        self.middle_frame = tk.Frame(self.root, bg="#2c3e50")
        self.middle_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left frame for original image
        self.left_frame = tk.LabelFrame(self.middle_frame, text="Original Image", 
                                       bg="#34495e", fg="white", font=("Arial", 12))
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right frame for transformed image
        self.right_frame = tk.LabelFrame(self.middle_frame, text="Transformed Image", 
                                        bg="#34495e", fg="white", font=("Arial", 12))
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Bottom frame for controls
        self.bottom_frame = tk.LabelFrame(self.root, text="Transformation Controls", 
                                         bg="#34495e", fg="white", font=("Arial", 12))
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
    def create_menu(self):
        """Create the menu bar and quick access buttons."""
        # Menu bar
        self.menu_bar = tk.Menu(self.root)
        
        # File menu
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.file_menu.add_command(label="Open Image", command=self.open_image)
        self.file_menu.add_command(label="Save Transformed Image", command=self.save_transformed_image)
        self.file_menu.add_command(label="Save All Styles", command=self.save_all_styles)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.root.quit)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)
        
        # Model menu
        self.model_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.model_menu.add_command(label="Load Models", command=self.load_models)
        # self.model_menu.add_command(label="View Model Architecture", command=self.view_model_architecture) # Optional: Can be added back if needed
        self.menu_bar.add_cascade(label="Models", menu=self.model_menu)
        
        # Help menu
        self.help_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.help_menu.add_command(label="About", command=self.show_about)
        self.menu_bar.add_cascade(label="Help", menu=self.help_menu)
        
        self.root.config(menu=self.menu_bar)
        
        # Quick access buttons
        self.open_button = tk.Button(self.top_frame, text="Open Image", 
                                    command=self.open_image, bg="#3498db", fg="white",
                                    font=("Arial", 10, "bold"), padx=10, pady=5)
        self.open_button.pack(side=tk.LEFT, padx=5)
        
        self.save_button = tk.Button(self.top_frame, text="Save Image", 
                                    command=self.save_transformed_image, bg="#2ecc71", fg="white",
                                    font=("Arial", 10, "bold"), padx=10, pady=5)
        self.save_button.pack(side=tk.LEFT, padx=5)
        
        self.compare_button = tk.Button(self.top_frame, text="Compare All Styles", 
                                      command=self.compare_all_styles, bg="#9b59b6", fg="white",
                                      font=("Arial", 10, "bold"), padx=10, pady=5)
        self.compare_button.pack(side=tk.LEFT, padx=5)
        
        # Style selection
        style_frame = tk.Frame(self.top_frame, bg="#2c3e50")
        style_frame.pack(side=tk.RIGHT, padx=10)
        
        tk.Label(style_frame, text="Style:", bg="#2c3e50", fg="white", 
                font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        
        # Define styles: Display Name and internal style_name used by StyleProcessor
        self.styles_available = [
            ("Classical Cartoon", "classical_cartoon"), 
            ("Neural Cartoon", "neural_cartoon"),
            ("Ghibli Style", "ghibli"),
            ("Sketch", "sketch")
        ]
        self.style_display_names = [s[0] for s in self.styles_available]
        # Map display name back to internal style_name
        self.style_mapping = {s[0]: s[1] for s in self.styles_available}
        
        self.style_menu = ttk.Combobox(style_frame, textvariable=self.current_style_display_name, 
                                      values=self.style_display_names, width=18, state="readonly")
        self.style_menu.pack(side=tk.LEFT, padx=5)
        self.style_menu.current(0) # Default to Classical Cartoon
        self.style_menu.bind("<<ComboboxSelected>>", self.update_style)
        
    def create_image_display(self):
        """Create the image display areas (canvases)."""
        # Original image display
        self.original_canvas = tk.Canvas(self.left_frame, bg="#2c3e50", 
                                        highlightthickness=0)
        self.original_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Transformed image display
        self.transformed_canvas = tk.Canvas(self.right_frame, bg="#2c3e50", 
                                          highlightthickness=0)
        self.transformed_canvas.pack(fill=tk.BOTH, expand=True)
        
    def create_controls(self):
        """Create the control panel (currently focused on classical cartoon)."""
        # NOTE: Controls for neural styles were present but not fully functional.
        # Keeping classical controls for now. Neural controls can be added back if needed.
        
        # Create notebook for tabs (even if only one tab initially)
        self.control_notebook = ttk.Notebook(self.bottom_frame)
        self.control_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # --- Classical cartoon controls --- 
        self.classical_frame = tk.Frame(self.control_notebook, bg="#34495e")
        self.control_notebook.add(self.classical_frame, text="Classical Cartoon Parameters")
        
        # Create frames for rows to organize sliders
        classical_frames = []
        num_rows = 3
        widgets_per_row = 3 # Adjust as needed
        for i in range(num_rows):
            frame = tk.Frame(self.classical_frame, bg="#34495e")
            frame.pack(fill=tk.X, padx=10, pady=2)
            classical_frames.append(frame)
        
        # Helper function to create sliders
        def create_slider(parent, text, variable, from_, to, default, command, length=180):
            tk.Label(parent, text=text, bg="#34495e", fg="white", font=("Arial", 9)).pack(side=tk.LEFT, padx=(10, 2))
            slider = Scale(parent, from_=from_, to=to, 
                           orient=HORIZONTAL, variable=variable,
                           command=command, length=length,
                           bg="#34495e", fg="white", troughcolor="#2c3e50",
                           highlightthickness=0, relief=tk.FLAT)
            variable.set(default)
            slider.pack(side=tk.LEFT, padx=(0, 10))
            return slider

        # Row 1
        self.line_size_var = tk.IntVar()
        create_slider(classical_frames[0], "Edge Thick:", self.line_size_var, 1, 15, 7, self.update_classical_cartoon_params)
        self.bilateral_d_var = tk.IntVar()
        create_slider(classical_frames[0], "Smoothing:", self.bilateral_d_var, 5, 15, 9, self.update_classical_cartoon_params)
        self.color_levels_var = tk.IntVar()
        create_slider(classical_frames[0], "Colors:", self.color_levels_var, 2, 16, 8, self.update_classical_cartoon_params)

        # Row 2
        self.bilateral_color_var = tk.IntVar()
        create_slider(classical_frames[1], "Color Smooth:", self.bilateral_color_var, 10, 150, 75, self.update_classical_cartoon_params)
        self.bilateral_space_var = tk.IntVar()
        create_slider(classical_frames[1], "Spatial Smooth:", self.bilateral_space_var, 10, 150, 75, self.update_classical_cartoon_params)

        # Row 3
        self.edge_t1_var = tk.IntVar()
        create_slider(classical_frames[2], "Edge Thresh 1:", self.edge_t1_var, 10, 100, 50, self.update_classical_cartoon_params)
        self.edge_t2_var = tk.IntVar()
        create_slider(classical_frames[2], "Edge Thresh 2:", self.edge_t2_var, 50, 200, 150, self.update_classical_cartoon_params)
        
        # --- Placeholder for Neural controls (can be added later) ---
        # self.neural_frame = tk.Frame(self.control_notebook, bg="#34495e")
        # self.control_notebook.add(self.neural_frame, text="Neural Network Parameters")
        # tk.Label(self.neural_frame, text="Neural controls (e.g., weights) would go here.", bg="#34495e", fg="white").pack(pady=20)

    def open_image(self):
        """Open an image file, display it, and apply the default transformation."""
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.current_image_path = file_path
                self.status_var.set(f"Loading: {os.path.basename(file_path)}")
                self.root.update_idletasks()
                
                # Load the image using OpenCV (handles various formats)
                img_bgr = cv2.imread(file_path)
                if img_bgr is None:
                    raise ValueError("Could not read image file.")
                self.original_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                
                # Clear previous results
                self.transformed_images = {}
                self.tk_images = {}
                
                # Display the original image
                self.display_image(self.original_image, self.original_canvas, "original")
                
                # Process and display the currently selected transformation
                self.process_current_style()
                self.status_var.set(f"Opened: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error Opening Image", f"Failed to open or process image:\n{e}")
                self.status_var.set("Error loading image.")
                self.original_image = None
                self.current_image_path = None
                self.clear_canvases()

    def display_image(self, img_rgb, canvas, tk_image_key, max_width=600, max_height=500):
        """Display an image (numpy array RGB) on a given canvas."""
        if img_rgb is None:
            canvas.delete("all") # Clear canvas if image is None
            return
            
        try:
            # Resize to fit the canvas area while maintaining aspect ratio
            img_resized = self.resize_image_for_display(img_rgb, max_width, max_height)
            
            # Convert numpy array (RGB) to PhotoImage
            img_pil = Image.fromarray(img_resized)
            self.tk_images[tk_image_key] = ImageTk.PhotoImage(image=img_pil)
            
            # Update canvas size and display image
            canvas.config(width=self.tk_images[tk_image_key].width(), 
                          height=self.tk_images[tk_image_key].height())
            canvas.delete("all") # Clear previous image
            canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_images[tk_image_key])
        except Exception as e:
             print(f"Error displaying image {tk_image_key}: {e}")
             messagebox.showerror("Display Error", f"Could not display image {tk_image_key}.\n{e}")
             canvas.delete("all")

    def resize_image_for_display(self, img, max_width, max_height):
        """Resize image for display, maintaining aspect ratio."""
        height, width = img.shape[:2]
        if width == 0 or height == 0:
             return img # Avoid division by zero
             
        scale = min(max_width / width, max_height / height, 1.0) # Don't scale up
        
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            # Use INTER_AREA for shrinking
            resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        else:
            resized = img # No resizing needed
            
        return resized

    def process_current_style(self):
        """Process the image using the currently selected style from the dropdown."""
        if self.original_image is None:
            # messagebox.showwarning("No Image", "Please open an image first.")
            self.status_var.set("Load an image first.")
            return

        selected_display_name = self.current_style_display_name.get()
        style_name = self.style_mapping.get(selected_display_name)

        if not style_name:
            messagebox.showerror("Error", "Invalid style selected.")
            return

        self.status_var.set(f"Processing {selected_display_name}...")
        self.root.update_idletasks() # Update GUI to show status

        try:
            transformed_img = None
            # Apply transformation based on style_name
            if style_name == "classical_cartoon":
                transformed_img = self.style_processor.apply_classical_cartoon(
                    img=self.original_image,
                    line_size=self.line_size_var.get(),
                    bilateral_d=self.bilateral_d_var.get(),
                    bilateral_sigma_color=self.bilateral_color_var.get(),
                    bilateral_sigma_space=self.bilateral_space_var.get(),
                    edge_threshold1=self.edge_t1_var.get(),
                    edge_threshold2=self.edge_t2_var.get(),
                    total_color_levels=self.color_levels_var.get()
                )
            elif style_name == "neural_cartoon":
                transformed_img = self.style_processor.apply_cartoon_style(img=self.original_image)
            elif style_name == "ghibli":
                transformed_img = self.style_processor.apply_ghibli_style(img=self.original_image)
            elif style_name == "sketch":
                transformed_img = self.style_processor.apply_sketch_style(img=self.original_image)
            
            if transformed_img is not None:
                self.transformed_images[style_name] = transformed_img
                self.display_image(transformed_img, self.transformed_canvas, "transformed")
                self.status_var.set(f"Displayed {selected_display_name}")
            else:
                 raise ValueError("Transformation function returned None.")
                 
        except Exception as e:
            messagebox.showerror("Transformation Error", f"Failed to apply {selected_display_name}:\n{e}")
            self.status_var.set(f"Error processing {selected_display_name}.")
            # Optionally display original image or clear canvas on error
            self.display_image(None, self.transformed_canvas, "transformed")

    def update_classical_cartoon_params(self, event=None):
        """Called when classical cartoon sliders change. Reprocess ONLY if classical cartoon is selected."""
        selected_display_name = self.current_style_display_name.get()
        style_name = self.style_mapping.get(selected_display_name)
        
        if style_name == "classical_cartoon" and self.original_image is not None:
            # Only re-process if the classical style is currently selected
            self.process_current_style()
            
    def update_style(self, event=None):
        """Called when a new style is selected from the dropdown. Process and display it."""
        # The process_current_style method now handles applying the selected style
        self.process_current_style()
        
        # Update active control tab (optional, can be refined)
        selected_display_name = self.current_style_display_name.get()
        style_name = self.style_mapping.get(selected_display_name)
        if style_name == "classical_cartoon":
            self.control_notebook.select(0) # Select classical tab
        # else:
            # If neural controls tab exists, select it
            # try:
            #     self.control_notebook.select(1)
            # except tk.TclError: # Tab might not exist
            #     pass 

    def save_transformed_image(self):
        """Save the currently displayed transformed image."""
        selected_display_name = self.current_style_display_name.get()
        style_name = self.style_mapping.get(selected_display_name)
        
        if style_name in self.transformed_images:
            img_to_save = self.transformed_images[style_name]
            
            # Suggest a filename based on original and style
            original_basename = "image"
            if self.current_image_path:
                original_basename = os.path.splitext(os.path.basename(self.current_image_path))[0]
            suggested_filename = f"{original_basename}_{style_name}.jpg"
            
            file_path = filedialog.asksaveasfilename(
                title="Save Transformed Image As",
                initialfile=suggested_filename,
                defaultextension=".jpg",
                filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("BMP files", "*.bmp"), ("All files", "*.*")]
            )
            
            if file_path:
                try:
                    # Convert RGB to BGR for OpenCV imwrite
                    img_bgr = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)
                    success = cv2.imwrite(file_path, img_bgr)
                    if success:
                        self.status_var.set(f"Saved: {os.path.basename(file_path)}")
                    else:
                         raise IOError("cv2.imwrite failed to save the image.")
                except Exception as e:
                    messagebox.showerror("Save Error", f"Failed to save image:\n{e}")
                    self.status_var.set("Error saving image.")
        else:
            messagebox.showwarning("No Image", "No transformed image available to save for the selected style.")
            self.status_var.set("No transformed image to save.")

    def save_all_styles(self):
        """Process (if needed) and save all available styles to a selected directory."""
        if self.original_image is None:
            messagebox.showwarning("No Image", "Please open an image first.")
            self.status_var.set("Load an image first.")
            return
            
        dir_path = filedialog.askdirectory(title="Select Directory to Save All Styles")
        
        if dir_path:
            self.status_var.set("Processing and saving all styles...")
            self.root.update_idletasks()
            
            saved_count = 0
            error_count = 0
            
            # Define styles to save (matches dropdown)
            styles_to_process = self.style_mapping.values()

            # Process and save each style
            for style_name in styles_to_process:
                try:
                    # Process if not already done (or re-process classical with current params)
                    if style_name not in self.transformed_images or style_name == "classical_cartoon":
                        print(f"Processing {style_name} for saving...") # Debug print
                        if style_name == "classical_cartoon":
                            img_to_save = self.style_processor.apply_classical_cartoon(
                                img=self.original_image,
                                line_size=self.line_size_var.get(),
                                bilateral_d=self.bilateral_d_var.get(),
                                bilateral_sigma_color=self.bilateral_color_var.get(),
                                bilateral_sigma_space=self.bilateral_space_var.get(),
                                edge_threshold1=self.edge_t1_var.get(),
                                edge_threshold2=self.edge_t2_var.get(),
                                total_color_levels=self.color_levels_var.get()
                            )
                        elif style_name == "neural_cartoon":
                            img_to_save = self.style_processor.apply_cartoon_style(img=self.original_image)
                        elif style_name == "ghibli":
                            img_to_save = self.style_processor.apply_ghibli_style(img=self.original_image)
                        elif style_name == "sketch":
                            img_to_save = self.style_processor.apply_sketch_style(img=self.original_image)
                        else:
                             continue # Should not happen with current setup
                             
                        self.transformed_images[style_name] = img_to_save # Store processed image
                    else:
                        img_to_save = self.transformed_images[style_name]

                    # Determine filename
                    original_basename = os.path.splitext(os.path.basename(self.current_image_path))[0] if self.current_image_path else "image"
                    output_filename = f"{original_basename}_{style_name}.jpg"
                    output_path = os.path.join(dir_path, output_filename)
                    
                    # Save the image (convert to BGR)
                    img_bgr = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)
                    success = cv2.imwrite(output_path, img_bgr)
                    if success:
                        saved_count += 1
                    else:
                         print(f"Failed to save {style_name} using cv2.imwrite.")
                         error_count += 1
                         
                except Exception as e:
                    print(f"Error processing or saving {style_name}: {e}")
                    error_count += 1
            
            # Optionally save the original image too
            try:
                 original_basename = os.path.splitext(os.path.basename(self.current_image_path))[0] if self.current_image_path else "image"
                 output_path = os.path.join(dir_path, f"{original_basename}_original.jpg")
                 img_bgr = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2BGR)
                 success = cv2.imwrite(output_path, img_bgr)
                 if success: saved_count += 1
                 else: error_count += 1
            except Exception as e:
                 print(f"Error saving original image: {e}")
                 error_count += 1
                 
            final_message = f"Saved {saved_count} images to: {dir_path}"
            if error_count > 0:
                final_message += f" ({error_count} errors occurred). Check console for details."
                messagebox.showwarning("Save All Warning", f"Finished saving, but {error_count} errors occurred. Check console output.")
            else:
                 messagebox.showinfo("Save All Complete", f"Successfully saved {saved_count} images.")
                 
            self.status_var.set(final_message)

    def compare_all_styles(self):
        """Show a comparison window with original and all available styles."""
        if self.original_image is None:
            messagebox.showwarning("No Image", "Please open an image first.")
            self.status_var.set("Load an image first.")
            return
            
        self.status_var.set("Processing all styles for comparison...")
        self.root.update_idletasks()
        
        # Prepare dictionary to hold images for comparison
        comparison_images = {
            'Original': self.original_image
        }
        
        # Define styles to compare (matches dropdown + original)
        styles_to_compare = self.style_mapping.values()
        display_names_map = {v: k for k, v in self.style_mapping.items()} # Map internal name back to display name

        # Process each style for comparison
        for style_name in styles_to_compare:
            try:
                # Process if not already done (or re-process classical with current params)
                if style_name not in self.transformed_images or style_name == "classical_cartoon":
                    print(f"Processing {style_name} for comparison...") # Debug print
                    if style_name == "classical_cartoon":
                        img_to_compare = self.style_processor.apply_classical_cartoon(
                            img=self.original_image,
                            line_size=self.line_size_var.get(),
                            bilateral_d=self.bilateral_d_var.get(),
                            bilateral_sigma_color=self.bilateral_color_var.get(),
                            bilateral_sigma_space=self.bilateral_space_var.get(),
                            edge_threshold1=self.edge_t1_var.get(),
                            edge_threshold2=self.edge_t2_var.get(),
                            total_color_levels=self.color_levels_var.get()
                        )
                    elif style_name == "neural_cartoon":
                        img_to_compare = self.style_processor.apply_cartoon_style(img=self.original_image)
                    elif style_name == "ghibli":
                        img_to_compare = self.style_processor.apply_ghibli_style(img=self.original_image)
                    elif style_name == "sketch":
                        img_to_compare = self.style_processor.apply_sketch_style(img=self.original_image)
                    else:
                         continue
                         
                    self.transformed_images[style_name] = img_to_compare # Store processed image
                else:
                    img_to_compare = self.transformed_images[style_name]
                
                # Add to comparison dictionary with the display name
                display_name = display_names_map.get(style_name, style_name.replace("_", " ").title())
                comparison_images[display_name] = img_to_compare
                
            except Exception as e:
                print(f"Error processing {style_name} for comparison: {e}")
                # Optionally add a placeholder or skip the style in comparison
                comparison_images[f"{display_name} (Error)"] = np.zeros_like(self.original_image) # Placeholder

        # --- Create Comparison Window --- 
        comparison_window = tk.Toplevel(self.root)
        comparison_window.title("Style Comparison")
        comparison_window.geometry("1200x800")
        comparison_window.configure(bg="#2c3e50")
        
        num_images = len(comparison_images)
        if num_images == 0: 
             messagebox.showerror("Error", "No images available for comparison.")
             comparison_window.destroy()
             return
             
        # Determine grid layout (e.g., 2 rows, up to 3 columns)
        cols = min(num_images, 3)
        rows = (num_images + cols - 1) // cols
        
        fig = plt.Figure(figsize=(4 * cols, 4 * rows), dpi=100)
        fig.patch.set_facecolor("#34495e") # Match background
        
        axes = fig.subplots(rows, cols)
        # If single row/col, axes might not be an array, flatten handles this
        axes = np.array(axes).flatten()
        
        for i, (title, img) in enumerate(comparison_images.items()):
            if i < len(axes):
                ax = axes[i]
                try:
                    ax.imshow(img)
                    ax.set_title(title, color="white", fontsize=10)
                except Exception as display_err:
                     print(f"Error displaying {title} in comparison: {display_err}")
                     ax.set_title(f"{title}\n(Display Error)", color="red", fontsize=10)
                ax.axis("off")
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
             axes[j].axis("off")
             
        fig.tight_layout(pad=1.5)
        
        # Embed Matplotlib figure in Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=comparison_window)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        canvas.draw()
        
        # Add save button for the comparison figure
        save_frame = tk.Frame(comparison_window, bg="#2c3e50")
        save_frame.pack(pady=10)
        
        save_comparison_button = tk.Button(save_frame, text="Save Comparison Figure", 
                                         command=lambda f=fig: self.save_comparison_figure(f),
                                         bg="#2ecc71", fg="white",
                                         font=("Arial", 10, "bold"), padx=10, pady=5)
        save_comparison_button.pack()
        
        self.status_var.set("Comparison view generated.")

    def save_comparison_figure(self, fig):
        """Save the comparison figure generated by Matplotlib."""
        original_basename = "comparison"
        if self.current_image_path:
            original_basename = os.path.splitext(os.path.basename(self.current_image_path))[0]
        suggested_filename = f"{original_basename}_style_comparison.png"
        
        file_path = filedialog.asksaveasfilename(
            title="Save Comparison Figure As",
            initialfile=suggested_filename,
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                fig.savefig(file_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
                self.status_var.set(f"Saved comparison figure to: {os.path.basename(file_path)}")
                messagebox.showinfo("Save Complete", f"Comparison figure saved to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save comparison figure:\n{e}")
                self.status_var.set("Error saving comparison figure.")

    def load_models(self):
        """Ask user for models directory and load them using StyleProcessor."""
        dir_path = filedialog.askdirectory(title="Select Directory Containing Model Files (.h5)")
        
        if dir_path:
            self.status_var.set("Loading models...")
            self.root.update_idletasks()
            
            try:
                load_status = self.style_processor.load_models(dir_path)
                message = load_status.get("message", "Model loading process completed.")
                
                # Show detailed status message
                messagebox.showinfo("Model Loading Status", message)
                
                if self.style_processor.models_loaded:
                    self.status_var.set("Neural models loaded (or partially loaded). Check details.")
                    # If an image is loaded, re-process the current style to use the new model if applicable
                    if self.original_image is not None:
                         self.process_current_style()
                else:
                    self.status_var.set("No neural models loaded. Using fallbacks.")
                    # If an image is loaded, ensure the fallback is displayed
                    if self.original_image is not None:
                         self.process_current_style()
                         
            except Exception as e:
                error_msg = f"An unexpected error occurred during model loading:\n{e}"
                messagebox.showerror("Model Loading Error", error_msg)
                self.status_var.set("Error loading models.")
                print(error_msg) # Also print to console

    def show_about(self):
        """Display information about the application."""
        about_text = ("Enhanced Image Transformation GUI (Fixed)\n\n" 
                      "Applies various artistic styles to images, including:\n" 
                      "- Classical Cartoon (algorithmic)\n" 
                      "- Neural Cartoon (requires model)\n" 
                      "- Ghibli Style (requires model)\n" 
                      "- Sketch (requires model)\n\n" 
                      "Uses fallback methods if neural models are not loaded.\n" 
                      "(Based on original project structure)")
        messagebox.showinfo("About", about_text)

    def clear_canvases(self):
         """Clear both image canvases."""
         self.original_canvas.delete("all")
         self.transformed_canvas.delete("all")
         self.tk_images = {}

# Main execution block
if __name__ == "__main__":
    root = tk.Tk()
    app = EnhancedCartoonGUI(root)
    root.mainloop()

