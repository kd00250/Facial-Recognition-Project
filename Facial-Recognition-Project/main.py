"""
main.py

This is the main entry point for the Facial Recognition Project.
It performs the following steps:
1. Extracts and arranges the dataset from a ZIP file.
2. Loads the dataset using data transformations and splits it into training and validation sets.
3. Initializes the LightCNN model for facial recognition.
4. Sets up the loss function, optimizer, and learning rate scheduler.
5. Trains the model using a custom training loop (ModelTrainer).
6. Performs real-time inference using the laptop's webcam.

To run this file in Google Colab (or locally), simply execute:
    !python Facial-Recognition-Project/main.py
Ensure that all required modules (data_preprocessing, data_loading, model_training, inference, and new_model)
are located in the same project folder along with Faces.zip.
"""

import sys
import os
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

from webcam_app import WebcamApp
# Tkinter imports (for GUI)
import tkinter as tk
from tkinter import Label, Button, Frame

# Add the current directory to the Python path (ensures module imports work correctly)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules from the project
from data_preprocessing import extract_and_arrange  # Handles extraction and arrangement of the dataset
from data_loading import get_data_loaders            # Loads the dataset and creates DataLoaders
from model_training import ModelTrainer              # Contains the training loop implementation
from inference import InferenceEngine                # Contains functions for image and webcam inference
from new_model import LightCNN                       # New CNN architecture (LightCNN) for facial recognition

def main():
    # --- Step 1: Data Preprocessing ---
    # Extract the Faces.zip file and arrange images into class-based folders.
    extract_and_arrange(zip_file="Faces.zip", extracted_folder="Faces_unorganized", structured_folder="Faces")
    
    # --- Step 2: Data Loading ---
    # Define parameters for the DataLoader and load the dataset.
    batch_size = 32
    image_size = (128, 128)
    train_dataset, val_dataset, train_loader, val_loader, class_names = get_data_loaders(
        data_dir="Faces", batch_size=batch_size, image_size=image_size
    )
    
    # --- Step 3: Model Initialization ---
    # Set the computation device (GPU if available, else CPU).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize the LightCNN model with the number of classes equal to the length of class_names.
    model = LightCNN(num_classes=len(class_names), in_channels=3)
    model = model.to(device)
    
    # --- Step 4: Loss, Optimizer, and Scheduler Setup ---
    # Define the loss function (CrossEntropyLoss) for classification.
    criterion = nn.CrossEntropyLoss()
    # Use Adam optimizer with a low learning rate and weight decay for fine-tuning.
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    # Initialize a OneCycleLR scheduler that adjusts the learning rate every iteration.
    scheduler = OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_loader), epochs=30)
    
    # --- Step 5: Training ---
    # Create an instance of ModelTrainer with the defined components and start training.
    trainer = ModelTrainer(model, train_loader, val_loader, train_dataset, val_dataset, device, criterion, optimizer, scheduler)
    trainer.train(num_epochs=30, early_stop_patience=10)
    
    # --- Step 6: Webcam Inference ---
    root = tk.Tk()
    app = WebcamApp(root, model, device, class_names, image_size)
    root.mainloop()

if __name__ == "__main__":
    main()
