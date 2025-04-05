import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_data_loaders(data_dir="Faces", batch_size=32, image_size=(128,128)):
    """
    Loads an image dataset from a specified directory, applies transformations,
    splits the dataset into training and validation sets, and returns DataLoaders.

    Args:
        data_dir (str): Directory containing the arranged image dataset.
                        The dataset should follow the ImageFolder structure.
        batch_size (int): Number of samples per batch.
        image_size (tuple): Desired image size (width, height) to which images will be resized.

    Returns:
        train_dataset (Subset): The training subset of the dataset.
        val_dataset (Subset): The validation subset of the dataset.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        classes (list): List of class names inferred from the dataset's folder names.
    """
    # Determine the absolute path to the data directory based on this file's location.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, data_dir)
    
    # Define a transformation pipeline to:
    # 1. Resize images to the given image_size.
    # 2. Convert images to tensors.
    # 3. Normalize image pixel values using mean and std for 3 channels.
    data_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Load the dataset using ImageFolder.
    # This assumes that the dataset folder is organized by class, with one subfolder per class.
    dataset = datasets.ImageFolder(root=data_path, transform=data_transforms)
    print("Classes found:", dataset.classes)
    
    # Split the dataset into training (80%) and validation (20%) sets.
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Create DataLoaders for the training and validation subsets.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Return the training and validation datasets, DataLoaders, and class names.
    return train_dataset, val_dataset, train_loader, val_loader, dataset.classes
