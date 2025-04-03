import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_data_loaders(data_dir="Faces", batch_size=32, image_size=(128,128)):
    # Define transformations: resize, convert to tensor, and normalize.
    data_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Load the dataset from the arranged folder
    dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
    print("Classes found:", dataset.classes)
    
    # Split the dataset into training and validation sets (80-20 split)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Return datasets, loaders and class names for later use
    return train_dataset, val_dataset, train_loader, val_loader, dataset.classes
