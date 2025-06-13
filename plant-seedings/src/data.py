import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(data_path, val_split=0.2, batch_size=32):
    """
    Load the dataset and return dataloaders for training and validation.

    Args:
        data_path (str): Path to the data directory with class folders
        val_split (float): Validation split ratio (0.0 to 1.0)
        batch_size (int): Batch size for dataloaders

    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        classes: List of class names
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load dataset using ImageFolder
    try:
        full_dataset = datasets.ImageFolder(root=data_path, transform=transform)
        print(f"Dataset loaded with {len(full_dataset)} samples and {len(full_dataset.classes)} classes.")
        classes = full_dataset.classes
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None, None

    # Split into train and validation
    val_size = int(val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Split dataset: {train_size} training samples, {val_size} validation samples")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    print(f"Created DataLoaders with batch size {batch_size}")
    return train_loader, val_loader, classes
