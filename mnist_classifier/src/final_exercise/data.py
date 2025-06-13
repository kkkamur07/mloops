
# imports
import torch
from torchvision import datasets, transforms

test_images = torch.load("data/processed/test_images.pt")
test_labels = torch.load("data/processed/test_target.pt")
train_labels = torch.load("data/processed/train_target.pt")
train_images = torch.load("data/processed/train_images.pt")

# Making the dataset better

def get_data_loaders(test_images = test_images, test_labels = test_labels, train_images = train_images, train_labels = train_labels):
    # Define the transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Float and Long
    train_images = train_images.unsqueeze(1).float()
    train_labels = train_labels.long()
    test_images = test_images.unsqueeze(1).float()
    test_labels = test_labels.long()

    # Create the datasets
    train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)

    # Create the data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader

if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders(test_images, test_labels, train_images, train_labels)
    print("Data loaders created successfully.")
