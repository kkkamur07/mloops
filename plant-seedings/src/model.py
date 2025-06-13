import torch
from torch import nn
import os

class BaselineCNN(nn.Module):
    def __init__(self, num_classes, lr=0.001, epochs=100):
        super().__init__()

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Model architecture stays the same
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 512, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes),
            nn.LogSoftmax(dim=1)
        )

        self.model = nn.Sequential(self.features, self.classifier)

        self.lr = lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.criterion = nn.NLLLoss()
        self.epochs = epochs

        # Move model to device
        self.to(self.device)

        # Information storage
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def forward(self, x):
        return self.model(x)

    def train_epoch(self, train_loader):
        self.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            # Move data to device
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct / total
        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_accuracy)

        return epoch_loss, epoch_accuracy

    def validate_epoch(self, val_loader):
        self.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_accuracy = correct / total
        self.val_losses.append(epoch_loss)
        self.val_accuracies.append(epoch_accuracy)

        return epoch_loss, epoch_accuracy

    def train_model(self, train_loader, val_loader, save_dir='models', verbose=True):
        # Create directory for saving models if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        for epoch in range(self.epochs):
            try:
                train_loss, train_accuracy = self.train_epoch(train_loader)
                val_loss, val_accuracy = self.validate_epoch(val_loader)

                if verbose:
                    print(f'Epoch {epoch+1}/{self.epochs}, '
                        f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
                        f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

                torch.save(self.model.state_dict(), f'{save_dir}/baseline_cnn_epoch{epoch+1}.pth')
                print(f'Model saved for epoch {epoch+1}')

            except Exception as e:
                print(f"Error during training: {e}")
                break

    def predict(self, images):
        """Generate predictions for a batch of images"""
        self.eval()
        with torch.no_grad():
            images = images.to(self.device)
            outputs = self(images)
            _, predicted = torch.max(outputs, 1)
        return predicted
