
# imports
import torch
from torch import nn, optim
import typer
from data import get_data_loaders

class MNISTModel(nn.Module) : 
    
    def __init__(self, epoch = 10, lr = 0.001) : 
        super().__init__()
        
        self.dropout = nn.Dropout(p=0.2)
        
        self.model = nn.Sequential(
            nn.Linear(784, 256),
            self.dropout,
            nn.ReLU(),
            nn.Linear(256, 128),
            self.dropout,
            nn.ReLU(),
            nn.Linear(128, 64),
            self.dropout,
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.LogSoftmax(dim=1)
        )
        self.epoch = epoch
        self.lr = lr
        
        self.criterion = nn.NLLLoss() 
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        self.train_loss = []
        self.test_loss = []
        
        
    def forward(self, image) : 
        image = image.view(image.shape[0], -1)
        output = self.model(image)
        return output
    
    # training the model
    def train_epoch(self, trainloader) : 
        self.model.train() # setting the model to training mode
        running_loss = 0
        
        for images, labels in trainloader :
            self.optimizer.zero_grad()
            output = self.forward(images)
            loss = self.criterion(output, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(trainloader)
        self.train_loss.append(avg_loss)
        print(f"Training loss: {avg_loss:.4f}")
        return avg_loss
    
    # testing the model
    
    def validation(self, testloader) : 
        self.model.eval()
        running_loss = 0
        correct = 0
        total = 0
        
        for images, labels in testloader :
            with torch.no_grad() :
                output = self.forward(images)
                loss = self.criterion(output, labels)
                running_loss += loss.item()
                
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(testloader)
        self.test_loss.append(avg_loss)
        accuracy = 100 * correct / total
        return avg_loss, accuracy
    
    def train_model(self, trainloader, testloader) : 
        for epoch in range(self.epoch) :
            print(f"Epoch {epoch+1}/{self.epoch}")
            
            train_loss = self.train_epoch(trainloader)
            test_loss, accuracy = self.validation(testloader)
            
            if (epoch + 1) % 5 == 0:
                torch.save(self.model.state_dict(), f"models/model_epoch_{epoch+1}.pth")
                print(f"Model saved at epoch {epoch+1}")
            
            print(f"Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")
            print("-" * 50)
            
# CLI app
app = typer.Typer()

@app.command()
def main(
    epochs: int = typer.Option(10, "--epochs", "-e", help="Number of training epochs"),
    lr: float = typer.Option(0.001, "--lr", "-l", help="Learning rate"),
    show_architecture: bool = typer.Option(False, "--show-architecture", "-a", help="Show model architecture")
) : 

    train_loader, test_loader = get_data_loaders()
    model = MNISTModel(epoch=epochs, lr=lr)
    
    if show_architecture:
        print(f"Model Architecture:\n{model.model}")
        print(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")
    
    model.train_model(train_loader, test_loader)
    
    print(f"Model trained successfully with {epochs} epochs and learning rate {lr}.")


if __name__ == "__main__":
    app()