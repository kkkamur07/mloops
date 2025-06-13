
# imports
import torch
from torch import nn, optim
import typer
from data import get_data_loaders
from hydra import main as hydra_main
from omegaconf import DictConfig
import logging
import wandb

log = logging.getLogger(__name__)
fh = logging.FileHandler('mnist.log')
fh.setLevel(logging.INFO)
log.addHandler(fh)

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
            
            wandb.log({
                "batch_loss" : loss.item(),
                "batch_accuracy" : (output.argmax(dim=1) == labels).float().mean().item()
            })
        
        avg_loss = running_loss / len(trainloader)
        self.train_loss.append(avg_loss)
        wandb.log({
            "train_loss" : avg_loss,
        })
        log.info(f"Training loss: {avg_loss:.4f}")
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
        wandb.log({
            "validation_loss" : avg_loss,
            "validation_accuracy" : accuracy
        })
        return avg_loss, accuracy
    
    def train_model(self, trainloader, testloader) : 
        for epoch in range(self.epoch) :
            log.info(f"Epoch {epoch+1}/{self.epoch}")
            
            train_loss = self.train_epoch(trainloader)
            test_loss, accuracy = self.validation(testloader)
            
            if (epoch + 1) % 5 == 0:
                torch.save(self.model.state_dict(), f"model_epoch_{epoch+1}.pth")
                log.info(f"Model saved at epoch {epoch+1}")
            
            log.info(f"Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")
            log.info("-" * 50)
            
# CLI app
app = typer.Typer()

# @hydra_main(config_path="config", config_name="conf")
# def main(
#     epochs: int = typer.Option(10, "--epochs", "-e", help="Number of training epochs"),
#     lr: float = typer.Option(0.001, "--lr", "-l", help="Learning rate"),
#     show_architecture: bool = typer.Option(False, "--show-architecture", "-a", help="Show model architecture")
# ) : 

#     train_loader, test_loader = get_data_loaders()
#     model = MNISTModel(epoch=epochs, lr=lr)
    
#     if show_architecture:
#         log.info(f"Model Architecture:\n{model.model}")
#         log.info(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")
    
#     model.train_model(train_loader, test_loader)
    
#     log.info(f"Model trained successfully with {epochs} epochs and learning rate {lr}.")


# if __name__ == "__main__":
#     app()
    
@hydra_main(config_path="../../configs", config_name="conf")
def main(cfg: DictConfig):
    # Access params via cfg
    epochs = cfg.epochs
    lr = cfg.lr
    show_architecture = cfg.show_architecture
    batch_size = cfg.get("batch_size", 32)  # Default batch size if not specified


    # Initialize 
    run = wandb.init(
        project="mnist-classifier",
        config={
            "epochs": epochs,
            "learning_rate": lr,
            "batch_size": batch_size
        }
    )
    
    train_loader, test_loader = get_data_loaders()
    model = MNISTModel(epoch=epochs, lr=lr)
    
    if show_architecture:
        log.info(f"Model Architecture:\n{model.model}")
        log.info(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")
    
    model.train_model(train_loader, test_loader)
    
    log.info(f"Model trained with {epochs} epochs and lr {lr}.")
    
    artifact = wandb.Artifact(
        name="mnist_model", 
        type="model",
        description="MNIST classifier model",
    )
    artifact.add_file("model_epoch_10.pth") 
    
    run.log_artifact(artifact)  # Log the artifact to W&B
    log.info("Model artifact logged to W&B.")
    
    wandb.finish()  # Finish the W&B run
    
    

if __name__ == "__main__":
    main()