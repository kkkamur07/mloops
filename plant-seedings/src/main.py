import typer
from src.data import get_dataloaders
from src.model import BaselineCNN
import os

app = typer.Typer()

@app.command()
def train(
    data_path: str = typer.Option(..., "-d", "--data", help="Path to the data directory with class folders"),
    val_split: float = typer.Option(0.2, "-v", "--val-split", help="Validation split ratio"),
    batch_size: int = typer.Option(32, "-b", "--batch-size", help="Batch size for training"),
    lr: float = typer.Option(0.001, "-l", "--learning-rate", help="Learning rate"),
    epochs: int = typer.Option(10, "-e", "--epochs", help="Number of training epochs"),
    save_dir: str = typer.Option("models", "-s", "--save-dir", help="Directory to save model checkpoints")
):
    """Train the plant seedlings classifier end-to-end."""

    # Step 1: Load and prepare data
    print(f"\n=== Data Loading ===")
    train_loader, val_loader, classes = get_dataloaders(
        data_path=data_path,
        val_split=val_split,
        batch_size=batch_size
    )

    if not train_loader or not val_loader:
        print("Error: Failed to load datasets. Exiting.")
        return

    # Step 2: Create model
    print(f"\n=== Model Creation ===")
    model = BaselineCNN(
        num_classes=len(classes),
        lr=lr,
        epochs=epochs
    )

    # Step 3: Train model
    print(f"\n=== Training Started ===")
    os.makedirs(save_dir, exist_ok=True)
    model.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=save_dir,
        verbose=True
    )

    print(f"\n=== Training Complete ===")
    print(f"Model checkpoints saved to {save_dir}/")

if __name__ == "__main__":
    app()
