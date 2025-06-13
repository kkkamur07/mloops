import typer
from data import BrainTumorDataset
from model import SimpleFCNN
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
import os

app = typer.Typer()
console = Console()

@app.command()
def main(
    data_dir: str = typer.Option("data", "-d", help="Path to the dataset directory"),
    batch_size: int = typer.Option(16, "-b", help="Batch size for training"),
    learning_rate: float = typer.Option(0.001, "-lr", help="Learning rate for the optimizer"),
    epochs: int = typer.Option(10, "-e", help="Number of epochs for training"),
):
    console.print("[bold green]Starting Brain Tumor Segmentation Training[/bold green]")

    train_dataset = BrainTumorDataset(data_dir, "train")
    val_dataset = BrainTumorDataset(data_dir, "valid")

    train_loader = train_dataset.create_dataloader(batch_size=batch_size, shuffle=True)
    val_loader = val_dataset.create_dataloader(batch_size=batch_size, shuffle=False)

    model = SimpleFCNN(lr=learning_rate, epochs=epochs)

    table = Table("Epoch", "Train Loss", "Val Loss", title="Training Progress")

    with Progress() as progress:
        epoch_task = progress.add_task("[cyan]Epochs", total=epochs)
        for epoch in range(epochs):
            train_loss = model.train_step(train_loader)
            val_loss = model.evaluate_step(val_loader)
            progress.update(epoch_task, advance=1)
            table.add_row(str(epoch + 1), f"{train_loss:.4f}", f"{val_loss:.4f}")
            console.print(table)

    model.save_model()
    console.print("[bold green]Model saved successfully![/bold green]")
    console.print("[bold green]Training completed![/bold green]")


if __name__ == "__main__":
    app()
