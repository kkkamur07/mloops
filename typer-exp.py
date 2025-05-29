import pickle
from typing import Annotated

import typer
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

app = typer.Typer()
train_app = typer.Typer()
color = typer.Typer()
app.add_typer(train_app, name="train")
app.add_typer(color, name="color")

# Load the dataset
data = load_breast_cancer()
x = data.data
y = data.target

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# This is how we can define the subcommands. 
@train_app.command()
def svm(kernel: str = "linear", output_file: Annotated[str, typer.Option("--output", "-o")] = "model.ckpt") -> None:
    """Train a SVM model."""
    model = SVC(kernel=kernel, random_state=42)
    model.fit(x_train, y_train)

    with open(output_file, "wb") as f:
        pickle.dump(model, f)


@train_app.command()
def knn(n_neighbors: int = 5, output_file: Annotated[str, typer.Option("--output", "-o")] = "model.ckpt") -> None:
    """Train a KNN model."""
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(x_train, y_train)

    with open(output_file, "wb") as f:
        pickle.dump(model, f)


@app.command()
def evaluate(model_file = "model.ckpt") -> None:
    """Evaluate the model."""
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    # Make predictions on the test set
    y_pred = model.predict(x_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)
    return accuracy, report


from rich import print
from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown

data = {
    "name": "Rick",
    "age": 42,
    "items": [{"name": "Portal Gun"}, {"name": "Plumbus"}],
    "active": True,
    "affiliation": None,
}

# Damn this is how you add colors
@color.command()
def main():
    print("Here's the data")
    print(data)
    print("[bold red]Alert![/bold red] [green]Portal gun[/green] shooting! :boom:")
    
MARKDOWN = """
# This is an h1

Rich can do a pretty *decent* job of rendering markdown.

1. This is a list item
2. This is another list item
"""

console = Console()
@color.command()
def table():
    table = Table("Name", "Item")
    table.add_row("Rick", "Portal Gun")
    table.add_row("Morty", "Plumbus")
    console.print(table)
    console.print(Markdown(MARKDOWN), markup=True)

    
# It also has other features like progress bars, prompt, markdown, panel

# Progress bars
import time
from rich.progress import Progress

@color.command()
def pros() : 
    with Progress() as progress:
        task1 = progress.add_task("[cyan]Processing...", total=100)
        task2 = progress.add_task("[magenta]Loading...", total=50)

        while not progress.finished:
            progress.update(task1, advance=0.5)
            progress.update(task2, advance=0.2)
            time.sleep(0.01)



if __name__ == "__main__":
    app()