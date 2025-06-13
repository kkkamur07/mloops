import torch
import torch.nn as nn
import numpy as np

class SimpleFCNN(nn.Module) :
    def __init__(self, lr, epochs, num_classes = 2) :

        super(SimpleFCNN, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), # 640 * 640 * 64
            nn.ReLU(),
            nn.MaxPool2d(2), # 320 * 320 * 64

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 320 * 320 * 128
            nn.ReLU(),
            nn.MaxPool2d(2), # 160 * 160 * 128
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size = 2, stride = 2), # 160 * 160 * 64
            nn.ReLU(),
            nn.ConvTranspose2d(64, num_classes, kernel_size=2, stride=2), # 160 * 160 * num_classes
        )

        self.model = nn.Sequential(
            self.encoder,
            self.decoder
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        # self.model_dir = model_dir
        self.epochs = epochs

        self.lr = lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def dice_loss(self, pred, target, smooth=1e-6):

        pred = torch.softmax(pred, dim=1)  # softmax over classes
        target_onehot = torch.nn.functional.one_hot(target, num_classes=pred.shape[1])
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()  # (N, C, H, W)

        intersection = (pred * target_onehot).sum(dim=(2,3))
        union = pred.sum(dim=(2,3)) + target_onehot.sum(dim=(2,3))

        dice = (2 * intersection + smooth) / (union + smooth)
        loss = 1 - dice.mean()  # average over batch and classes

        return loss

    def train_step(self, trainLoaders):
        self.model.train()
        total_loss = 0

        for batch in trainLoaders:
            images = batch["image"].to(self.device)
            masks = batch["mask"].long().to(self.device)

            self.optimizer.zero_grad()
            outputs = self(images)
            loss = self.dice_loss(outputs, masks)
            loss.backward()
            self.optimizer.step()

            total_loss += loss

        avg_loss = total_loss / len(trainLoaders)
        return avg_loss


    def evaluate_step(self, validLoaders):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in validLoaders:
                images = batch["image"].to(self.device)
                masks = batch["mask"].long().to(self.device)

                outputs = self(images)
                loss = self.dice_loss(outputs, masks)

                total_loss += loss

        avg_loss = total_loss / len(validLoaders)
        return avg_loss

    def predict(self, testLoaders):
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in testLoaders:
                images = batch["image"].to(self.device)
                outputs = self(images)
                _, predicted_masks = torch.max(outputs, dim=1)
                predictions.append(predicted_masks.cpu().numpy())

        return np.concatenate(predictions, axis=0)

    def train_model(self, trainLoaders, validLoaders):
        for epoch in range(self.epochs):
            train_loss = self.train_step(trainLoaders)
            valid_loss = self.evaluate_step(validLoaders)
            self.save_model()
            return epoch, train_loss, valid_loss


    def save_model(self):
        torch.save(self.model.state_dict(), "simpleCNN.pth")
