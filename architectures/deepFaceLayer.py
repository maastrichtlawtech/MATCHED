# %% Loading 
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score

import lightning.pytorch as pl

from deepface import DeepFace

class DeepFaceModel(pl.LightningModule):
    def __init__(self, model_name, num_classes, learning_rate=1e-3):
        super(DeepFaceModel, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        # Set the embedding size based on the model name
        if model_name == "VGG-Face":
            self.embedding_size = 2622
        elif model_name == "Facenet":
            self.embedding_size = 128
        elif model_name == "Facenet512":
            self.embedding_size = 512
        elif model_name == "OpenFace":
            self.embedding_size = 128
        elif model_name == "DeepFace":
            self.embedding_size = 4096
        elif model_name == "DeepID":
            self.embedding_size = 160
        elif model_name == "ArcFace":
            self.embedding_size = 512
        elif model_name == "Dlib":
            self.embedding_size = 128
        elif model_name == "SFace":
            self.embedding_size = 128
        elif model_name == "GhostFaceNet":
            self.embedding_size = 512 
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        self.classifier = nn.Linear(self.embedding_size, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        embeddings = []
        for img in x:
            img = img.permute(1, 2, 0).cpu().numpy()
            embedding = DeepFace.represent(img, model_name=self.model_name, enforce_detection=False)
            embeddings.append(embedding[0]['embedding'])
        embeddings = torch.tensor(embeddings).to(self.device)
        return self.classifier(embeddings), embeddings

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self(x)
        loss = self.criterion(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = accuracy_score(y.cpu(), preds.cpu())
        balanced_acc = balanced_accuracy_score(y.cpu(), preds.cpu())
        micro_f1 = f1_score(y.cpu(), preds.cpu(), average='micro')
        macro_f1 = f1_score(y.cpu(), preds.cpu(), average='macro')
        weighted_f1 = f1_score(y.cpu(), preds.cpu(), average='weighted')
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_balanced_acc', balanced_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_micro_f1', micro_f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_macro_f1', macro_f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_weighted_f1', weighted_f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self(x)
        loss = self.criterion(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = accuracy_score(y.cpu(), preds.cpu())
        balanced_acc = balanced_accuracy_score(y.cpu(), preds.cpu())
        micro_f1 = f1_score(y.cpu(), preds.cpu(), average='micro')
        macro_f1 = f1_score(y.cpu(), preds.cpu(), average='macro')
        weighted_f1 = f1_score(y.cpu(), preds.cpu(), average='weighted')
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_balanced_acc', balanced_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_micro_f1', micro_f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_macro_f1', macro_f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_weighted_f1', weighted_f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self(x)
        loss = self.criterion(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = accuracy_score(y.cpu(), preds.cpu())
        balanced_acc = balanced_accuracy_score(y.cpu(), preds.cpu())
        micro_f1 = f1_score(y.cpu(), preds.cpu(), average='micro')
        macro_f1 = f1_score(y.cpu(), preds.cpu(), average='macro')
        weighted_f1 = f1_score(y.cpu(), preds.cpu(), average='weighted')
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_balanced_acc', balanced_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_micro_f1', micro_f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_macro_f1', macro_f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_weighted_f1', weighted_f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer