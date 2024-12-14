# %% Loading 
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score

import lightning.pytorch as pl

import timm

# Lightning Model
class PreTrainedVisionModel(pl.LightningModule):
    def __init__(self, model_name, num_classes, learning_rate=1e-4):
        super().__init__()
        self.model = self.load_model(model_name, num_classes)
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()

    def load_model(self, model_name, num_classes):
        model = timm.create_model(model_name, pretrained=True)
        
        if 'efficientnet' in model_name or 'efficientnetv2' in model_name:
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, num_classes)
        elif 'convnext' in model_name:
            num_ftrs = model.head.fc.in_features
            model.head.fc = nn.Linear(num_ftrs, num_classes)
        elif 'vit' in model_name:
            num_ftrs = model.head.in_features
            model.head = nn.Linear(num_ftrs, num_classes)
        elif 'vgg' in model_name or 'densenet' in model_name:
            model.reset_classifier(num_classes=num_classes)
        else:
            num_ftrs = model.get_classifier().in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
        
        return model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        metrics = self.compute_metrics(preds, y, 'train')
        metrics['train_loss'] = loss
        self.log_dict(metrics)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        metrics = self.compute_metrics(preds, y, 'val')
        metrics['val_loss'] = loss
        self.log_dict(metrics)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        metrics = self.compute_metrics(preds, y, 'test')
        metrics['test_loss'] = loss
        self.log_dict(metrics)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def compute_metrics(self, preds, labels, stage):
        acc = accuracy_score(labels.cpu(), preds.cpu())
        precision = precision_score(labels.cpu(), preds.cpu(), average='micro')
        recall = recall_score(labels.cpu(), preds.cpu(), average='micro')
        f1 = f1_score(labels.cpu(), preds.cpu(), average='micro')
        balanced_acc = balanced_accuracy_score(labels.cpu(), preds.cpu())
        macro_f1 = f1_score(labels.cpu(), preds.cpu(), average='macro')
        weighted_f1 = f1_score(labels.cpu(), preds.cpu(), average='weighted')
        
        return {
            f'{stage}_accuracy': acc,
            f'{stage}_precision': precision,
            f'{stage}_recall': recall,
            f'{stage}_f1': f1,
            f'{stage}_balanced_accuracy': balanced_acc,
            f'{stage}_macro_f1': macro_f1,
            f'{stage}_weighted_f1': weighted_f1,
        }