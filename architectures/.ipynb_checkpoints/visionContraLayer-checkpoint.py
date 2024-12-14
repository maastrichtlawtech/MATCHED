import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score

import lightning.pytorch as pl

import timm

# Lightning Model
class PreTrainedVisionContraModel(pl.LightningModule):
    def __init__(self, model_name, num_classes, learning_rate=1e-4, temp=0.07, contrastive_weight=1):
        super().__init__()
        self.model = self.load_model(model_name, num_classes)
        self.learning_rate = learning_rate
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.contrastive_loss = SupervisedContrastiveLoss(temperature=temp)
        self.contrastive_weight = contrastive_weight

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
        else:
            num_ftrs = model.get_classifier().in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
        return model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        features = self.model.forward_features(x)  # Extract features for contrastive loss
        features = features[:, 0, :]  # Use [CLS] token if available
        y_hat = self(x)  # Predictions for cross-entropy loss
        
        ce_loss = self.cross_entropy_loss(y_hat, y)
        # ce_loss = 0
        contrastive_loss = self.contrastive_loss(features, y)
        loss = ce_loss + self.contrastive_weight * contrastive_loss

        preds = torch.argmax(y_hat, dim=1)
        metrics = self.compute_metrics(preds, y, 'train')
        metrics['train_loss'] = loss
        metrics['train_cross_entropy_loss'] = ce_loss
        metrics['train_contrastive_loss'] = contrastive_loss
        self.log_dict(metrics)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        features = self.model.forward_features(x)
        features = features[:, 0, :]  # Use [CLS] token if available
        y_hat = self(x)
        
        ce_loss = self.cross_entropy_loss(y_hat, y)
        # ce_loss = 0
        contrastive_loss = self.contrastive_loss(features, y)
        loss = ce_loss + self.contrastive_weight * contrastive_loss

        preds = torch.argmax(y_hat, dim=1)
        metrics = self.compute_metrics(preds, y, 'val')
        metrics['val_loss'] = loss
        metrics['val_cross_entropy_loss'] = ce_loss
        metrics['val_contrastive_loss'] = contrastive_loss
        self.log_dict(metrics)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        features = self.model.forward_features(x)
        features = features[:, 0, :]  # Use [CLS] token if available
        y_hat = self(x)
        
        ce_loss = self.cross_entropy_loss(y_hat, y)
        # ce_loss = 0
        contrastive_loss = self.contrastive_loss(features, y)
        loss = ce_loss + self.contrastive_weight * contrastive_loss

        preds = torch.argmax(y_hat, dim=1)
        metrics = self.compute_metrics(preds, y, 'test')
        metrics['test_loss'] = loss
        metrics['test_cross_entropy_loss'] = ce_loss
        metrics['test_contrastive_loss'] = contrastive_loss
        self.log_dict(metrics)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)  # Example scheduler
        return [optimizer], [scheduler]

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

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        features = F.normalize(features, dim=1)

        dot_product = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
        logits = dot_product - logits_max.detach()

        mask = mask.to(device)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(features.shape[0]).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)  # Add epsilon for numerical stability

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)  # Add epsilon for numerical stability

        loss = -mean_log_prob_pos
        loss = loss.mean()

        return loss