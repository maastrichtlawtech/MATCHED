import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score

import lightning.pytorch as pl

import timm
from transformers import AdamW, get_linear_schedule_with_warmup

# Lightning Model
class PreTrainedVisionContraModel(pl.LightningModule):
    def __init__(self, model_name, num_classes, num_training_steps, warmup_steps=0, weight_decay=0.01, learning_rate=1e-4, temp=0.07, loss2_type="SupCon", nb_negatives=5, contrastive_weight=1, epsilon=1e-12):
        super().__init__()
        self.model = self.load_model(model_name, num_classes)
        self.learning_rate = learning_rate
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.loss2_type = loss2_type
        self.contrastive_weight = contrastive_weight
        self.nb_negatives = nb_negatives
        self.eps = epsilon
        self.num_training_steps = num_training_steps
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay

        if self.loss2_type == "SupCon":
            self.contrastive_loss = SupervisedContrastiveLoss(temperature=temp, num_negatives=nb_negatives)
        else:
            self.contrastive_loss = TripletLoss(margin=1.0)

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
        # ce_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        # Computing the contrastive loss
        if self.loss2_type == 'SupCon':
            contrastive_loss = self.contrastive_loss(features, y)
        elif self.loss2_type == 'triplet':
            contrastive_loss = compute_triplet_loss(features, y, self.contrastive_loss, self.nb_negatives)

        loss = self.epsilon + ce_loss + (self.contrastive_weight * contrastive_loss) # Epsilon added to make sure that the loss is never zero

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
        # ce_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        # Computing the contrastive loss
        if self.loss2_type == 'SupCon':
            contrastive_loss = self.contrastive_loss(features, y)
        elif self.loss2_type == 'triplet':
            contrastive_loss = compute_triplet_loss(features, y, self.contrastive_loss, self.nb_negatives)

        loss = self.epsilon + ce_loss + (self.contrastive_weight * contrastive_loss) # Epsilon added to make sure that the loss is never zero

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
        # ce_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        # Computing the contrastive loss
        if self.loss2_type == 'SupCon':
            contrastive_loss = self.contrastive_loss(features, y)
        elif self.loss2_type == 'triplet':
            contrastive_loss = compute_triplet_loss(features, y, self.contrastive_loss, self.nb_negatives)

        loss = self.epsilon + ce_loss + (self.contrastive_weight * contrastive_loss) # Epsilon added to make sure that the loss is never zero

        preds = torch.argmax(y_hat, dim=1)
        metrics = self.compute_metrics(preds, y, 'test')
        metrics['test_loss'] = loss
        metrics['test_cross_entropy_loss'] = ce_loss
        metrics['test_contrastive_loss'] = contrastive_loss
        self.log_dict(metrics)
        return loss

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [{'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay':self.weight_decay}, 
                                        {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay}]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=self.eps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.num_training_steps)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

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
    def __init__(self, temperature=0.07, num_negatives=5):
        """
        Args:
            temperature (float): The temperature parameter for scaling.
            num_negatives (int or None): The number of in-batch negatives to use.
                                         If None, use all available in-batch negatives.
        """
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.num_negatives = num_negatives

    def forward(self, features, labels):
        device = features.device
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        features = F.normalize(features, dim=1)

        dot_product = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
        logits = dot_product - logits_max.detach()

        logits_mask = torch.ones_like(mask)
        logits_mask = torch.scatter(
            logits_mask,
            1,
            torch.arange(features.shape[0]).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        if self.num_negatives is not None:
            # Apply the number of negatives only if there are enough negatives available
            for i in range(mask.size(0)):
                positive_count = int(mask[i].sum().item()) - 1  # Exclude self
                available_negatives = int(logits_mask[i].sum().item()) - positive_count

                if available_negatives > 0:
                    max_negatives = min(self.num_negatives, available_negatives)
                    negative_indices = torch.nonzero(mask[i] == 0, as_tuple=True)[0]

                    if negative_indices.size(0) > 0:
                        selected_negatives = torch.randperm(negative_indices.size(0))[:max_negatives]
                        negative_mask = torch.zeros_like(mask[i])
                        negative_mask[negative_indices[selected_negatives]] = 1
                        mask[i] = mask[i] + negative_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)  # Add epsilon for numerical stability

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)  # Add epsilon for numerical stability

        loss = -mean_log_prob_pos.mean()

        return loss

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Normalize the embeddings
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)

        # Compute the cosine similarity
        pos_sim = F.cosine_similarity(anchor, positive)
        neg_sim = F.cosine_similarity(anchor, negative)

        # Triplet loss
        loss = F.relu(self.margin - pos_sim + neg_sim)

        return loss.mean()

def compute_triplet_loss(features, labels, triplet_loss_fn, num_negatives):
    # Ensure features and labels are on the same device
    device = features.device
    labels = labels.to(device)
    
    triplets = []
    for i in range(len(labels)):
        anchor = features[i]
        pos_indices = torch.where(labels == labels[i])[0]
        neg_indices = torch.where(labels != labels[i])[0]
        
        # Ensure there are enough positive and negative samples
        if len(pos_indices) <= 1 or len(neg_indices) == 0:
            continue
        
        # Select a single positive sample
        pos_index = pos_indices[pos_indices != i][torch.randint(len(pos_indices) - 1, (1,)).item()]
        positive = features[pos_index]
        
        # Select multiple negative samples
        for _ in range(num_negatives):
            neg_index = neg_indices[torch.randint(len(neg_indices), (1,)).item()]
            negative = features[neg_index]
            triplets.append((anchor, positive, negative))
    
    if len(triplets) == 0:
        return torch.tensor(0.0, requires_grad=True, device=device)
    
    anchor, positive, negative = zip(*triplets)
    anchor = torch.stack(anchor)
    positive = torch.stack(positive)
    negative = torch.stack(negative)

    return triplet_loss_fn(anchor, positive, negative)

class SelfConstrativeVisionModel(pl.LightningModule):
    def __init__(self, model_name, num_training_steps, warmup_steps=0, weight_decay=0.01, learning_rate=1e-4, temp=0.07, loss_type="SupCon", nb_negatives=5, contrastive_weight=1, epsilon=1e-12):
        super().__init__()
        self.model = self.load_model(model_name)
        self.learning_rate = learning_rate
        self.loss_type = loss_type
        self.contrastive_weight = contrastive_weight
        self.nb_negatives = nb_negatives
        self.eps = epsilon
        self.num_training_steps = num_training_steps
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay

        if self.loss_type == "SupCon":
            self.contrastive_loss = SupervisedContrastiveLoss(temperature=temp, num_negatives=nb_negatives)
        else:
            self.contrastive_loss = TripletLoss(margin=1.0)

    def load_model(self, model_name):
        model = timm.create_model(model_name, pretrained=True, num_classes=0)  # No classification head
        return model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        # Forward pass to extract features
        features = self.model.forward_features(x)
        features = features[:, 0, :]  # Use [CLS] token if available

        # Compute contrastive loss
        if self.loss_type == 'SupCon':
            contrastive_loss = self.contrastive_loss(features, y)
        elif self.loss_type == 'triplet':
            contrastive_loss = compute_triplet_loss(features, y, self.contrastive_loss, self.nb_negatives)

        loss = self.contrastive_weight * contrastive_loss

        self.log_dict({'train_loss': loss})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # Forward pass to extract features
        features = self.model.forward_features(x)
        features = features[:, 0, :]  # Use [CLS] token if available

        # Compute contrastive loss
        if self.loss_type == 'SupCon':
            contrastive_loss = self.contrastive_loss(features, y)
        elif self.loss_type == 'triplet':
            contrastive_loss = compute_triplet_loss(features, y, self.contrastive_loss, self.nb_negatives)

        loss = self.contrastive_weight * contrastive_loss

        self.log_dict({'val_loss': loss})
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        # Forward pass to extract features
        features = self.model.forward_features(x)
        features = features[:, 0, :]  # Use [CLS] token if available

        # Compute contrastive loss
        if self.loss_type == 'SupCon':
            contrastive_loss = self.contrastive_loss(features, y)
        elif self.loss_type == 'triplet':
            contrastive_loss = compute_triplet_loss(features, y, self.contrastive_loss, self.nb_negatives)

        loss = self.contrastive_weight * contrastive_loss

        self.log_dict({'test_loss': loss})
        return loss

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [{'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay':self.weight_decay}, 
                                        {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay}]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=self.eps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.num_training_steps)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]