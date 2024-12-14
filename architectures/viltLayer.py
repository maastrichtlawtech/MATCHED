"""
Python version: 3.10
Description: Contains the architectural implementation of ViLT based multimodal classifier trained with concatenation based
            fusion techniques.
Reference: https://arxiv.org/abs/2102.03334
"""

# %% Importing libraries
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report

import torch
from torch import nn
import torch.nn.functional as F

import lightning.pytorch as pl

from transformers import ViTModel, get_linear_schedule_with_warmup

# %% Model Definition for ViLT
class ViLTClassifier(pl.LightningModule):
    def __init__(self, vilt_model, learning_rate, num_classes, weight_decay, eps, warmup_steps, num_training_steps):
        super(ViLTClassifier, self).__init__()
        self.vilt_model = vilt_model
        self.classifier = nn.Linear(self.vilt_model.config.hidden_size, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.eps = eps
        self.warmup_steps = warmup_steps
        self.num_training_steps = num_training_steps
        self.test_outputs = [] 

    def forward(self, input_ids, attention_mask, pixel_values):
        outputs = self.vilt_model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        pooled_output = outputs.pooler_output  # ViLT provides a pooled output directly
        logits = self.classifier(pooled_output)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self(batch['input_ids'], batch['attention_mask'], batch['pixel_values'])
        loss = self.criterion(logits, batch['label'])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        pixel_values = batch['pixel_values']
        labels = batch['label']
        logits = self(input_ids, attention_mask, pixel_values)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)

        acc = balanced_accuracy_score(labels.cpu(), preds.cpu())
        f1_weighted = f1_score(labels.cpu(), preds.cpu(), average='weighted')
        f1_micro = f1_score(labels.cpu(), preds.cpu(), average='micro')
        f1_macro = f1_score(labels.cpu(), preds.cpu(), average='macro')

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_f1_weighted', f1_weighted, prog_bar=True)
        self.log('val_f1_micro', f1_micro, prog_bar=True)
        self.log('val_f1_macro', f1_macro, prog_bar=True)
        
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        pixel_values = batch['pixel_values']
        labels = batch['label']
        logits = self(input_ids, attention_mask, pixel_values)
        preds = torch.argmax(logits, dim=1)

        acc = balanced_accuracy_score(labels.cpu(), preds.cpu())
        f1_weighted = f1_score(labels.cpu(), preds.cpu(), average='weighted')
        f1_micro = f1_score(labels.cpu(), preds.cpu(), average='micro')
        f1_macro = f1_score(labels.cpu(), preds.cpu(), average='macro')

        self.test_outputs.append({"acc": acc, "f1_weighted": f1_weighted, "f1_micro": f1_micro, "f1_macro": f1_macro, "labels": labels.cpu(), "preds": preds.cpu()})

        return {"acc": acc, "f1_weighted": f1_weighted, "f1_micro": f1_micro, "f1_macro": f1_macro}

    def on_test_epoch_end(self):
        avg_acc = torch.tensor([x['acc'] for x in self.test_outputs]).mean()
        avg_f1_weighted = torch.tensor([x['f1_weighted'] for x in self.test_outputs]).mean()
        avg_f1_micro = torch.tensor([x['f1_micro'] for x in self.test_outputs]).mean()
        avg_f1_macro = torch.tensor([x['f1_macro'] for x in self.test_outputs]).mean()

        labels = torch.cat([x['labels'] for x in self.test_outputs])
        preds = torch.cat([x['preds'] for x in self.test_outputs])

        self.log('test_acc', avg_acc)
        self.log('test_f1_weighted', avg_f1_weighted)
        self.log('test_f1_micro', avg_f1_micro)
        self.log('test_f1_macro', avg_f1_macro)

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [{'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay':self.weight_decay}, 
                                        {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay}]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=self.eps)

        # We also use a scheduler that is supplied by transformers.
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.num_training_steps)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]
