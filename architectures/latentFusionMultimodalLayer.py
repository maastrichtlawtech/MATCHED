"""
Python version: 3.10
Description: Contains the architectural implementation of Declur-small and ViT-patch16 based multimodal classifier trained with latent
            fusion techniques like concatenation, mean, addition, learned fusion using secondary neural network, and attention mechanism 
"""

# %% Importing libraries
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report

import torch
from torch import nn
import torch.nn.functional as F

import lightning.pytorch as pl

from transformers import AutoModel, ViTModel, get_linear_schedule_with_warmup

# %% Model class
class LatentFusionMultimodalModel(pl.LightningModule):
    def __init__(self, text_model, image_model, fusion_technique, num_classes, learning_rate, weight_decay, eps, warmup_steps, 
                num_training_steps):
        super(LatentFusionMultimodalModel, self).__init__()
        self.text_model = text_model
        self.image_model = image_model
        self.fusion_technique = fusion_technique
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.eps = eps
        self.warmup_steps = warmup_steps
        self.num_training_steps = num_training_steps

        # Define MultiheadAttention layer for attention fusion
        if fusion_technique == 'attention':
            self.attention_layer = nn.MultiheadAttention(embed_dim=768, num_heads=8)

        # Define Linear layer for learned fusion
        if fusion_technique == 'learned_fusion':
            self.learned_fusion = nn.Linear(1536, 768)  # Map concatenated embeddings to the same size

        # Classifier dimensions based on fusion technique
        if fusion_technique == 'concat':
            self.multimodal_classifier = nn.Linear(1536, self.num_classes)
        else:
            self.multimodal_classifier = nn.Linear(768, self.num_classes)

        self.validation_outputs = []
        self.test_outputs = []

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None):
        text_embeddings = None
        image_embeddings = None

        if input_ids is not None and attention_mask is not None:
            text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
            text_embeddings = text_outputs.last_hidden_state[:, 0, :]  # CLS token

        if pixel_values is not None:
            image_outputs = self.image_model(pixel_values=pixel_values)
            image_embeddings = image_outputs.last_hidden_state[:, 0, :]  # CLS token

        if text_embeddings is not None and image_embeddings is not None:
            if self.fusion_technique == 'mean':
                combined_embeddings = (text_embeddings + image_embeddings) / 2
            elif self.fusion_technique == 'concat':
                combined_embeddings = torch.cat((text_embeddings, image_embeddings), dim=1)
            elif self.fusion_technique == 'add':
                combined_embeddings = text_embeddings + image_embeddings
            elif self.fusion_technique == 'multiply':
                combined_embeddings = text_embeddings * image_embeddings
            elif self.fusion_technique == 'attention':
                combined_embeddings, _ = self.attention_layer(text_embeddings.unsqueeze(1), image_embeddings.unsqueeze(1), image_embeddings.unsqueeze(1))
                combined_embeddings = combined_embeddings.squeeze(1)
            elif self.fusion_technique == 'learned_fusion':
                combined_embeddings = torch.cat((text_embeddings, image_embeddings), dim=1)
                combined_embeddings = self.learned_fusion(combined_embeddings)  # Map back to the same size
            else:
                raise ValueError("The chosen fusion technique is not implemented")

            logits = self.multimodal_classifier(combined_embeddings)
            return logits, combined_embeddings
        else:
            embeddings = text_embeddings if text_embeddings is not None else image_embeddings
            logits = self.multimodal_classifier(embeddings)
            return logits, embeddings

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, pixel_values, labels = batch.get('input_ids'), batch.get('attention_mask'), batch.get('pixel_values'), batch['label']
        logits, _ = self(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        loss = F.cross_entropy(logits, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, pixel_values, labels = batch.get('input_ids'), batch.get('attention_mask'), batch.get('pixel_values'), batch['label']
        logits, _ = self(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        self.validation_outputs.append({'preds': preds, 'labels': labels})
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        val_preds = torch.cat([x['preds'] for x in self.validation_outputs])
        val_labels = torch.cat([x['labels'] for x in self.validation_outputs])
        val_acc = balanced_accuracy_score(val_labels.cpu().numpy(), val_preds.cpu().numpy())
        val_f1_weighted = f1_score(val_labels.cpu().numpy(), val_preds.cpu().numpy(), average='weighted')
        val_f1_micro = f1_score(val_labels.cpu().numpy(), val_preds.cpu().numpy(), average='micro')
        val_f1_macro = f1_score(val_labels.cpu().numpy(), val_preds.cpu().numpy(), average='macro')
        self.log('val_acc', val_acc, on_step=False, on_epoch=True)
        self.log('val_f1_weighted', val_f1_weighted, on_step=False, on_epoch=True)
        self.log('val_f1_micro', val_f1_micro, on_step=False, on_epoch=True)
        self.log('val_f1_macro', val_f1_macro, on_step=False, on_epoch=True)
        self.validation_outputs = []

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, pixel_values, labels = batch.get('input_ids'), batch.get('attention_mask'), batch.get('pixel_values'), batch['label']
        logits, _ = self(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        self.test_outputs.append({'preds': preds, 'labels': labels})
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        return loss

    def on_test_epoch_end(self):
        test_preds = torch.cat([x['preds'] for x in self.test_outputs])
        test_labels = torch.cat([x['labels'] for x in self.test_outputs])
        test_acc = balanced_accuracy_score(test_labels.cpu().numpy(), test_preds.cpu().numpy())
        test_f1_weighted = f1_score(test_labels.cpu().numpy(), test_preds.cpu().numpy(), average='weighted')
        test_f1_micro = f1_score(test_labels.cpu().numpy(), test_preds.cpu().numpy(), average='micro')
        test_f1_macro = f1_score(test_labels.cpu().numpy(), test_preds.cpu().numpy(), average='macro')
        self.log('test_acc', test_acc, on_step=False, on_epoch=True)
        self.log('test_f1_weighted', test_f1_weighted, on_step=False, on_epoch=True)
        self.log('test_f1_micro', test_f1_micro, on_step=False, on_epoch=True)
        self.log('test_f1_macro', test_f1_macro, on_step=False, on_epoch=True)
        self.test_outputs = []

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [{'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay':self.weight_decay}, 
                                        {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay}]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=self.eps)

        # We also use a scheduler that is supplied by transformers.
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.num_training_steps)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]

    def extract_embeddings(self, input_ids=None, attention_mask=None, pixel_values=None):
        self.eval()
        with torch.no_grad():
            if input_ids is not None and attention_mask is not None and pixel_values is not None:
                _, combined_embeddings = self(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
                return combined_embeddings
            elif input_ids is not None and attention_mask is not None:
                text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
                text_embeddings = text_outputs.last_hidden_state[:, 0, :]  # CLS token
                return text_embeddings
            elif pixel_values is not None:
                image_outputs = self.image_model(pixel_values=pixel_values)
                image_embeddings = image_outputs.last_hidden_state[:, 0, :]  # CLS token
                return image_embeddings