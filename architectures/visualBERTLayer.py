"""
Python version: 3.10
Description: Contains the architectural implementation of visualBERT based multimodal classifier trained with concatenation based
            fusion techniques.
Reference: https://arxiv.org/pdf/1908.03557 
"""

# %% Importing libraries
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report

import torch
from torch import nn
import torch.nn.functional as F

import lightning.pytorch as pl

from transformers import ViTModel, get_linear_schedule_with_warmup

class VisualBERTClassifier(pl.LightningModule):
    def __init__(self, visualbert_model, vit_model, learning_rate, num_classes, weight_decay, eps, warmup_steps, num_training_steps, max_seq_length=512, 
                max_visual_tokens=197):
        super(VisualBERTClassifier, self).__init__()
        self.visualbert_model = visualbert_model
        self.vit_model = vit_model
        self.classifier = nn.Linear(self.visualbert_model.config.hidden_size, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        # Max sequence length for transformers model
        self.max_seq_length = max_seq_length
        # Since we using the ViT-patch 16, max_visual_tokens = (Image Height/Patch Size) x (Image Height/Patch Size)
        # = (224/16) x (224/16) = 196
        self.max_visual_tokens = max_visual_tokens  # This is standard for ViT with 224x224 images
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.eps = eps
        self.warmup_steps = warmup_steps
        self.num_training_steps = num_training_steps
        self.test_outputs = []

    def forward(self, input_ids, attention_mask, pixel_values):
        # Extract visual embeddings using ViT
        vit_outputs = self.vit_model(pixel_values)
        visual_embeds = vit_outputs.last_hidden_state

        # Ensure the visual_embeds shape matches expected shape by VisualBERT
        batch_size, num_visual_tokens, hidden_dim = visual_embeds.shape

        # Trim or pad visual embeddings
        if num_visual_tokens > self.max_visual_tokens:  # Trim if the size exceeds expected
            visual_embeds = visual_embeds[:, :self.max_visual_tokens, :]
        elif num_visual_tokens < self.max_visual_tokens:  # Pad if the size is less than expected
            padding = torch.zeros((batch_size, self.max_visual_tokens - num_visual_tokens, hidden_dim), device=visual_embeds.device)
            visual_embeds = torch.cat((visual_embeds, padding), dim=1)

        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long).to(input_ids.device)
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float).to(input_ids.device)

        # Adjust input_ids and attention_mask to ensure the total length is within the limit
        total_length = input_ids.size(1) + self.max_visual_tokens
        if total_length > self.max_seq_length:
            excess_length = total_length - self.max_seq_length
            input_ids = input_ids[:, :-excess_length]
            attention_mask = attention_mask[:, :-excess_length]

        # Concatenate text and visual embeddings
        text_embeds = self.visualbert_model.embeddings.word_embeddings(input_ids)
        token_type_embeddings = self.visualbert_model.embeddings.token_type_embeddings(
            torch.cat((torch.zeros_like(input_ids), visual_token_type_ids), dim=1))
        position_ids = torch.arange(text_embeds.size(1) + visual_embeds.size(1), dtype=torch.long, device=input_ids.device)
        position_embeddings = self.visualbert_model.embeddings.position_embeddings(position_ids)

        embeddings = torch.cat((text_embeds, visual_embeds), dim=1)
        embeddings += token_type_embeddings + position_embeddings
        embeddings = self.visualbert_model.embeddings.LayerNorm(embeddings)
        embeddings = self.visualbert_model.embeddings.dropout(embeddings)

        # Concatenate attention masks
        combined_attention_mask = torch.cat((attention_mask, visual_attention_mask), dim=1)
        combined_attention_mask = combined_attention_mask.unsqueeze(1).unsqueeze(2)

        encoder_outputs = self.visualbert_model.encoder(
            embeddings,
            attention_mask=combined_attention_mask,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True,
        )

        pooled_output = encoder_outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled_output)
        return logits

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        pixel_values = batch['pixel_values']
        labels = batch['label']
        logits = self(input_ids, attention_mask, pixel_values)
        loss = self.criterion(logits, labels)
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

        self.log('val_loss', loss)
        self.log('val_acc', acc)
        self.log('val_f1_weighted', f1_weighted)
        self.log('val_f1_micro', f1_micro)
        self.log('val_f1_macro', f1_macro)

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