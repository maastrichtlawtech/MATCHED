# %% Importing libraries
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report

import torch
from torch import nn
import torch.nn.functional as F

import lightning.pytorch as pl

from transformers import AutoModel, ViTModel, get_linear_schedule_with_warmup

# %% Supervised Contrastive Loss (SupCon) with hard negatives definition
# Define the Supervised Contrastive Loss class with support for hard negatives
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, num_hard_negatives=5, eps=1e-8):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.num_hard_negatives = num_hard_negatives
        self.eps = eps

    def forward(self, features, labels):
        device = features.device
        labels = labels.to(device)
        batch_size = features.shape[0]

        # Normalize the features to the unit sphere
        features = F.normalize(features, p=2, dim=1)

        # Compute the cosine similarity matrix and apply temperature scaling
        sim_matrix = torch.mm(features, features.t())
        sim_matrix = torch.clamp(sim_matrix, min=-10, max=10) / self.temperature  # Clamping extreme values

        # Mask for identifying positive pairs (excluding the diagonal)
        pos_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
        pos_mask.fill_diagonal_(0)

        # Mask for identifying negatives
        neg_mask = labels.unsqueeze(1) != labels.unsqueeze(0)

        # Select hard negatives: top 'num_hard_negatives' from each row, considering only negatives
        negative_scores = sim_matrix * neg_mask.float()
        
        # Get the actual number of negative samples in the batch
        num_negatives = neg_mask.sum(dim=1)
        actual_num_hard_negatives = torch.minimum(num_negatives, torch.tensor(self.num_hard_negatives).to(device))
        
        # Select top 'actual_num_hard_negatives' for each row
        top_negatives = torch.zeros(batch_size, self.num_hard_negatives, device=device)
        for i in range(batch_size):
            if actual_num_hard_negatives[i] > 0:
                top_negatives[i, :actual_num_hard_negatives[i]] = torch.topk(negative_scores[i], k=actual_num_hard_negatives[i].item()).values

        # Log-sum-exp trick for numerical stability: max_sim for each row
        max_sim, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        exp_sim_matrix = torch.exp(sim_matrix - max_sim)

        # Sum of exps for positive and hard negative pairs
        exp_pos = (exp_sim_matrix * pos_mask.float()).sum(dim=1, keepdim=True)
        exp_hard_neg = torch.exp(top_negatives - max_sim)

        # Combine positives and hard negatives in the denominator
        denom = exp_pos + exp_hard_neg.sum(dim=1, keepdim=True) + torch.exp(negative_scores - max_sim).sum(dim=1, keepdim=True) - exp_hard_neg

        # Log probability of positive pairs
        log_prob_pos = torch.log(exp_pos / (denom + self.eps) + self.eps)

        # Compute the mean of negative log probabilities across the batch
        loss = -log_prob_pos.mean()
        return loss

# Define NT-XENT Loss class
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features):
        device = features.device
        batch_size = features.shape[0]

        # Normalize the features to the unit sphere
        features = F.normalize(features, p=2, dim=1)

        # Compute the cosine similarity matrix and apply temperature scaling
        sim_matrix = torch.mm(features, features.t()) / self.temperature

        # Create labels for contrastive learning
        labels = torch.arange(batch_size).to(device)
        labels = torch.cat([labels, labels])

        # Log-sum-exp trick for numerical stability: max_sim for each row
        max_sim, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        exp_sim_matrix = torch.exp(sim_matrix - max_sim)

        # Sum of exp for all pairs
        denom = exp_sim_matrix.sum(dim=1, keepdim=True)

        # Log probability of positive pairs
        log_prob_pos = torch.log(exp_sim_matrix[range(batch_size), range(batch_size)] / (denom + 1e-8))

        # Compute the mean of negative log probabilities across the batch
        loss = -log_prob_pos.mean()
        return loss

# Define Image-Text Matching Loss (ITM) as a class
class ITMLoss(nn.Module):
    def __init__(self):
        super(ITMLoss, self).__init__()

    def forward(self, text_embeddings, image_embeddings, labels):
        """
        Image-Text Matching Loss (ITM)
        """
        batch_size = text_embeddings.size(0)

        if text_embeddings.shape[0] == 0 or image_embeddings.shape[0] == 0:
            raise RuntimeError(f"Embeddings are empty: text_embeddings shape {text_embeddings.shape}, image_embeddings shape {image_embeddings.shape}")

        # Normalize the embeddings
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        image_embeddings = F.normalize(image_embeddings, p=2, dim=1)

        # Cosine similarity between text and image embeddings
        similarities = torch.matmul(text_embeddings, image_embeddings.t())

        # Generate labels for ITM
        itm_labels = torch.arange(batch_size).to(text_embeddings.device)

        # Cross entropy loss
        itm_loss_value = F.cross_entropy(similarities, itm_labels)

        return itm_loss_value

class QFormer(nn.Module):
    def __init__(self, hidden_dim, num_queries, num_layers, num_heads):
        super(QFormer, self).__init__()
        self.query_embeddings = nn.Parameter(torch.randn(num_queries, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, text_features, image_features):
        batch_size = text_features.size(0)
        query_embeddings = self.query_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # Expand queries for the batch

        # Ensure text_features and image_features have the same dimensionality as query_embeddings
        text_features = text_features.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        image_features = image_features.unsqueeze(1)  # [batch_size, 1, hidden_dim]

        # Concatenate the queries with the text and image features
        features = torch.cat([query_embeddings, text_features, image_features], dim=1)
        transformed_features = self.transformer_encoder(features)

        # Return the transformed features (query embeddings + transformed text & image features)
        return transformed_features[:, :self.query_embeddings.size(0)], transformed_features[:, self.query_embeddings.size(0):]

# Define the LatentFusionMultimodalModel class
class multimodalFusionModel(pl.LightningModule):
    def __init__(self, text_model, image_model, fusion_technique, num_classes, learning_rate, weight_decay, eps, warmup_steps, 
                num_training_steps, temperature, ce_weight, supcon_weight, itm_weight, ntxent_weight, num_hard_negatives, loss_function='CE'):
        super(multimodalFusionModel, self).__init__()
        self.text_model = text_model
        self.image_model = image_model
        self.fusion_technique = fusion_technique
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.eps = eps
        self.warmup_steps = warmup_steps
        self.num_training_steps = num_training_steps
        self.loss_function = loss_function
        self.ce_weight = ce_weight
        self.supcon_weight = supcon_weight
        self.itm_weight = itm_weight
        self.ntxent_weight = ntxent_weight
        self.num_hard_negatives = num_hard_negatives

        # Q-Former for advanced fusion technique
        if fusion_technique == 'qformer':
            self.q_former = QFormer(hidden_dim=768, num_queries=32, num_layers=6, num_heads=8)
            # self.q_former = QFormer(hidden_dim=768, num_queries=32, num_layers=6, num_heads=2)

        # Define MultiheadAttention layer for cross-head attention fusion
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

        self.validation_outputs = []  # To store validation outputs
        self.test_outputs = []  # To store test outputs
        self.supcon_loss = SupConLoss(temperature, num_hard_negatives=self.num_hard_negatives)  # Initialize SupConLoss
        self.ntxent_loss = NTXentLoss(temperature)  # Initialize NTXentLoss
        self.itm_loss = ITMLoss()  # Initialize ITMLoss

    # Define the forward pass
    def forward(self, input_ids=None, attention_mask=None, pixel_values=None):
        text_embeddings = None
        image_embeddings = None

        # Get text embeddings if input_ids and attention_mask are provided
        if input_ids is not None and attention_mask is not None:
            text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
            text_embeddings = text_outputs.last_hidden_state[:, 0, :]  # CLS token

        # Get image embeddings if pixel_values are provided
        if pixel_values is not None:
            image_outputs = self.image_model(pixel_values=pixel_values)
            image_embeddings = image_outputs.last_hidden_state[:, 0, :]  # CLS token

        # Combine text and image embeddings based on the fusion technique
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
            elif self.fusion_technique == 'qformer':
                query_embeddings, _ = self.q_former(text_embeddings, image_embeddings)
                combined_embeddings = query_embeddings.mean(dim=1)  # Aggregate across query tokens
            else:
                raise ValueError("The chosen fusion technique is not implemented")

            logits = self.multimodal_classifier(combined_embeddings)
            return logits, combined_embeddings
        else:
            embeddings = text_embeddings if text_embeddings is not None else image_embeddings
            logits = self.multimodal_classifier(embeddings)
            return logits, embeddings

    # Define the training step
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, pixel_values, labels = batch.get('input_ids'), batch.get('attention_mask'), batch.get('pixel_values'), batch['label']
        logits, combined_embeddings = self(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)

        # Initialize loss components
        ce_loss = supcon_loss = ntxent_loss = itm_loss_value = 0

        # Compute CE loss if applicable
        if 'CE' in self.loss_function:
            ce_loss = F.cross_entropy(logits, labels)
            loss = self.ce_weight * ce_loss

        # Compute SupCon loss if applicable
        if 'SupCon' in self.loss_function:
            supcon_loss = self.supcon_loss(combined_embeddings, labels)
            loss += self.supcon_weight * supcon_loss

        # Compute NT-XENT loss if applicable
        if 'NTXent' in self.loss_function:
            ntxent_loss = self.ntxent_loss(combined_embeddings)
            loss += self.ntxent_weight * ntxent_loss

        # Compute ITM loss if applicable
        if 'ITM' in self.loss_function:
            if self.fusion_technique in ['concat', 'mean']:
                text_embeddings, image_embeddings = self._get_separate_embeddings(input_ids, attention_mask, pixel_values, combined_embeddings)
                itm_loss_value = self.itm_loss(text_embeddings, image_embeddings, labels)
                loss += self.itm_weight * itm_loss_value
            else:
                raise ValueError("ITM loss is only supported for 'concat' and 'mean' fusion techniques")

        # Log the total loss and individual losses for monitoring
        self.log('train_loss', loss)
        if 'CE' in self.loss_function: self.log('ce_loss', ce_loss)
        if 'SupCon' in self.loss_function: self.log('supcon_loss', supcon_loss)
        if 'NTXent' in self.loss_function: self.log('ntxent_loss', ntxent_loss)
        if 'ITM' in self.loss_function: self.log('itm_loss', itm_loss_value)

        return loss

    # Define the validation step
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, pixel_values, labels = batch.get('input_ids'), batch.get('attention_mask'), batch.get('pixel_values'), batch['label']
        logits, _ = self(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        self.validation_outputs.append({'preds': preds, 'labels': labels})
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss

    # At the end of validation epoch, calculate accuracy and F1 scores
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

    # Define the test step
    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, pixel_values, labels = batch.get('input_ids'), batch.get('attention_mask'), batch.get('pixel_values'), batch['label']
        logits, _ = self(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        self.test_outputs.append({'preds': preds, 'labels': labels})
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        return loss

    # At the end of the test epoch, calculate accuracy and F1 scores
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

    # Configure optimizers and schedulers
    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=self.eps)

        # We also use a scheduler that is supplied by transformers.
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.num_training_steps)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]

    # Extract embeddings for text or image inputs
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
