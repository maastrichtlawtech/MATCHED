import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning.pytorch as pl

from transformers import AutoModel, ViTModel, get_linear_schedule_with_warmup
from transformers import BertConfig, BertModel
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

# import bitsandbytes as bnb
# from deepspeed.ops.adam import DeepSpeedCPUAdam

from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report

from multimodalLayer import SupConLoss

class QFormer(nn.Module):
    def __init__(
        self,
        num_queries=32,
        d_model=768,
        num_attention_heads=12,
        num_hidden_layers=12,  # Set to 6 as per CLIPITMModel initialization
        intermediate_size=3072,
        cross_attention_frequency=1,  # Set to 1 to apply cross-attention in every layer
        dropout=0.1,
    ):
        super(QFormer, self).__init__()

        # Learnable query embeddings (similar to Blip-2's Q-Former)
        self.query_embeddings = nn.Parameter(torch.randn(1, num_queries, d_model))

        # Configuration for a Transformer with cross-attention
        config = BertConfig(
            hidden_size=d_model,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            hidden_act="gelu",
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            is_decoder=True,  # Enable cross-attention by setting is_decoder to True
            add_cross_attention=True,
        )

        # Initialize BertModel with the above configuration
        self.bert = BertModel(config)

        # Cross-attention frequency
        self.cross_attention_frequency = cross_attention_frequency

    def forward(self, image_embeddings):
        """
        image_embeddings: The output from the vision model (e.g., ViT).
        image_embeddings shape: (batch_size, seq_len_image, d_model)
        """
        batch_size, seq_len_image, d_model = image_embeddings.shape

        # Expand query embeddings to match the batch size
        query_embeddings = self.query_embeddings.expand(batch_size, -1, -1)  # [batch_size, num_queries, d_model]
        # print(f"Initial query_embeddings shape: {query_embeddings.shape}")  # Expected: [40, 32, 768]
        # print(f"Image embeddings shape: {image_embeddings.shape}")  # Expected: [40, 197, 768]

        # Create attention masks
        # Self-attention mask for queries (decoder input)
        attention_mask = torch.ones(batch_size, query_embeddings.size(1), device=query_embeddings.device)  # [batch_size, num_queries]
        # print(f"Attention mask shape: {attention_mask.shape}")  # Expected: [40, 32]

        # Encoder attention mask for image embeddings (encoder input)
        encoder_attention_mask = torch.ones(batch_size, seq_len_image, device=image_embeddings.device)  # [batch_size, seq_len_image]
        # print(f"Encoder attention mask shape: {encoder_attention_mask.shape}")  # Expected: [40, 197]

        # Get extended attention masks using BERT's utility function
        extended_attention_mask = self.bert.get_extended_attention_mask(
            attention_mask, attention_mask.shape, image_embeddings.device
        )  # Shape: [batch_size, 1, 1, num_queries]
        # print(f"Extended self-attention mask shape: {extended_attention_mask.shape}")  # Expected: [40, 1, 1, 32]

        # Repeat the self-attention mask for each attention head
        extended_attention_mask = extended_attention_mask.repeat(1, self.bert.config.num_attention_heads, 1, 1)  # [batch_size, num_heads, 1, num_queries]
        # print(f"Extended self-attention mask after repeat: {extended_attention_mask.shape}")  # Expected: [40, 12, 1, 32]

        # Create cross-attention mask: [batch_size, num_queries, seq_len_image]
        cross_attention_mask = torch.ones(batch_size, query_embeddings.size(1), seq_len_image, device=image_embeddings.device)  # [40, 32, 197]
        # print(f"Cross-Attention mask shape: {cross_attention_mask.shape}")  # Expected: [40, 32, 197]

        # Get extended cross-attention mask
        encoder_extended_cross_attention_mask = self.bert.get_extended_attention_mask(
            cross_attention_mask, cross_attention_mask.shape, image_embeddings.device
        )  # Shape: [batch_size, 1, num_queries, seq_len_image]
        # print(f"Encoder extended cross-attention mask shape before repeat: {encoder_extended_cross_attention_mask.shape}")  # Expected: [40, 1, 32, 197]

        # Repeat the cross-attention mask for each attention head
        encoder_extended_cross_attention_mask = encoder_extended_cross_attention_mask.repeat(1, self.bert.config.num_attention_heads, 1, 1)  # [batch_size, num_heads, num_queries, seq_len_image]
        # print(f"Encoder extended cross-attention mask shape after repeat: {encoder_extended_cross_attention_mask.shape}")  # Expected: [40, 12, 32, 197]

        # Initialize hidden states
        hidden_states = query_embeddings  # [batch_size, num_queries, d_model]
        # print(f"Hidden states shape: {hidden_states.shape}")  # Expected: [40, 32, 768]

        # Iterate through each BERT layer
        for i, layer_module in enumerate(self.bert.encoder.layer):
            if i % self.cross_attention_frequency == 0:
                # print(f"Layer {i}: Applying Cross-Attention between query_embeddings and image_embeddings.")
                # Apply cross-attention
                outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,  # Self-attention mask for queries
                    encoder_hidden_states=image_embeddings,
                    encoder_attention_mask=encoder_extended_cross_attention_mask,  # Cross-attention mask for image embeddings
                )
            else:
                # print(f"Layer {i}: Applying Self-Attention only on query_embeddings.")
                # Apply only self-attention by setting encoder_hidden_states=None
                outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,  # Self-attention mask for queries
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                )

            # Update hidden states
            hidden_states = outputs[0]  # [batch_size, num_queries, d_model]
            # print(f"After Layer {i}, query_embeddings shape: {hidden_states.shape}")  # Expected: [40, 32, 768]

        return hidden_states  # Final query embeddings

class BLIP2Model(pl.LightningModule):
    def __init__(
        self,
        weight_decay,
        eps,
        warmup_steps,
        num_training_steps,
        text_model_name='johngiorgi/declutr-small',
        image_model_name='google/vit-base-patch16-224',
        t5_model_name='google/flan-t5-small',
        learning_rate=0.00001,
        num_negatives=5,
        temperature=0.5,
        num_query_tokens=32,
        qformer_hidden_size=768,
        cross_attention_frequency=1,  # Set to 1 to align with CustomQFormer
        **kwargs
    ):
        super(BLIP2Model, self).__init__()
        # Text Model
        self.text_model = AutoModel.from_pretrained(text_model_name)
        # Vision Model
        self.image_model = ViTModel.from_pretrained(image_model_name)
        # Custom Q-Former with cross-attention in every layer
        self.qformer = QFormer(
            num_queries=num_query_tokens,
            d_model=qformer_hidden_size,
            num_attention_heads=12,
            num_hidden_layers=12,
            cross_attention_frequency=cross_attention_frequency,
        )

        # Loss components
        self.itm_criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.num_negatives = num_negatives
        self.temperature = temperature
        self.weight_decay = weight_decay
        self.eps = eps
        self.warmup_steps = warmup_steps
        self.num_training_steps = num_training_steps

        # Initialize T5 model
        self.t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        # Projects from 768 (Q-Former hidden size) to 512 (T5 hidden size).
        self.query_proj_t5 = nn.Linear(qformer_hidden_size, self.t5_model.config.d_model)


    def forward(self, input_ids, attention_mask, pixel_values, pos_input_ids_t5, pos_attention_mask_t5, neg_pixel_values=None):
        # Process text embeddings
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings = text_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        # print(f"Text embeddings shape: {text_embeddings.shape}")

        # Process image embeddings
        image_outputs = self.image_model(pixel_values=pixel_values)
        image_embeddings = image_outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        # print(f"Image embeddings shape: {image_embeddings.shape}")

        # Pass through the custom Q-Former
        query_embeddings = self.qformer(image_embeddings)
        query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
        # print(f"Query embeddings shape after QFormer: {query_embeddings.shape}")

        # Process negative image embeddings if provided
        neg_image_embeddings = None
        if neg_pixel_values is not None:
            neg_image_embeddings = self._process_negative_images(neg_pixel_values)
            # print(f"Negative image embeddings shape: {neg_image_embeddings.shape}")

        # Use query_embeddings as encoder outputs for T5
        # T5 expects encoder outputs in the shape (batch_size, seq_len, hidden_size)
        # Our query_embeddings are already in this shape (batch_size, num_queries, hidden_size)

        # Project unnormalized query_embeddings to match T5 hidden size
        projected_query_embeddings = self.query_proj_t5(query_embeddings)
        # Create T5 encoder outputs
        encoder_outputs = BaseModelOutput(last_hidden_state=projected_query_embeddings)
        # Create encoder attention mask
        encoder_attention_mask = torch.ones(
            projected_query_embeddings.size()[:-1],
            dtype=torch.long,
            device=projected_query_embeddings.device
            )
        # Pass through T5 model
        t5_outputs = self.t5_model(
            attention_mask=encoder_attention_mask,
            encoder_outputs=encoder_outputs,
            labels=pos_input_ids_t5,
            return_dict=True,
        )
        text_generation_loss = t5_outputs.loss

        return text_embeddings, query_embeddings, neg_image_embeddings, text_generation_loss

    def _process_negative_images(self, neg_pixel_values):
        batch_size, num_negatives, _, _, _ = neg_pixel_values.shape
        neg_pixel_values = neg_pixel_values.view(-1, *neg_pixel_values.shape[2:])  # (batch_size * num_negatives, C, H, W)
        neg_image_outputs = self.image_model(pixel_values=neg_pixel_values)
        neg_image_embeddings = neg_image_outputs.last_hidden_state.mean(dim=1)  # (batch_size * num_negatives, d_model)
        neg_image_embeddings = F.normalize(neg_image_embeddings, p=2, dim=-1)
        neg_image_embeddings = neg_image_embeddings.view(batch_size, num_negatives, -1)  # (batch_size, num_negatives, d_model)
        return neg_image_embeddings

    def compute_clip_loss(self, pos_text_embeddings, query_embeddings, neg_image_embeddings):
        # Compute similarities
        # print(f"Positive text embeddings shape: {pos_text_embeddings.shape}")
        # print(f"Query embeddings shape: {query_embeddings.shape}")
        # print(f"Negative image embeddings shape: {neg_image_embeddings.shape}")
        
        # Positive similarities between text and positive image queries
        pos_sim = torch.einsum('bqd,bd->bq', query_embeddings, pos_text_embeddings) / self.temperature  # (batch_size, num_queries)
        pos_sim = pos_sim.max(dim=1, keepdim=True).values  # (batch_size, 1)

        # Negative similarities between text and negative images
        neg_sim = torch.einsum('bd,bnd->bn', pos_text_embeddings, neg_image_embeddings) / self.temperature  # (batch_size, num_negatives)

        # Combine logits
        logits = torch.cat([pos_sim, neg_sim], dim=1)  # (batch_size, 1 + num_negatives)
        # print(f"Logits shape (for CLIP loss): {logits.shape}")

        labels = torch.zeros(logits.size(0), dtype=torch.long).to(logits.device)  # (batch_size,)

        # Compute CLIP loss using cross-entropy
        clip_loss = self.itm_criterion(logits, labels)
        return clip_loss

    def compute_itm_loss(self, pos_text_embeddings, query_embeddings, neg_image_embeddings):
        # Compute positive scores
        pos_sim = torch.einsum('bqd,bd->bq', query_embeddings, pos_text_embeddings) / self.temperature  # (batch_size, num_queries)
        pos_scores = pos_sim.max(dim=1).values  # (batch_size,)

        # Compute negative scores
        neg_sim = torch.einsum('bd,bnd->bn', pos_text_embeddings, neg_image_embeddings) / self.temperature  # (batch_size, num_negatives)
        neg_scores = neg_sim.view(-1)  # (batch_size * num_negatives,)

        # Combine scores and labels
        scores = torch.cat([pos_scores, neg_scores], dim=0)  # (batch_size + batch_size * num_negatives,)
        labels = torch.cat([
            torch.ones(pos_scores.size(0), device=scores.device),
            torch.zeros(neg_scores.size(0), device=scores.device)
        ], dim=0)  # (batch_size + batch_size * num_negatives,)
        
        # print(f"Scores shape: {scores.shape}, Labels shape: {labels.shape}")

        # Compute binary cross-entropy loss with logits
        itm_loss = F.binary_cross_entropy_with_logits(scores, labels)
        return itm_loss

    def training_step(self, batch, batch_idx):
        pos_text_embeddings, query_embeddings, neg_image_embeddings, text_generation_loss = self(
            batch['pos_input_ids'],
            batch['pos_attention_mask'],
            batch['pos_pixel_values'],
            batch['pos_input_ids_t5'],
            batch['pos_attention_mask_t5'],
            neg_pixel_values=batch.get('neg_pixel_values')
            )

        clip_loss = self.compute_clip_loss(pos_text_embeddings, query_embeddings, neg_image_embeddings)
        itm_loss = self.compute_itm_loss(pos_text_embeddings, query_embeddings, neg_image_embeddings)

        total_loss = clip_loss + itm_loss + text_generation_loss
        # print(f"Training Step: CLIP Loss: {clip_loss.item()}, ITM Loss: {itm_loss.item()}, Total Loss: {total_loss.item()}")
        self.log('clip_loss', clip_loss)
        self.log('itm_loss', itm_loss)
        self.log('text_generation_loss', text_generation_loss)
        self.log('train_loss', total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        pos_text_embeddings, query_embeddings, neg_image_embeddings, text_generation_loss = self(
            batch['pos_input_ids'],
            batch['pos_attention_mask'],
            batch['pos_pixel_values'],
            batch['pos_input_ids_t5'],
            batch['pos_attention_mask_t5'],
            neg_pixel_values=batch.get('neg_pixel_values')
            )

        clip_loss = self.compute_clip_loss(pos_text_embeddings, query_embeddings, neg_image_embeddings)
        itm_loss = self.compute_itm_loss(pos_text_embeddings, query_embeddings, neg_image_embeddings)

        total_loss = clip_loss + itm_loss + text_generation_loss
        # print(f"Training Step: CLIP Loss: {clip_loss.item()}, ITM Loss: {itm_loss.item()}, Total Loss: {total_loss.item()}")
        self.log('clip_loss', clip_loss)
        self.log('itm_loss', itm_loss)
        self.log('text_generation_loss', text_generation_loss)
        self.log('val_loss', total_loss)
        return total_loss

    def test_step(self, batch, batch_idx):
        pos_text_embeddings, query_embeddings, neg_image_embeddings, text_generation_loss = self(
            batch['pos_input_ids'],
            batch['pos_attention_mask'],
            batch['pos_pixel_values'],
            batch['pos_input_ids_t5'],
            batch['pos_attention_mask_t5'],
            neg_pixel_values=batch.get('neg_pixel_values')
            )

        clip_loss = self.compute_clip_loss(pos_text_embeddings, query_embeddings, neg_image_embeddings)
        itm_loss = self.compute_itm_loss(pos_text_embeddings, query_embeddings, neg_image_embeddings)

        total_loss = clip_loss + itm_loss + text_generation_loss
        # print(f"Training Step: CLIP Loss: {clip_loss.item()}, ITM Loss: {itm_loss.item()}, Total Loss: {total_loss.item()}")
        self.log('clip_loss', clip_loss)
        self.log('itm_loss', itm_loss)
        self.log('text_generation_loss', text_generation_loss)
        self.log('test_loss', total_loss)
        return total_loss

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=self.eps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.num_training_steps
        )
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def on_validation_epoch_end(self):
        if 'val_loss' in self.trainer.callback_metrics:
            avg_val_loss = self.trainer.callback_metrics['val_loss'].mean()
            # print(f"[Validation End] Average validation loss: {avg_val_loss}")
            self.log('avg_val_loss', avg_val_loss)

    def on_test_epoch_end(self):
        if 'test_loss' in self.trainer.callback_metrics:
            avg_test_loss = self.trainer.callback_metrics['test_loss'].mean()
            # print(f"[Test End] Average test loss: {avg_test_loss}")
            self.log('avg_test_loss', avg_test_loss)

class BLIP2ConditionalModel(pl.LightningModule):
    def __init__(
        self,
        weight_decay,
        eps,
        warmup_steps,
        num_training_steps,
        text_model_name='johngiorgi/declutr-small',
        image_model_name='google/vit-base-patch16-224',
        t5_model_name='google/flan-t5-small',
        learning_rate=0.00001,
        num_negatives=5,
        temperature=0.5,
        num_query_tokens=32,
        qformer_hidden_size=768,
        cross_attention_frequency=1,  # Set to 1 to align with CustomQFormer
        **kwargs
    ):
        super(BLIP2Model, self).__init__()
        # Text Model
        self.text_model = AutoModel.from_pretrained(text_model_name)
        # Vision Model
        self.image_model = ViTModel.from_pretrained(image_model_name)
        # Custom Q-Former with cross-attention in every layer
        self.qformer = QFormer(
            num_queries=num_query_tokens,
            d_model=qformer_hidden_size,
            num_attention_heads=12,
            num_hidden_layers=12,
            cross_attention_frequency=cross_attention_frequency,
        )

        # Loss components
        self.itm_criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.num_negatives = num_negatives
        self.temperature = temperature
        self.weight_decay = weight_decay
        self.eps = eps
        self.warmup_steps = warmup_steps
        self.num_training_steps = num_training_steps

        # Initialize T5 model
        self.t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)

        # Ensure that pad_token_id is set
        if self.t5_model.config.pad_token_id is None:
            self.t5_model.config.pad_token_id = self.t5_model.config.eos_token_id

        if self.t5_model.config.decoder_start_token_id is None:
            self.t5_model.config.decoder_start_token_id = self.t5_model.config.pad_token_id


        # Projects from 768 (Q-Former hidden size) to 512 (T5 hidden size).
        self.query_proj_t5 = nn.Linear(qformer_hidden_size, self.t5_model.config.d_model)


    def forward(self, input_ids, attention_mask, pixel_values, conditional_input_ids, conditional_attention_mask, target_input_ids,
                target_attention_mask, neg_pixel_values=None):
        # Process text embeddings for CLIP and ITM losses (full text)
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings = text_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

        # Process image embeddings
        image_outputs = self.image_model(pixel_values=pixel_values)
        image_embeddings = image_outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)

        # Pass through the custom Q-Former
        query_embeddings = self.qformer(image_embeddings)
        query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)

        # Process negative image embeddings if provided
        neg_image_embeddings = None
        if neg_pixel_values is not None:
            neg_image_embeddings = self._process_negative_images(neg_pixel_values)

        # Text Generation Loss
        # Encode conditional text using T5 encoder
        conditional_text_outputs = self.t5_model.encoder(
            input_ids=conditional_input_ids,
            attention_mask=conditional_attention_mask,
            return_dict=True,
        )
        conditional_text_embeddings = conditional_text_outputs.last_hidden_state  # (batch_size, seq_len_conditional, hidden_size)

        # Project query embeddings to match T5 hidden size
        projected_query_embeddings = self.query_proj_t5(query_embeddings)

        # Combine conditional text embeddings and projected query embeddings
        combined_encoder_embeddings = torch.cat([conditional_text_embeddings, projected_query_embeddings], dim=1)  # Concatenate along sequence length

        # Create encoder attention mask
        projected_query_attention_mask = torch.ones(
            projected_query_embeddings.size()[:-1],
            dtype=torch.long,
            device=self.device
        )
        combined_attention_mask = torch.cat([conditional_attention_mask, projected_query_attention_mask], dim=1)

        # Prepare labels for T5 (target text)
        labels = target_input_ids.clone()
        labels[labels == self.t5_model.config.pad_token_id] = -100  # Mask padding tokens in loss computation

        # Pass through T5 model
        t5_outputs = self.t5_model(
            encoder_outputs=BaseModelOutput(last_hidden_state=combined_encoder_embeddings),
            attention_mask=combined_attention_mask,
            labels=labels,
            return_dict=True,
        )
        text_generation_loss = t5_outputs.loss

        return text_embeddings, query_embeddings, neg_image_embeddings, text_generation_loss


    def _process_negative_images(self, neg_pixel_values):
        batch_size, num_negatives, _, _, _ = neg_pixel_values.shape
        neg_pixel_values = neg_pixel_values.view(-1, *neg_pixel_values.shape[2:])  # (batch_size * num_negatives, C, H, W)
        neg_image_outputs = self.image_model(pixel_values=neg_pixel_values)
        neg_image_embeddings = neg_image_outputs.last_hidden_state.mean(dim=1)  # (batch_size * num_negatives, d_model)
        neg_image_embeddings = F.normalize(neg_image_embeddings, p=2, dim=-1)
        neg_image_embeddings = neg_image_embeddings.view(batch_size, num_negatives, -1)  # (batch_size, num_negatives, d_model)
        return neg_image_embeddings

    def compute_clip_loss(self, pos_text_embeddings, query_embeddings, neg_image_embeddings):
        # Compute similarities
        # print(f"Positive text embeddings shape: {pos_text_embeddings.shape}")
        # print(f"Query embeddings shape: {query_embeddings.shape}")
        # print(f"Negative image embeddings shape: {neg_image_embeddings.shape}")
        
        # Positive similarities between text and positive image queries
        pos_sim = torch.einsum('bqd,bd->bq', query_embeddings, pos_text_embeddings) / self.temperature  # (batch_size, num_queries)
        pos_sim = pos_sim.max(dim=1, keepdim=True).values  # (batch_size, 1)

        # Negative similarities between text and negative images
        neg_sim = torch.einsum('bd,bnd->bn', pos_text_embeddings, neg_image_embeddings) / self.temperature  # (batch_size, num_negatives)

        # Combine logits
        logits = torch.cat([pos_sim, neg_sim], dim=1)  # (batch_size, 1 + num_negatives)
        # print(f"Logits shape (for CLIP loss): {logits.shape}")

        labels = torch.zeros(logits.size(0), dtype=torch.long).to(logits.device)  # (batch_size,)

        # Compute CLIP loss using cross-entropy
        clip_loss = self.itm_criterion(logits, labels)
        return clip_loss

    def compute_itm_loss(self, pos_text_embeddings, query_embeddings, neg_image_embeddings):
        # Compute positive scores
        pos_sim = torch.einsum('bqd,bd->bq', query_embeddings, pos_text_embeddings) / self.temperature  # (batch_size, num_queries)
        pos_scores = pos_sim.max(dim=1).values  # (batch_size,)

        # Compute negative scores
        neg_sim = torch.einsum('bd,bnd->bn', pos_text_embeddings, neg_image_embeddings) / self.temperature  # (batch_size, num_negatives)
        neg_scores = neg_sim.view(-1)  # (batch_size * num_negatives,)

        # Combine scores and labels
        scores = torch.cat([pos_scores, neg_scores], dim=0)  # (batch_size + batch_size * num_negatives,)
        labels = torch.cat([
            torch.ones(pos_scores.size(0), device=scores.device),
            torch.zeros(neg_scores.size(0), device=scores.device)
        ], dim=0)  # (batch_size + batch_size * num_negatives,)
        
        # print(f"Scores shape: {scores.shape}, Labels shape: {labels.shape}")

        # Compute binary cross-entropy loss with logits
        itm_loss = F.binary_cross_entropy_with_logits(scores, labels)
        return itm_loss


    def training_step(self, batch, batch_idx):
        pos_text_embeddings, query_embeddings, neg_image_embeddings, text_generation_loss = self(
            input_ids=batch['pos_input_ids'],
            attention_mask=batch['pos_attention_mask'],
            pixel_values=batch['pos_pixel_values'],
            conditional_input_ids=batch['conditional_input_ids'],
            conditional_attention_mask=batch['conditional_attention_mask'],
            target_input_ids=batch['target_input_ids'],
            target_attention_mask=batch['target_attention_mask'],
            neg_pixel_values=batch.get('neg_pixel_values')
        )

        clip_loss = self.compute_clip_loss(pos_text_embeddings, query_embeddings, neg_image_embeddings)
        itm_loss = self.compute_itm_loss(pos_text_embeddings, query_embeddings, neg_image_embeddings)

        train_loss = clip_loss + itm_loss + text_generation_loss
        self.log('train_clip_loss', clip_loss)
        self.log('train_itm_loss', itm_loss)
        self.log('train_text_generation_loss', text_generation_loss)
        self.log('train_loss', train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx):
        pos_text_embeddings, query_embeddings, neg_image_embeddings, text_generation_loss = self(
            input_ids=batch['pos_input_ids'],
            attention_mask=batch['pos_attention_mask'],
            pixel_values=batch['pos_pixel_values'],
            conditional_input_ids=batch['conditional_input_ids'],
            conditional_attention_mask=batch['conditional_attention_mask'],
            target_input_ids=batch['target_input_ids'],
            target_attention_mask=batch['target_attention_mask'],
            neg_pixel_values=batch.get('neg_pixel_values')
        )

        clip_loss = self.compute_clip_loss(pos_text_embeddings, query_embeddings, neg_image_embeddings)
        itm_loss = self.compute_itm_loss(pos_text_embeddings, query_embeddings, neg_image_embeddings)

        val_loss = clip_loss + itm_loss + text_generation_loss
        self.log('val_clip_loss', clip_loss)
        self.log('val_itm_loss', itm_loss)
        self.log('val_text_generation_loss', text_generation_loss)
        self.log('val_loss', val_loss)
        return val_loss

    def test_step(self, batch, batch_idx):
        pos_text_embeddings, query_embeddings, neg_image_embeddings, text_generation_loss = self(
            input_ids=batch['pos_input_ids'],
            attention_mask=batch['pos_attention_mask'],
            pixel_values=batch['pos_pixel_values'],
            conditional_input_ids=batch['conditional_input_ids'],
            conditional_attention_mask=batch['conditional_attention_mask'],
            target_input_ids=batch['target_input_ids'],
            target_attention_mask=batch['target_attention_mask'],
            neg_pixel_values=batch.get('neg_pixel_values')
        )

        clip_loss = self.compute_clip_loss(pos_text_embeddings, query_embeddings, neg_image_embeddings)
        itm_loss = self.compute_itm_loss(pos_text_embeddings, query_embeddings, neg_image_embeddings)

        test_loss = clip_loss + itm_loss + text_generation_loss
        self.log('test_clip_loss', clip_loss)
        self.log('test_itm_loss', itm_loss)
        self.log('test_text_generation_loss', text_generation_loss)
        self.log('test_loss', test_loss)
        return test_loss


    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=self.eps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.num_training_steps
        )
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def on_validation_epoch_end(self):
        if 'val_loss' in self.trainer.callback_metrics:
            avg_val_loss = self.trainer.callback_metrics['val_loss'].mean()
            # print(f"[Validation End] Average validation loss: {avg_val_loss}")
            self.log('avg_val_loss', avg_val_loss)

    def on_test_epoch_end(self):
        if 'test_loss' in self.trainer.callback_metrics:
            avg_test_loss = self.trainer.callback_metrics['test_loss'].mean()
            # print(f"[Test End] Average test loss: {avg_test_loss}")
            self.log('avg_test_loss', avg_test_loss)

class BLIP2CheckpointModel(pl.LightningModule):
    def __init__(self, model, processor, max_length=512, learning_rate=1e-4):
        super().__init__()
        self.model = model
        self.processor = processor
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.save_hyperparameters(ignore=['model', 'processor'])  # Saves hyperparameters for checkpoints

        # Ensure the model is in training mode
        self.model.train()

    def compute_text_embeddings(self, target_texts):
        device = self.device  # Current device
        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for target_text in target_texts:
            # Tokenize the target text with truncation and padding
            tokenized_text = self.processor.tokenizer(
                target_text,
                return_tensors='pt',
                padding="max_length",  # Pad to the maximum length
                truncation=True,       # Truncate to the maximum length
                max_length=self.max_length  # Maximum length of 512 tokens
            )
            input_ids = tokenized_text['input_ids'].squeeze(0)  # [seq_len]
            attention_mask = tokenized_text['attention_mask'].squeeze(0)  # [seq_len]

            # Prepare labels
            labels = input_ids.clone()
            labels[labels == self.processor.tokenizer.pad_token_id] = -100  # Replace pad token IDs with -100 for loss computation

            input_ids_list.append(input_ids.to(device))
            attention_mask_list.append(attention_mask.to(device))
            labels_list.append(labels.to(device))

        return input_ids_list, attention_mask_list, labels_list

    def forward(self, pixel_values, input_ids_list, attention_mask_list, labels_list):
        # Ensure pixel_values are in the correct data type for the model
        pixel_values = pixel_values.to(dtype=self.model.dtype)
        total_loss = 0.0
        batch_size = pixel_values.size(0)

        for idx in range(batch_size):
            image = pixel_values[idx].unsqueeze(0)  # [1, 3, H, W]
            input_ids = input_ids_list[idx].unsqueeze(0)  # [1, seq_len]
            attention_mask = attention_mask_list[idx].unsqueeze(0)  # [1, seq_len]
            labels = labels_list[idx].unsqueeze(0)  # [1, seq_len]

            # Forward pass through the model
            outputs = self.model(
                pixel_values=image,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

            total_loss += loss

        # Average loss over the batch
        total_loss = total_loss / batch_size
        return total_loss

    def training_step(self, batch, batch_idx):
        print(f"Training Step: Memory allocated before forward pass: {torch.cuda.memory_allocated()/1e9} GB")
        print(f"Training Step: Max Memory allocated before forward pass: {torch.cuda.max_memory_allocated()/1e9} GB")
        images = batch['images']
        target_texts = batch['target_texts']

        # Process images using the processor
        image_inputs = self.processor(images=images, return_tensors='pt').to(self.device)
        pixel_values = image_inputs['pixel_values']  # [batch_size, 3, H, W]

        # Compute text embeddings (input_ids, attention_mask, labels)
        input_ids_list, attention_mask_list, labels_list = self.compute_text_embeddings(target_texts)

        # Forward pass
        loss = self.forward(pixel_values, input_ids_list, attention_mask_list, labels_list)
        self.log('train_loss', loss, batch_size=len(images))
        print(f"Training Step: Memory allocated after forward pass: {torch.cuda.memory_allocated()/1e9} GB")
        print(f"Training Step: Max Memory allocated after forward pass: {torch.cuda.max_memory_allocated()/1e9} GB")
        return loss

    def test_step(self, batch, batch_idx):
        print(f"Test Step: Memory allocated before forward pass: {torch.cuda.memory_allocated()/1e9} GB")
        print(f"Test Step: Max Memory allocated before forward pass: {torch.cuda.max_memory_allocated()/1e9} GB")
        images = batch['images']
        target_texts = batch['target_texts']

        # Process images using the processor
        image_inputs = self.processor(images=images, return_tensors='pt').to(self.device)
        pixel_values = image_inputs['pixel_values']  # [batch_size, 3, H, W]

        # Compute text embeddings (input_ids, attention_mask, labels)
        input_ids_list, attention_mask_list, labels_list = self.compute_text_embeddings(target_texts)

        # Forward pass
        loss = self.forward(pixel_values, input_ids_list, attention_mask_list, labels_list)
        self.log('test_loss', loss, batch_size=len(images))
        print(f"Test Step: Memory allocated after forward pass: {torch.cuda.memory_allocated()/1e9} GB")
        print(f"Test Step: Max Memory allocated after forward pass: {torch.cuda.max_memory_allocated()/1e9} GB")
        return loss

    def validation_step(self, batch, batch_idx):
        print(f"Validation Step: Memory allocated before forward pass: {torch.cuda.memory_allocated()/1e9} GB")
        print(f"Validation Step: Max Memory allocated before forward pass: {torch.cuda.max_memory_allocated()/1e9} GB")
        images = batch['images']
        target_texts = batch['target_texts']

        # Process images using the processor
        image_inputs = self.processor(images=images, return_tensors='pt').to(self.device)
        pixel_values = image_inputs['pixel_values']  # [batch_size, 3, H, W]

        # Compute text embeddings (input_ids, attention_mask, labels)
        input_ids_list, attention_mask_list, labels_list = self.compute_text_embeddings(target_texts)

        # Forward pass
        loss = self.forward(pixel_values, input_ids_list, attention_mask_list, labels_list)
        self.log('val_loss', loss, batch_size=len(images))
        print(f"Validation Step: Memory allocated after forward pass: {torch.cuda.memory_allocated()/1e9} GB")
        print(f"Validation Step: Max Memory allocated after forward pass: {torch.cuda.max_memory_allocated()/1e9} GB")
        return loss

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        # optimizer = bnb.optim.AdamW8bit(self.parameters(), lr=self.learning_rate)
        # optimizer = DeepSpeedCPUAdam(self.parameters(), lr=self.learning_rate)
        # return optimizer
        return None

    def generate_text(self, images, text_inputs=None, max_gen_length=50):
        """
        Generate text outputs for the given images and optional text inputs.

        Args:
            images (list): List of PIL images.
            text_inputs (list, optional): List of text prompts. Defaults to None.
            max_gen_length (int, optional): Maximum length of generated text. Defaults to 50.

        Returns:
            list: List of generated texts.
        """
        device = self.device  # Current device

        # Process images
        image_inputs = self.processor(images=images, return_tensors='pt').to(device)
        pixel_values = image_inputs['pixel_values']  # [batch_size, 3, H, W]

        generated_texts = []

        batch_size = len(images)

        for idx in range(batch_size):
            image = pixel_values[idx].unsqueeze(0)  # [1, 3, H, W]
            text_input = text_inputs[idx] if text_inputs is not None else ""

            # Tokenize the text input
            tokenized_input = self.processor.tokenizer(
                text_input,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            input_ids = tokenized_input['input_ids'].to(device)  # [1, seq_len]
            attention_mask = tokenized_input['attention_mask'].to(device)  # [1, seq_len]

            # Generate output
            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values=image,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_gen_length,
                    do_sample=False
                )

            generated_text = self.processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            generated_texts.append(generated_text)

        return generated_texts
    
    def generate_embeddings(self, images=None, texts=None):
        """
        Generate embeddings for the given images and/or texts.

        Parameters:
            images (list): List of PIL images.
            texts (list): List of text inputs.

        Returns:
            dict: A dictionary containing 'image_embeddings' and/or 'text_embeddings'.
        """
        device = self.device  # Current device
        results = {}

        if images is not None:
            # Process images using the processor
            image_inputs = self.processor(images=images, return_tensors='pt').to(device)
            pixel_values = image_inputs['pixel_values']  # [batch_size, 3, H, W]

            with torch.no_grad():
                # Obtain image embeddings from the vision encoder and Q-Former
                vision_outputs = self.model.vision_model(pixel_values=pixel_values)
                image_embeds = vision_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

                # Forward through Q-Former
                # Prepare query tokens
                query_tokens = self.model.query_tokens.unsqueeze(0).expand(pixel_values.size(0), -1, -1)
                attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=device)

                # Get image embeddings from Q-Former
                encoder_outputs = self.model.qformer(
                    inputs_embeds=query_tokens,
                    attention_mask=attention_mask,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=torch.ones(
                        image_embeds.size()[:-1], dtype=torch.long, device=device
                    ),
                    return_dict=True,
                )
                image_embeddings = encoder_outputs.last_hidden_state  # [batch_size, num_queries, hidden_size]

                # Optionally, pool or reshape embeddings
                image_embeddings = image_embeddings.mean(dim=1)  # [batch_size, hidden_size]

                results['image_embeddings'] = image_embeddings

        if texts is not None:
            # Tokenize texts
            text_inputs = self.processor.tokenizer(
                texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(device)
            input_ids = text_inputs['input_ids']
            attention_mask = text_inputs['attention_mask']

            with torch.no_grad():
                # Obtain text embeddings from the language model's embedding layer
                text_embeddings = self.model.language_model.encoder.embed_tokens(input_ids)  # [batch_size, seq_len, hidden_size]

                # Optionally, get hidden states from the language model
                outputs = self.model.language_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
                text_hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

                # For consistency, you might want to average or pool over seq_len
                text_embeddings = text_hidden_states.mean(dim=1)  # [batch_size, hidden_size]

                results['text_embeddings'] = text_embeddings

        return results

class FineTuneBLIP2Classifier(pl.LightningModule):
    def __init__(self, pretrained_model, finetune_mode, num_classes, weight_decay, eps, warmup_steps, num_training_steps, 
                learning_rate, loss_fn, temperature):        
        super(FineTuneBLIP2Classifier, self).__init__()
        self.text_model = pretrained_model.text_model
        self.image_model = pretrained_model.image_model
        self.qformer = pretrained_model.qformer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.eps = eps
        self.warmup_steps = warmup_steps
        self.num_training_steps = num_training_steps
        self.loss_fn_name = loss_fn
        self.finetune_mode = finetune_mode
        self.temperature = temperature

        # Classification head
        self.classifier = nn.Linear(self.text_model.config.hidden_size, num_classes)

        self.validation_outputs = []  # To store validation outputs
        self.test_outputs = []  # To store test outputs

        # Loss function
        self.ce_loss = nn.CrossEntropyLoss()

        if self.loss_fn_name == "CE+SupCon":
            self.supcon_loss = SupConLoss(self.temperature)

    def forward(self, input_ids, attention_mask, pixel_values):
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings = text_outputs.last_hidden_state[:, 0, :]  # [CLS] token

        image_outputs = self.image_model(pixel_values=pixel_values)
        image_embeddings = image_outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)

        # Pass through the custom Q-Former
        query_embeddings = self.qformer(image_embeddings)
        query_embeddings = query_embeddings.mean(dim=1)  # Mean pooling over queries

        # Take the mean of text and image embeddings
        embeddings = (text_embeddings + query_embeddings) / 2

        logits = self.classifier(embeddings)
        return logits, embeddings

    def training_step(self, batch, batch_idx):
        logits, embeddings = self(
            batch["input_ids"], batch["attention_mask"], batch["pixel_values"]
        )
        loss = self.ce_loss(logits, batch["labels"])

        if self.loss_fn_name == "CE+SupCon":
            features = F.normalize(embeddings, dim=1)
            supcon_loss = self.supcon_loss(features, batch["labels"])
            loss += supcon_loss

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, embeddings = self(
            batch["input_ids"], batch["attention_mask"], batch["pixel_values"]
        )
        loss = self.ce_loss(logits, batch["labels"])

        if self.loss_fn_name == "CE+SupCon":
            features = F.normalize(embeddings, dim=1)
            supcon_loss = self.supcon_loss(features, batch["labels"])
            loss += supcon_loss

        preds = torch.argmax(logits, dim=1)
        # acc = (preds == batch["labels"]).float().mean()
        self.validation_outputs.append({'preds': preds, 'labels': batch["labels"]})

        self.log("val_loss", loss)
        # self.log("val_acc", acc)
        return loss

    def test_step(self, batch, batch_idx):
        logits, embeddings = self(
            batch["input_ids"], batch["attention_mask"], batch["pixel_values"]
        )
        loss = self.ce_loss(logits, batch["labels"])

        preds = torch.argmax(logits, dim=1)
        # acc = (preds == batch["labels"]).float().mean()
        self.test_outputs.append({'preds': preds, 'labels': batch["labels"]})

        self.log("test_loss", loss)
        # self.log("test_acc", acc)
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

    def configure_optimizers(self):
        # Freeze layers if needed
        if self.finetune_mode == "finetune_layers":
            for name, param in self.text_model.named_parameters():
                layer_number = int(name.split(".")[2]) if "layer" in name else None
                if layer_number not in args.layers_to_finetune:
                    param.requires_grad = False

            for name, param in self.image_model.named_parameters():
                layer_number = int(name.split(".")[2]) if "layer" in name else None
                if layer_number not in args.layers_to_finetune:
                    param.requires_grad = False

            for name, param in self.qformer.named_parameters():
                layer_number = int(name.split(".")[2]) if "layer" in name else None
                if layer_number not in args.layers_to_finetune:
                    param.requires_grad = False

        elif self.finetune_mode == "all":
            # Unfreeze all layers
            for param in self.text_model.parameters():
                param.requires_grad = True
            for param in self.image_model.parameters():
                param.requires_grad = True
            for param in self.qformer.parameters():
                param.requires_grad = True

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=self.learning_rate, eps=self.eps
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.num_training_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]