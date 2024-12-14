import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from transformers import AutoModel, ViTModel, get_linear_schedule_with_warmup
from transformers import CLIPProcessor

from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report

from multimodalLayer import SupConLoss

class CLIPModel(pl.LightningModule):
    def __init__(self, weight_decay, eps, warmup_steps, num_training_steps, text_model_name='johngiorgi/declutr-small', 
                image_model_name='google/vit-base-patch16-224', learning_rate=0.00001, num_negatives=5, temperature=0.5):
        super(CLIPModel, self).__init__()
        self.text_model = AutoModel.from_pretrained(text_model_name)
        self.image_model = ViTModel.from_pretrained(image_model_name)
        self.learning_rate = learning_rate
        self.num_negatives = num_negatives
        self.temperature = temperature
        self.weight_decay = weight_decay
        self.eps = eps
        self.warmup_steps = warmup_steps
        self.num_training_steps = num_training_steps

        # Store outputs for validation and testing
        self.validation_outputs = []
        self.test_outputs = []
        
    def forward(self, input_ids, attention_mask, pixel_values):
        """
        # Get text embeddings
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings = text_outputs.last_hidden_state[:, 0, :]  # Use CLS token embedding
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)  # Normalize embeddings
        """
        # Get text embeddings
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        eos_mask = input_ids.eq(self.text_model.config.eos_token_id)
        eos_indices = eos_mask.nonzero(as_tuple=False)
        text_embeddings = text_outputs.last_hidden_state[eos_indices[:, 0], eos_indices[:, 1]]
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)  # Normalize embeddings
        
        # Get image embeddings
        image_outputs = self.image_model(pixel_values=pixel_values)
        image_embeddings = image_outputs.last_hidden_state[:, 0, :]  # Use CLS token embedding
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)  # Normalize embeddings
        
        return text_embeddings, image_embeddings

    def compute_loss(self, pos_text_embeddings, pos_image_embeddings, neg_text_embeddings, neg_image_embeddings):
        # Normalize embeddings
        pos_text_embeddings = F.normalize(pos_text_embeddings, p=2, dim=1)
        pos_image_embeddings = F.normalize(pos_image_embeddings, p=2, dim=1)
        neg_text_embeddings = F.normalize(neg_text_embeddings, p=2, dim=2)  # Normalized over the last dimension
        neg_image_embeddings = F.normalize(neg_image_embeddings, p=2, dim=2)  # Normalized over the last dimension

        # Positive pairs similarity
        pos_sim = torch.exp(torch.sum(pos_text_embeddings * pos_image_embeddings, dim=-1) / self.temperature)
        
        # Negative pairs similarity (text to image)
        neg_sim_text_image = torch.exp(torch.einsum('bij,bj->bi', neg_text_embeddings, pos_image_embeddings) / self.temperature)
        # Negative pairs similarity (image to text)
        neg_sim_image_text = torch.exp(torch.einsum('bi,bkj->bk', pos_text_embeddings, neg_image_embeddings) / self.temperature)

        # Calculate the loss for text-to-image
        denominator_text_image = pos_sim + neg_sim_text_image.sum(dim=1)
        loss_text_image = -torch.log(pos_sim / denominator_text_image)

        # Calculate the loss for image-to-text
        denominator_image_text = pos_sim.unsqueeze(1) + neg_sim_image_text
        loss_image_text = -torch.log(pos_sim.unsqueeze(1) / denominator_image_text).sum(dim=1)

        # Combine both losses
        loss = (loss_text_image + loss_image_text).mean()

        return loss


    def training_step(self, batch, batch_idx):
        # Forward pass
        pos_text_embeddings, pos_image_embeddings = self(batch['pos_input_ids'], batch['pos_attention_mask'], batch['pos_pixel_values'])
        neg_text_embeddings, neg_image_embeddings = self(batch['neg_input_ids'].view(-1, batch['neg_input_ids'].shape[-1]), 
                                                         batch['neg_attention_mask'].view(-1, batch['neg_attention_mask'].shape[-1]), 
                                                         batch['neg_pixel_values'].view(-1, batch['neg_pixel_values'].shape[-3], 
                                                                                        batch['neg_pixel_values'].shape[-2], batch['neg_pixel_values'].shape[-1]))
        
        # Reshape negative embeddings
        neg_text_embeddings = neg_text_embeddings.view(batch['neg_input_ids'].shape[0], self.num_negatives, -1)
        neg_image_embeddings = neg_image_embeddings.view(batch['neg_input_ids'].shape[0], self.num_negatives, -1)
        
        # Compute loss
        loss = self.compute_loss(pos_text_embeddings, pos_image_embeddings, neg_text_embeddings, neg_image_embeddings)
        
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Forward pass
        pos_text_embeddings, pos_image_embeddings = self(batch['pos_input_ids'], batch['pos_attention_mask'], batch['pos_pixel_values'])
        neg_text_embeddings, neg_image_embeddings = self(batch['neg_input_ids'].view(-1, batch['neg_input_ids'].shape[-1]), 
                                                         batch['neg_attention_mask'].view(-1, batch['neg_attention_mask'].shape[-1]), 
                                                         batch['neg_pixel_values'].view(-1, batch['neg_pixel_values'].shape[-3], 
                                                                                        batch['neg_pixel_values'].shape[-2], batch['neg_pixel_values'].shape[-1]))
        
        # Reshape negative embeddings
        neg_text_embeddings = neg_text_embeddings.view(batch['neg_input_ids'].shape[0], self.num_negatives, -1)
        neg_image_embeddings = neg_image_embeddings.view(batch['neg_input_ids'].shape[0], self.num_negatives, -1)
        
        # Compute loss
        loss = self.compute_loss(pos_text_embeddings, pos_image_embeddings, neg_text_embeddings, neg_image_embeddings)

        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        # Forward pass
        pos_text_embeddings, pos_image_embeddings = self(batch['pos_input_ids'], batch['pos_attention_mask'], batch['pos_pixel_values'])
        neg_text_embeddings, neg_image_embeddings = self(batch['neg_input_ids'].view(-1, batch['neg_input_ids'].shape[-1]), 
                                                         batch['neg_attention_mask'].view(-1, batch['neg_attention_mask'].shape[-1]), 
                                                         batch['neg_pixel_values'].view(-1, batch['neg_pixel_values'].shape[-3], 
                                                                                        batch['neg_pixel_values'].shape[-2], batch['neg_pixel_values'].shape[-1]))
        
        # Reshape negative embeddings
        neg_text_embeddings = neg_text_embeddings.view(batch['neg_input_ids'].shape[0], self.num_negatives, -1)
        neg_image_embeddings = neg_image_embeddings.view(batch['neg_input_ids'].shape[0], self.num_negatives, -1)
        
        # Compute loss
        loss = self.compute_loss(pos_text_embeddings, pos_image_embeddings, neg_text_embeddings, neg_image_embeddings)

        self.test_outputs.append(loss)
        self.log('test_loss', loss)
        return loss

    def on_test_epoch_end(self):
        if self.test_outputs:
            avg_loss = torch.stack(self.test_outputs).mean()
            self.log('avg_test_loss', avg_loss)
        self.test_outputs.clear()

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=self.eps)

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.num_training_steps)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]

class CLIPCheckpointModel(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-4):
        super().__init__()
        self.model = model  # This should be a pre-trained CLIPModel
        self.learning_rate = learning_rate
        self.clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        self.device_type = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, images, texts):
        # Process images
        image_inputs = self.clip_processor(images=images, return_tensors='pt').to(self.device_type)
        image_embeddings = self.model.get_image_features(**image_inputs)

        # Process texts
        text_embeddings = self.compute_text_embeddings(texts)

        # Normalize embeddings
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

        return image_embeddings, text_embeddings

    def compute_text_embeddings(self, texts):
        text_embeddings = []
        for text in texts:
            # Tokenize the text without truncation
            tokens = self.clip_processor.tokenizer(
                text,
                return_tensors='pt',
                truncation=False,
                add_special_tokens=False
            )['input_ids'].squeeze(0)

            # Implement sliding window
            window_size = 77
            stride = 50
            num_tokens = tokens.size(0)
            window_embeddings = []

            for i in range(0, num_tokens, stride):
                window_tokens = tokens[i:i + window_size]
                if window_tokens.size(0) == 0:
                    break

                if window_tokens.size(0) < window_size:
                    padding = torch.full(
                        (window_size - window_tokens.size(0),),
                        self.clip_processor.tokenizer.pad_token_id
                    )
                    window_tokens = torch.cat([window_tokens, padding])

                attention_mask = (window_tokens != self.clip_processor.tokenizer.pad_token_id).long()

                window_tokens = window_tokens.unsqueeze(0).to(self.device)
                attention_mask = attention_mask.unsqueeze(0).to(self.device)

                # Encode window tokens
                with torch.no_grad():
                    outputs = self.model.text_model(
                        input_ids=window_tokens,
                        attention_mask=attention_mask
                    )
                embedding = outputs.last_hidden_state[:, 0, :]
                window_embeddings.append(embedding)

            # Aggregate embeddings
            if window_embeddings:
                window_embeddings = torch.cat(window_embeddings, dim=0)
                aggregated_embedding = window_embeddings.mean(dim=0)
            else:
                aggregated_embedding = torch.zeros(self.model.config.hidden_size).to(self.device)

            text_embeddings.append(aggregated_embedding)

        text_embeddings = torch.stack(text_embeddings)
        return text_embeddings

    def compute_loss(self, image_embeddings, text_embeddings):
        """
        Compute contrastive loss for the image and text embeddings.
        """
        # Compute similarity scores
        logits_per_image = image_embeddings @ text_embeddings.t() * self.model.logit_scale.exp()
        logits_per_text = logits_per_image.t()

        # Labels
        batch_size = image_embeddings.size(0)
        labels = torch.arange(batch_size).to(image_embeddings.device)

        # Compute cross-entropy loss
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2

        return loss

    def training_step(self, batch, batch_idx):
        images = batch['images']
        texts = batch['texts']
        batch_size = len(images)  # Calculate batch size
        image_embeddings, text_embeddings = self(images, texts)
        loss = self.compute_loss(image_embeddings, text_embeddings)
        self.log('train_loss', loss, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch['images']
        texts = batch['texts']
        batch_size = len(images)  # Calculate batch size
        image_embeddings, text_embeddings = self(images, texts)
        loss = self.compute_loss(image_embeddings, text_embeddings)
        self.log('val_loss', loss, batch_size=batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        images = batch['images']
        texts = batch['texts']
        batch_size = len(images)  # Calculate batch size
        image_embeddings, text_embeddings = self(images, texts)
        loss = self.compute_loss(image_embeddings, text_embeddings)
        self.log('test_loss', loss, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def generate_embeddings(self, images=None, texts=None):
        """
        Generate embeddings for the given images or texts.

        Parameters:
        - images: List of images to generate embeddings for.
        - texts: List of texts to generate embeddings for.

        Returns:
        - Dictionary with 'image_embeddings' and 'text_embeddings' as keys, depending on the inputs.
        """
        results = {}

        # Check if images are provided
        if images is not None:
            self.model.eval()  # Set model to evaluation mode
            with torch.no_grad():
                image_inputs = self.clip_processor(images=images, return_tensors='pt').to(self.device_type)
                image_embeddings = self.model.get_image_features(**image_inputs)
                image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
                results['image_embeddings'] = image_embeddings

        # Check if texts are provided
        if texts is not None:
            self.model.eval()  # Set model to evaluation mode
            with torch.no_grad():
                text_embeddings = self.compute_text_embeddings(texts)
                text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
                results['text_embeddings'] = text_embeddings

        return results

class FineTuneCLIPClassifier(pl.LightningModule):
    def __init__(self, pretrained_model, finetune_mode, extract_representation_from, num_classes, weight_decay, eps, warmup_steps, num_training_steps, 
                learning_rate, loss_fn, temperature):
        super(FineTuneCLIPClassifier, self).__init__()
        self.text_model = pretrained_model.text_model
        self.image_model = pretrained_model.image_model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.eps = eps
        self.warmup_steps = warmup_steps
        self.num_training_steps = num_training_steps
        self.loss_fn_name = loss_fn
        self.extract_representation_from = extract_representation_from
        self.temperature = temperature
        self.finetune_mode = finetune_mode

        self.validation_outputs = []  # To store validation outputs
        self.test_outputs = []  # To store test outputs

        # Classification head
        self.classifier = nn.Linear(self.text_model.config.hidden_size, num_classes)

        # Loss function
        self.ce_loss = nn.CrossEntropyLoss()

        if self.loss_fn_name == "CE+SupCon":
            self.supcon_loss = SupConLoss(self.temperature)

    def forward(self, input_ids, attention_mask, pixel_values):
        # Get text embeddings
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)

        if self.extract_representation_from == "CLS":
            # Use CLS token embedding
            text_embeddings = text_outputs.last_hidden_state[:, 0, :]
        elif self.extract_representation_from == "EOS":
            # Get the positions of the last non-padding tokens
            sequence_lengths = attention_mask.sum(dim=1) - 1
            text_embeddings = text_outputs.last_hidden_state[torch.arange(input_ids.size(0)), sequence_lengths, :]
        else:
            raise ValueError("extract_representation_from must be 'CLS' or 'EOS'")

        # Get image embeddings
        image_outputs = self.image_model(pixel_values=pixel_values)
        image_embeddings = image_outputs.last_hidden_state[:, 0, :]  # Use CLS token embedding

        # Take the mean of text and image embeddings
        embeddings = (text_embeddings + image_embeddings) / 2

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

        elif self.finetune_mode == "all":
            # Unfreeze all layers
            for param in self.text_model.parameters():
                param.requires_grad = True
            for param in self.image_model.parameters():
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
