""" 
Python version: 3.9
Description: Contains helper classes and functions to load a Declutr-small model with supervised contrastive loss into the LightningModule.
"""

import sys
from sklearn.metrics import f1_score, balanced_accuracy_score

import torch
from torch import nn
import lightning.pytorch as pl

from transformers import RobertaModel
from transformers import AutoModel
from transformers import AdamW, get_linear_schedule_with_warmup

# from deepspeed.ops.adam import DeepSpeedCPUAdam

# Custom library
sys.path.append('../process/')
from contraUtilities import compute_sim_matrix, compute_target_matrix, contrastive_loss

class DeclutrClassifier(nn.Module):
    def __init__(self, model, classifier):
        super().__init__()
        self.model = model
        self.fc = classifier

    def forward(self, x, return_feat=False):
        # x is a tokenized input
        # feature = self.model(input_ids=x[0], token_type_ids=x[1], attention_mask=x[2])
        feature = self.model(input_ids=x[0], attention_mask=x[2])
        # out = self.fc(feature.pooler_output.flatten(1))       # not good for our task     # (BS, E)
        hidden_states = feature["hidden_states"]
        out = self.fc(feature.last_hidden_state.flatten(1))  # (BS, T, E)
        if return_feat:
            # print(feature.shape)
            return out, hidden_states, feature.last_hidden_state.flatten(1)
        return out

class LogisticRegression(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout=0):
        super().__init__()
        # print(f'Logistic Regression classifier of dim ({in_dim} {hid_dim} {out_dim})')

        self.nn = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_dim, hid_dim, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hid_dim, out_dim, bias=True),
        )

    def forward(self, x, return_feat=False):
        out = self.nn(x)
        if return_feat:
            return out, x
        return out

class HTContraClassifierModel(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        self.save_hyperparameters()
        if isinstance(args, tuple) and len(args) > 0: 
            self.args = args[0]
            self.hparams.learning_rate = self.args.learning_rate
            self.hparams.eps = self.args.adam_epsilon
            self.hparams.weight_decay = self.args.weight_decay
            self.hparams.model_name_or_path = self.args.model_name_or_path
            self.hparams.num_classes = self.args.num_classes
            self.hparams.num_training_steps = self.args.num_training_steps
            self.hparams.warmup_steps = self.args.warmup_steps
            self.hparams.emb_len = self.args.emb_len
            self.hparams.max_seq_length = self.args.max_seq_length
            self.hparams.hidden_dim = self.args.hidden_dim
            self.hparams.dropout = self.args.dropout
            self.hparams.nb_epochs = self.args.nb_epochs
            self.hparams.temp = self.args.temp
            self.hparams.coefficient = 1
        
        # freeze
        self._frozen = False
        self.criterion = nn.CrossEntropyLoss()
        
        # Loading model
        self.model = AutoModel.from_pretrained(self.hparams.model_name_or_path)
        self.model = DeclutrClassifier(self.model, LogisticRegression(self.hparams.emb_len * self.hparams.max_seq_length, 
                                                                      self.hparams.hidden_dim, self.hparams.num_classes, 
                                                                      dropout=self.hparams.dropout))
        # self.model = nn.DataParallel(self.model).cuda()

    def forward(self, batch):
        # The batch contains the input_ids, the input_put_mask and the labels (for training)
        input_ids, token_ids, attention_mask, y = batch
        # x, y = (input_ids.cuda(), token_ids.cuda(), attention_mask.cuda()), y.cuda()
        x, y = (input_ids, token_ids, attention_mask), y
        pred, _, feats = self.model(x, return_feat=True)
        
        # classification loss
        loss_1 = self.criterion(pred, y.long())
        
        # generate the mask
        # mask = y.clone().cpu().apply_(lambda x: x not in []).type(torch.bool).cuda()
        mask = y.clone().cpu().apply_(lambda x: x not in []).type(torch.bool)
        feats, pred, y = feats[mask], pred[mask], y[mask]

        # contrastive learning
        sim_matrix = compute_sim_matrix(feats)
        target_matrix = compute_target_matrix(y)
        loss_2 = contrastive_loss(sim_matrix, target_matrix, self.hparams.temp, y)

        # total loss
        loss = loss_1 + self.hparams.coefficient * loss_2
        
        return pred, feats, y, loss

    def training_step(self, batch, batch_nb):
        # the training step is a (virtual) method,specified in the interface, that the pl.LightningModule
        # class stipulates you to overwrite. This we do here, by virtue of this definition
        _, _, _, train_loss = self(batch)  # self refers to the model, which in turn acceses the forward method
        self.log_dict({"train_loss": train_loss}, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return train_loss
        # the training_step method expects a dictionary, which should at least contain the loss

    def validation_step(self, batch, batch_nb):
        # the training step is a (virtual) method,specified in the interface, that the pl.LightningModule
        # class  wants you to overwrite, in case you want to do validation. This we do here, by virtue of this definition.

        pred, _, y, val_loss = self(batch)
        # self refers to the model, which in turn accesses the forward method

        # Evaluating the performance
        predictions = torch.argmax(pred, dim=1)
        balanced_accuracy = balanced_accuracy_score(y.detach().cpu().numpy(), predictions.detach().cpu().numpy(), adjusted=True)
        macro_accuracy = f1_score(y.detach().cpu().numpy(), predictions.detach().cpu().numpy(), average='macro')
        micro_accuracy = f1_score(y.detach().cpu().numpy(), predictions.detach().cpu().numpy(), average='micro')
        weighted_accuracy = f1_score(y.detach().cpu().numpy(), predictions.detach().cpu().numpy(), average='weighted')        
        
        self.log_dict({"val_loss": val_loss, 'accuracy': balanced_accuracy, 'macro-F1': macro_accuracy, 'micro-F1': micro_accuracy, 'weighted-F1':weighted_accuracy}, 
                       on_step=False, on_epoch=True, prog_bar=True, logger=True)

        
        return val_loss
    
    def test_step(self, batch, batch_nb):
        # the training step is a (virtual) method,specified in the interface, that the pl.LightningModule
        # class  wants you to overwrite, in case you want to do test. This we do here, by virtue of this definition.

        pred, _, y, test_loss = self(batch)
        # self refers to the model, which in turn accesses the forward method

        # Evaluating the performance
        predictions = torch.argmax(pred, dim=1)
        balanced_accuracy = balanced_accuracy_score(y.detach().cpu().numpy(), predictions.detach().cpu().numpy(), adjusted=True)
        macro_accuracy = f1_score(y.detach().cpu().numpy(), predictions.detach().cpu().numpy(), average='macro')
        micro_accuracy = f1_score(y.detach().cpu().numpy(), predictions.detach().cpu().numpy(), average='micro')
        weighted_accuracy = f1_score(y.detach().cpu().numpy(), predictions.detach().cpu().numpy(), average='weighted')
        
        self.log_dict({"test_loss": test_loss, 'accuracy': balanced_accuracy, 'macro-F1': macro_accuracy, 'micro-F1': micro_accuracy, 'weighted-F1':weighted_accuracy}, 
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return test_loss
    
    def predict_step(self, batch, batch_nb):
        # the training step is a (virtual) method,specified in the interface, that the pl.LightningModule
        # class  wants you to overwrite, in case you want to do validation. This we do here, by virtue of this definition.
        return None

    def configure_optimizers(self):
        # The configure_optimizers is a (virtual) method, specified in the interface, that the
        # pl.LightningModule class wants you to overwrite.

        # In this case we define that some parameters are optimized in a different way than others. In
        # particular we single out parameters that have 'bias', 'LayerNorm.weight' in their names. For those
        # we do not use an optimization technique called weight decay.

        no_decay = ['bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [{'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay':self.hparams.weight_decay}, 
                                        {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.eps)
        # optimizer = DeepSpeedCPUAdam(optimizer_grouped_parameters, adamw_mode=True, lr=self.hparams.learning_rate, betas=(0.9, 0.999), eps=self.hparams.eps)

        # We also use a scheduler that is supplied by transformers.
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.hparams.num_training_steps)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.nb_epochs)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]

    def freeze(self) -> None:
        # freeze all layers, except the final classifier layers
        for name, param in self.model.named_parameters():
            if 'classifier' not in name:  # classifier layer
                param.requires_grad = False

        self._frozen = True

    def unfreeze(self) -> None:
        if self._frozen:
            for name, param in self.model.named_parameters():
                if 'classifier' not in name:  # classifier layer
                    param.requires_grad = True

        self._frozen = False

    def train_epoch_start(self):
        """pytorch lightning hook"""
        if self.current_epoch < self.hparams.nr_frozen_epochs:
            self.freeze()

        if self.current_epoch >= self.hparams.nr_frozen_epochs:
            self.unfreeze() 



