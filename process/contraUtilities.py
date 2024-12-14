""" 
Python version: 3.9
Description: Contains utility functions and helper classes for different layers.
"""

# %% Importing libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        else:
            raise NotImplementedError

def cl_init(cls, pooler_type_, config, temp_):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = pooler_type_
    cls.pooler = Pooler(pooler_type_)
    cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=temp_)
    cls.init_weights()

def cl_forward(cls, encoder, pooler_type, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, 
                labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
    #import ipdb; ipdb.set_trace();
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    batch_size = int(input_ids.size(0))
    
    # mlm_outputs = None
    # Flatten input for encoding
    # input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    # attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    # if token_type_ids is not None:
    #     token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)

    # Get raw embeddings
    outputs = encoder(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
                        output_attentions=output_attentions, output_hidden_states=False if pooler_type == 'cls' else True, return_dict=True)

    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view((batch_size, pooler_output.size(-1))) # (bs, num_sent, hidden)

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)

    return pooler_output

def sentemb_forward(cls, encoder, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, 
                    inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = encoder(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask,
                        inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
                        return_dict=True)

    pooler_output = cls.pooler(attention_mask, outputs)
    if cls.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(pooler_output=pooler_output, last_hidden_state=outputs.last_hidden_state, 
                                                        hidden_states=outputs.hidden_states)

def compute_sim_matrix(feats):
    """
    Takes in a batch of features of size (bs, feat_len).
    """
    sim_matrix = F.cosine_similarity(feats.unsqueeze(2).expand(-1, -1, feats.size(0)),
                                     feats.unsqueeze(2).expand(-1, -1, feats.size(0)).transpose(0, 2),
                                     dim=1)

    return sim_matrix

def compute_target_matrix(labels):
    """
    Computes a target matrix for contrastive learning based on class labels.

    This function generates a square matrix where each element (i, j) indicates whether 
    the labels for samples i and j are the same. This binary matrix serves as the target 
    for similarity in contrastive learning tasks, facilitating the training to learn 
    embeddings that are closer for similar (same label) instances and further apart for 
    dissimilar (different label) ones.

    Parameters:
        labels (torch.Tensor): A 1D tensor containing the class labels for a batch of samples.
                               The tensor should have a shape of (bs,), where bs is the batch size.

    Returns:
        torch.Tensor: A 2D square tensor of shape (bs, bs) where each element is 1.0 if the 
                      corresponding labels are the same, and 0.0 otherwise. This tensor is of
                      type float.
    """
    label_matrix = labels.unsqueeze(-1).expand((labels.shape[0], labels.shape[0]))
    trans_label_matrix = torch.transpose(label_matrix, 0, 1)
    target_matrix = (label_matrix == trans_label_matrix).type(torch.float)

    return target_matrix


def kl_contrastive_loss(pred_sim_matrix, target_matrix, temperature, labels):
    return F.kl_div(F.softmax(pred_sim_matrix / temperature).log(), F.softmax(target_matrix / temperature),
                    reduction="batchmean", log_target=False)

def supervised_infoNCE_loss(features, labels, temperature=0.1):
    """
    Computes the Supervised InfoNCE loss using class labels to determine positive and negative pairs.

    Args:
        features (torch.Tensor): Embeddings of shape (N, D) where N is batch size and D is embedding dimension.
        labels (torch.Tensor): Corresponding labels of shape (N,).
        temperature (float): A temperature scaling factor to soften the softmax output.

    Returns:
        torch.Tensor: The computed loss.
    """
    device = features.device
    n_samples = features.size(0)
    labels = labels.contiguous().view(-1, 1)

    # Normalize the features to simplify the cosine similarity calculation
    features = F.normalize(features, p=2, dim=1)

    # Compute cosine similarity matrix
    similarity_matrix = torch.mm(features, features.T) / temperature

    # Create a mask to identify positive and negative samples
    positive_mask = torch.eq(labels, labels.T).float().to(device)

    # Subtract large value from diagonal to ignore self-comparison
    logits_mask = torch.scatter(
        torch.ones_like(similarity_matrix),
        1,
        torch.arange(n_samples).view(-1, 1).to(device),
        0
    )
    masked_logits = similarity_matrix * logits_mask

    # Compute log-sum-exp across all samples (for denominator of softmax)
    exp_logits = torch.exp(masked_logits)
    log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))

    # Compute mean of log-likelihood over positive samples
    mean_log_prob_pos = (positive_mask * log_prob).sum(1) / positive_mask.sum(1)

    # Loss is negative log of mean positive log-probabilities
    loss = -mean_log_prob_pos.mean()
    return loss

def supervised_infoNCE_loss_with_negatives(features, labels, temperature=0.1, num_hard_negatives=5, epsilon=1e-6):
    device = features.device
    n_samples = features.size(0)
    labels = labels.view(-1, 1)

    # Normalize features to the unit sphere to simplify cosine similarity
    features = F.normalize(features, p=2, dim=1)

    # Compute cosine similarity matrix and scale by temperature
    similarity_matrix = torch.mm(features, features.T) / temperature

    # Mask to select positive samples
    positive_mask = torch.eq(labels, labels.T).float().to(device)

    # Mask to exclude self-similarity
    diagonal_mask = torch.eye(n_samples, device=device).bool()
    positive_mask[diagonal_mask] = 0

    # Hard negatives: select the top-k negative examples (excluding self)
    negative_mask = 1 - positive_mask
    negative_mask[diagonal_mask] = 0
    top_negatives = torch.topk(similarity_matrix * negative_mask, k=num_hard_negatives, dim=1).values

    # Compute log-sum-exp for normalization over chosen hard negatives and all positives
    positive_counts = positive_mask.sum(1).int().tolist()  # Convert to int here
    positives = similarity_matrix[positive_mask.bool()].split(positive_counts)  # Split into lists of positives per sample
    positives_padded = torch.nn.utils.rnn.pad_sequence(positives, batch_first=True, padding_value=-float('Inf'))  # Pad sequences to allow concatenation

    max_sim, _ = torch.max(torch.cat([positives_padded, top_negatives], dim=1), dim=1, keepdim=True)
    exp_sim_sum = torch.exp(similarity_matrix - max_sim).clamp(min=epsilon).sum(1, keepdim=True)

    # Compute log probability of positive samples
    log_prob_pos = torch.log(torch.exp(positives_padded - max_sim).clamp(min=epsilon).sum(1, keepdim=True) / exp_sim_sum + epsilon)

    # Negative log likelihood
    loss = -log_prob_pos.mean()
    return loss

def supervised_contrastive_loss(features, labels, temperature=0.1):
    """
    Computes the supervised contrastive loss.

    Parameters:
        features (torch.Tensor): The embeddings for the batch, shape (N, D)
                                 where N is the batch size and D is the dimension of the embeddings.
        labels (torch.Tensor): The class labels for the batch, shape (N,)
                               with each value in range [0, C-1] where C is the number of classes.
        temperature (float): A temperature scaling factor to adjust the sharpness of
                             the softmax distribution.

    Returns:
        torch.Tensor: The computed supervised contrastive loss.
    """
    device = features.device
    labels = labels.to(device)
    batch_size = features.shape[0]

    # Normalize the features to the unit sphere
    features = F.normalize(features, p=2, dim=1)

    # Compute the cosine similarity matrix
    sim_matrix = torch.mm(features, features.t()) / temperature

    # Mask for identifying positive pairs (excluding the diagonal)
    pos_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
    pos_mask.fill_diagonal_(0)

    # Compute the logits
    exp_sim = torch.exp(sim_matrix) * pos_mask

    # Sum of exps for all positive pairs
    sum_pos = exp_sim.sum(dim=1, keepdim=True)

    # Log-sum-exp trick for numerical stability
    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    log_prob_denom = torch.log(torch.exp(sim_matrix - logits_max).sum(dim=1, keepdim=True) + 1e-12) + logits_max

    # Compute the loss
    loss = -torch.log(sum_pos / torch.exp(log_prob_denom) + 1e-12)
    loss = loss.mean()

    return loss

def supervised_contrastive_loss_with_negatives(features, labels, temperature=0.5, num_hard_negatives=5, eps=1e-8):
    device = features.device
    labels = labels.to(device)
    batch_size = features.shape[0]

    # Normalize the features to the unit sphere
    features = F.normalize(features, p=2, dim=1)

    # Compute the cosine similarity matrix and apply temperature scaling
    sim_matrix = torch.mm(features, features.t())
    sim_matrix = torch.clamp(sim_matrix, min=-10, max=10) / temperature  # Clamping extreme values

    # Mask for identifying positive pairs (excluding the diagonal)
    pos_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
    pos_mask.fill_diagonal_(0)

    # Mask for identifying negatives
    neg_mask = labels.unsqueeze(1) != labels.unsqueeze(0)

    # Select hard negatives: top 'num_hard_negatives' from each row, considering only negatives
    negative_scores = sim_matrix * neg_mask.float()
    
    # Adjust num_hard_negatives if it exceeds the number of valid negative samples
    k = min(num_hard_negatives, neg_mask.sum(dim=1).min().item())
    top_negatives, _ = torch.topk(negative_scores, k=k, dim=1)

    # Log-sum-exp trick for numerical stability: max_sim for each row
    max_sim, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    exp_sim_matrix = torch.exp(sim_matrix - max_sim)

    # Sum of exps for positive and hard negative pairs
    exp_pos = (exp_sim_matrix * pos_mask.float()).sum(dim=1, keepdim=True)
    exp_hard_neg = torch.exp(top_negatives - max_sim)

    # Combine positives and hard negatives in the denominator
    denom = exp_pos + exp_hard_neg.sum(dim=1, keepdim=True) + torch.exp(negative_scores - max_sim).sum(dim=1, keepdim=True) - exp_hard_neg

    # Log probability of positive pairs
    log_prob_pos = torch.log(exp_pos / (denom + eps) + eps)

    # Compute the mean of negative log probabilities across the batch
    loss = -log_prob_pos.mean()
    return loss
    
def create_triplets(embeddings, labels, num_triplets_per_sample):
    triplets = []
    for i in range(len(labels)):
        anchor = embeddings[i]
        pos_indices = torch.where(labels == labels[i])[0]
        neg_indices = torch.where(labels != labels[i])[0]
        
        # Ensure there are enough positive samples
        if len(pos_indices) <= 1:
            continue
        
        # Ensure there are enough negative samples
        if len(neg_indices) == 0:
            continue
        
        # Select a single positive sample
        pos_index = pos_indices[pos_indices != i][torch.randint(len(pos_indices)-1, (1,)).item()]
        positive = embeddings[pos_index]

        # Create multiple triplets with different negative samples
        for _ in range(num_triplets_per_sample):
            neg_index = neg_indices[torch.randint(len(neg_indices), (1,)).item()]
            negative = embeddings[neg_index]

            triplets.append((anchor, positive, negative))
    
    if len(triplets) == 0:
        return None
    
    anchor, positive, negative = zip(*triplets)
    return torch.stack(anchor), torch.stack(positive), torch.stack(negative)

def compute_triplet_loss(embeddings, labels, num_triplets_per_sample, margin=1.0):
    triplets = create_triplets(embeddings, labels, num_triplets_per_sample)
    if triplets is None:
        return torch.tensor(0.0, requires_grad=True), None
    
    anchor, positive, negative = triplets

    pos_dist = 1 - F.cosine_similarity(anchor, positive)
    neg_dist = 1 - F.cosine_similarity(anchor, negative)
    triplet_loss = F.relu(pos_dist - neg_dist + margin)

    return triplet_loss.mean(), triplet_loss