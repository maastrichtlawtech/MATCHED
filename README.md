# MATCHED: Multimodal Authorship-Attribution To Combat Human Trafficking in Escort-Advertisement Data

**Abstract:** Human trafficking (HT) remains a critical issue, with traffickers exploiting digital platforms to advertise victims anonymously. Current detection methods, including Authorship Attribution (AA), focus on textual data but overlook the multimodal nature of online ads, which often pair text with images. This research introduces MATCHED, a multimodal dataset comprising 27,619 unique text descriptions and 55,115 unique images from seven U.S. cities across four geographical regions. Our study extensively benchmarks text-only, vision-only, and multimodal baselines for vendor identification and verification tasks, employing multitask training objectives that achieve superior performance on in-distribution and out-of-distribution datasets. This dual-objective approach enables law enforcement agencies (LEAs) to identify known vendors while linking emerging ones. Integrating multimodal features further enhances performance, capturing complementary patterns across text and images. While text remains the dominant modality, visual data adds stylistic cues that enrich model performance. Moreover, text-image alignment strategies like CLIP and BLIP2 struggle due to low semantic overlap and ineffective use of stylistic cues, with end-to-end multimodal training proving more robust. Our findings emphasize the potential of multimodal AA to combat HT, providing LEAs with robust tools to link ads and dismantle trafficking networks.

# IDTraffickers Dataset (will be made accessible through the Dataverse Portal)
The MATCHED dataset is a novel, multimodal collection of escort advertisements designed for Authorship Attribution (AA) tasks. It comprises 27,619 unique text descriptions and 55,115 associated images, sourced from the Backpage platform across seven U.S. cities and categorized into four geographical regions. The dataset spans December 2015 to April 2016, offering a rich blend of textual and visual data for vendor identification and verification tasks. To protect individual identities, all personally identifiable information has been meticulously removed using advanced masking techniques. While the raw dataset remains securely stored, only metadata is available on Dataverse. Interested parties can access the complete dataset under strict ethical guidelines by signing a Non-Disclosure Agreement (NDA) and Data Transfer Agreement with us.

<p align="center">
  <img src="/Images/Dataset.png" alt="Dataset" style="width:50%; max-width:500px;">
</p>

After the request to access is granted, download the datasets and keep the text dataset in a folder "data/processed/" and image dataset in a folder "data/IMAGES/" wrt to your working directory.

# Setup
This repository is tested on Python 3.10 and [conda](https://docs.conda.io/projects/miniconda/en/latest/). First, you should install a virtual environment:
```
conda create -n MATCHED python=3.10
```

To activate the conda environment, run:
```
conda activate MATCHED
```

Then, you can install all dependencies:
```
pip install -r requirements.txt
```

Additionally, to perform the authorship verification task, please install the FAISS package as suggested [here](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)

# Experiments

### Classification task

Our research explores a range of baselines to establish benchmarks for text-only, vision-only, and multimodal approaches in authorship identification. Among these, we identify the DeCLUTR-small backbone as the most effective for text-only tasks, the ViT-base-patch16-244 backbone as the best-performing vision-only model, and their combination with mean pooling as the optimal multimodal backbone. The comprehensive results for all our authorship identification baselines are presented below:

<p align="center">
  <img src="/Images/identification.png" alt="Classification" style="width:65%; max-width:700px;">
</p>

#### Text Baselines
- Specify the GPU to use. CUDA_VISIBLE_DEVICES=0 means only GPU 0 will be used.
- `batch_size`: Set the batch size for training. Larger values use more memory but may speed up training.
- `geography`: Specify the geographical subset of the dataset on which to train. This could be "south," "midwest," "west," or "northeast." 
- `model_name_or_path`: Define the pretrained model for classification. The implementation is tested for "johngiorgi/declutr-small" and "AnnaWegmann/Style-Embedding" models.
- `tokenizer_name_or_path`: Specify the tokenizer to use. It should match the model to ensure compatibility.
- `seed`: Set the random seed for the reproducibility of the results.
- `logged_entry_name`: Provide a log entry name to help identify the experiment configuration. Please set up a [weights and biases](https://wandb.ai/site/) account first. 
- `learning_rate`: Specify the learning rate for the optimizer. 
- `save_dir`: Directory for models to be saved.
- `temp`: Set the temperature value for contrastive loss computation.
- `loss1_type`: Define the type of the first loss function. This could be CE or None for metric learning tasks. Here, "CE" stands for Cross-Entropy loss.
- `loss2_type`: Define the type of the second loss function. This could be SupCon, Triplet, infoNCE, SupCon-negatives or infoNCE-negatives.  "SupCon-negatives" refers to Supervised Contrastive loss with in-batch negative examples.
- `num_hard_negatives`: Number of in-batch hard negatives taken from other classes in the batch
- `task`: can be classification or metric-learning
- `save_dir`: Directory for models to be saved
- `nb_epochs`: The number of epochs for training
- `data_dir`: Directory of the text dataset
  
To train the text-only benchmark with DeCLUTR-small backbone and CE loss, run:

```python
CUDA_VISIBLE_DEVICES=0 python train/text/textClassifier.py \
    --batch_size 32 \
    --geography south \
    --model_name_or_path johngiorgi/declutr-small \
    --tokenizer_name_or_path johngiorgi/declutr-small \
    --seed 1111 \
    --logged_entry_name declutr-text-only-seed:1111-bs:32-loss:CE-south \
    --learning_rate 0.0001 \
    --save_dir models/text-baseline/unimodal/ \
    --nb_epochs 40 \
    --data_dir /data/processed/
```

To train the text-only benchmark with DeCLUTR-small backbone and joint loss, run:

```python
CUDA_VISIBLE_DEVICES=0 python train/text/textContraLearn.py \
    --batch_size 32 \
    --geography south \
    --loss1_type CE \
    --loss2_type SupCon-negatives \
    --model_name_or_path johngiorgi/declutr-small \
    --tokenizer_name_or_path johngiorgi/declutr-small \
    --seed 1111 \
    --logged_entry_name declutr-text-only-seed:1111-bs:32-loss:CE-SupCon-south-temp:0.1 \
    --learning_rate 0.0001 \
    --temp 0.1 \
    --num_hard_negatives 5 \
    --task classification \
    --nb_triplets 5 \
    --save_dir models/text-baseline/joint-loss/ \
    --nb_epochs 40 \
    --data_dir /data/processed/
```

#### Vision Baselines
- Specify the GPU to use. CUDA_VISIBLE_DEVICES=0 means only GPU 0 will be used.
- `batch_size`: Set the batch size for training. Larger values use more memory but may speed up training.
- `geography`: Specify the geographical subset of the dataset on which to train. This could be "south," "midwest," "west," or "northeast." 
- `model_name_or_path`: Define the pretrained model for classification. The implementation is tested for 'vgg16', 'vgg19', "resnet50", "resnet101", "resnet152", "mobilenet", "mobilenetv2", "densenet121", "densenet169", "efficientnet-b0", "efficientnet-b1", "efficientnet-b2",                         "efficientnet-b3", "efficientnet-b4", "efficientnet-b5", "efficientnet-b6", "efficientnet-b7", "efficientnetv2_rw_m", "efficientnetv2_rw_s", "efficientnetv2_rw_t", "convnext_tiny", "convnext_small", "convnext_base", "convnext_large", "convnext_xlarge",                         "vit_base_patch16_224", "vit_large_patch16_224", "vit_base_patch32_224", "vit_large_patch32_224", "inception_v3", "inception_resnet_v2" models.
- `seed`: Set the random seed for the reproducibility of the results.
- `logged_entry_name`: Provide a log entry name to help identify the experiment configuration. Please set up a [weights and biases](https://wandb.ai/site/) account first. 
- `learning_rate`: Specify the learning rate for the optimizer. 
- `save_dir`: Directory for models to be saved.
- `temp`: Set the temperature value for contrastive loss computation.
- `loss2_type`: Can be SupCon or triplet, the first loss is set to CE. This is activated only during the classification task.
- `num_hard_negatives`: Number of in-batch hard negatives taken from other classes in the batch
- `task`: can be classification or metric-learning
- `augment_data`: Perform augmentations, including HorizontalFlip, VerticalFlip, RandomResizedCrop, Rotate, ShiftScaleRotate, and GaussNoise the image data.
- `nb_augmented_samples`: if augment_data=True; Number of augmented samples, if set to 1–all the above operations will be applied once
- `data_type`: can be faces for the dataset with human faces, nofaces for images without faces, or all
- `save_dir`: Directory for models to be saved
- `nb_epochs`: The number of epochs for training
- `image_dir`: Directory of the image dataset
- `data_dir`: Directory of the text dataset
- `self_sup_loss`: Can be SupCon or Triplet. Only activated during the metric-learning task.

To train the vision-only benchmark with ViT-base-patch16-224 backbone and CE loss, run:

```python
CUDA_VISIBLE_DEVICES=0 python train/image/imageClassifiers.py \
    --batch_size 32 \
    --geography south \
    --data_type all \
    --augment_data False \
    --model_name_or_path vit_base_patch16_224 \
    --logged_entry_name vit-vision-only-south-noaugment-loss:CE-seed:1111 \
    --seed 1111 \
    --learning_rate 0.0001 \
    --save_dir models/image-baseline/unimodal/ \
    --nb_epochs 40 \
    --image_dir /data/IMAGES/ \
    --data_dir /data/processed/
```

To train the vision-only benchmark with ViT-base-patch16-224 backbone and joint loss, run:

```python
CUDA_VISIBLE_DEVICES=0 python train/image/imageContraClassifier.py \
    --batch_size 32 \
    --augment_data False \
    --city south \
    --data_type all \
    --model_name_or_path vit_base_patch16_224 \
    --seed 1111 \
    --logged_entry_name vit-vision-only-south-noaugment-loss:CE-SupCon-seed:1111 \
    --learning_rate 0.0001 \
    --temp 0.1 \
    --nb_epochs 40 \
    --task classification \
    --loss2_type SupCon \
    --save_dir models/image-baseline/joint-loss/  \
    --image_dir /data/IMAGES/ \
    --data_dir /data/processed/
```

#### Multimodal Baselines
- Specify the GPU to use. CUDA_VISIBLE_DEVICES=0 means only GPU 0 will be used.
- `batch_size`: Set the batch size for training. Larger values use more memory but may speed up training.
- `geography`: Specify the geographical subset of the dataset on which to train. This could be "south," "midwest," "west," or "northeast." 
- `seed`: Set the random seed for the reproducibility of the results.
- `logged_entry_name`: Provide a log entry name to help identify the experiment configuration. Please set up a [weights and biases](https://wandb.ai/site/) account first. 
- `learning_rate`: Specify the learning rate for the optimizer. 
- `save_dir`: Directory for models to be saved.
- `temp`: Set the temperature value for contrastive loss computation.
- `loss`: Can be "CE", "CE+SupCon", "CE+SupCon+ITM", "SupCon", "SupCon+ITM", "ITM", "NTXent", "CE+NTXent", "CE+NTXent+ITM"
- `nb_negatives`: Number of hard negatives taken from regions outside the training sample
- `save_dir`: Directory for models to be saved
- `nb_epochs`: The number of epochs for training
- `pairing_mode`: assoicated-The negatives are text-image pairs associated but come from a different region. Or non-associated-The negatives are text-image pairs that are not associated with each other, and they also come from a different region
- `image_dir`: Directory of the image dataset
- `data_dir`: Directory of the text dataset
- `model_type`: Type of pre-training strategy to be carried, can be between CLIP, CLIPITM, and BLIP2
- `finetune_mode`: Currently, the script only accepts the parameter value as "all," indicating all model layers will be fine-tuned. We plan to extend this script to adapt the selective fine-tuning of layers.
- `pretrained_model_dir`: Directory of pre-trained text-alignment model
- `extract_representation_from`: Token to extract representations from: "CLS" or "EOS" for the ITC loss
- `augment_data`: Perform augmentations, including HorizontalFlip, VerticalFlip, RandomResizedCrop, Rotate, ShiftScaleRotate, and GaussNoise the image data.
- `nb_augmented_samples`: if augment_data=True; Number of augmented samples, if set to 1–all the above operations will be applied once
- `fusion_technique`: Fusion technique for merging the representations from DeCLUTR and ViT backbones. Can be amongst mean, concat, add, multiply, attention, qformer, or learned_fusion

To perform pre-training on the text-image alignment task using the DeCLUTR-ViT backbone, run:

```python
CUDA_VISIBLE_DEVICES=0 python train/multimodal/CLIPStyleModels.py \
    --nb_epochs 40 \
    --nb_negatives 5 \
    --temp 0.1 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --logged_entry_name blip2_pretraining:non_associated:5-seed:1111-temp:0.1 \
    --pairing_mode non-associated \
    --model_type BLIP2 \
    --save_dir models/multimodal-baseline/pre-trained/  \
    --image_dir /data/IMAGES/ \
    --data_dir /data/processed/
 ```

To fine-tune the pre-trained text-alignment model with a joint loss (by default the model uses mean pooling as the fusion technique, any other fusion technique is not yet implemented), run:

```python
CUDA_VISIBLE_DEVICES=0 python train/multimodal/finetuneCLIPStyleModels.py \
    --nb_epochs 40 \
    --temp 0.1 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --model_type BLIP2 \
    --loss CE+SupCon \
    --finetune_mode all \
    --logged_entry_name finetunedBLIP2-temp:0.1-loss:CE+SupCon-layers:all-neg:5 \
    --save_dir models/multimodal-baseline/fine-tuned/  \
    --image_dir /data/IMAGES/ \
    --data_dir /data/processed/ \
    --extract_representation_from EOS \
    --pretrained_model_dir models/multimodal-baseline/pre-trained/
 ```

To train the end-to-end DeCLUTR-ViT backbone with mean pooling-based fusion and joint loss, run:

```python
CUDA_VISIBLE_DEVICES=0 python train/multimodal/e2e_fusion.py \
    --batch_size 32 \
    --augment_data False \
    --city south \
    --seed 100 \
    --logged_entry_name DeCLUTR-ViT:CE+SupCon_fusion:mean_city:south_seed:100_temp:0.1 \
    --learning_rate 0.0001 \
    --nb_epochs 40 \
    --fusion_technique mean \
    --loss CE+SupCon \
    --temp 0.1 \
    --save_dir models/multimodal-baseline/e2e/  \
    --image_dir /data/IMAGES/ \
    --data_dir /data/processed/
```

### Metric-learning Task
Our research also explores a range of metric-learning baselines to establish text-only and vision-only benchmarks for authorship verification tasks. These baselines are trained using Triplet or SupCon losses. 

#### Text Baselines
For the DeCLUTR-small backbone to be trained on the metric-learning task with SupCon loss and in-batch negatives, please run the textContraLearn.py script above with the following changes:

```python
CUDA_VISIBLE_DEVICES=0 python train/text/textContraLearn.py \
    --batch_size 32 \
    --geography south \
    --loss1_type None \
    --loss2_type SupCon-negatives \
    --model_name_or_path johngiorgi/declutr-small \
    --tokenizer_name_or_path johngiorgi/declutr-small \
    --seed 1111 \
    --logged_entry_name declutr-text-only-seed:1111-bs:32-loss:SupCon-south-temp:0.1 \
    --learning_rate 0.0001 \
    --temp 0.1 \
    --num_hard_negatives 5 \
    --task metric-learning \
    --nb_triplets 5 \
    --save_dir models/text-baseline/metric-learning/ \
    --nb_epochs 40 \
    --data_dir /data/processed/
```

#### Vision Baselines
For the ViT-base-patch16-244 backbone to be trained on the metric-learning task with SupCon loss and in-batch negatives, please run the imageContraClassifier.py script above with the following changes:

```python
CUDA_VISIBLE_DEVICES=0 python train/image/imageContraClassifier.py \
    --batch_size 32 \
    --augment_data False \
    --city south \
    --data_type all \
    --model_name_or_path vit_base_patch16_224 \
    --seed 1111 \
    --logged_entry_name vit-vision-only-south-noaugment-loss:SupCon-seed:1111 \
    --learning_rate 0.0001 \
    --temp 0.1 \
    --nb_epochs 40 \
    --task metric-learning \
    --self_sup_loss SupCon \
    --save_dir models/image-baseline/metric-learning/  \
    --image_dir /data/IMAGES/ \
    --data_dir /data/processed/
```
