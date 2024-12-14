# MATCHED: Multimodal Authorship-Attribution To Combat Human Trafficking in Escort-Advertisement Data

**Abstract:** Human trafficking (HT) remains a critical issue, with traffickers exploiting digital platforms to advertise victims anonymously. Current detection methods, including Authorship Attribution (AA), focus on textual data but overlook the multimodal nature of online ads, which often pair text with images. This research introduces MATCHED, a multimodal dataset comprising 27,619 unique text descriptions and 55,115 unique images from seven U.S. cities across four geographical regions. Our study extensively benchmarks text-only, vision-only, and multimodal baselines for vendor identification and verification tasks, employing multitask training objectives that achieve superior performance on in-distribution and out-of-distribution datasets. This dual-objective approach enables law enforcement agencies (LEAs) to identify known vendors while linking emerging ones. Integrating multimodal features further enhances performance, capturing complementary patterns across text and images. While text remains the dominant modality, visual data adds stylistic cues that enrich model performance. Moreover, text-image alignment strategies like CLIP and BLIP2 struggle due to low semantic overlap and ineffective use of stylistic cues, with end-to-end multimodal training proving more robust. Our findings emphasize the potential of multimodal AA to combat HT, providing LEAs with robust tools to link ads and dismantle trafficking networks.

# IDTraffickers Dataset (will be made accessible through the Dataverse Portal)
The MATCHED dataset is a novel, multimodal collection of escort advertisements designed for Authorship Attribution (AA) tasks. It comprises 27,619 unique text descriptions and 55,115 associated images, sourced from the Backpage platform across seven U.S. cities and categorized into four geographical regions. The dataset spans December 2015 to April 2016, offering a rich blend of textual and visual data for vendor identification and verification tasks. To protect individual identities, all personally identifiable information has been meticulously removed using advanced masking techniques. While the raw dataset remains securely stored, only metadata is available on Dataverse. Interested parties can access the complete dataset under strict ethical guidelines by signing a Non-Disclosure Agreement (NDA) and Data Transfer Agreement with us.

<p align="center">
  <img src="/Images/Dataset.png" alt="Dataset" style="width:50%; max-width:500px;">
</p>

After the request to access is granted, download the datasets and keep them in a folder "data/processed/" wrt to your working directory.

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

Our research explores a range of baselines to establish benchmarks for text-only, vision-only, and multimodal approaches in authorship identification. Among these, we identify the DeCLUTR-small backbone as the most effective for text-only tasks, the ViT-base-patch16-244 backbone as the best-performing vision-only model, and their combination with mean pooling as the optimal multimodal backbone. The comprehensive results for all our authorship identification baselines are presented below:

<p align="center">
  <img src="/Images/identification.png" alt="Classification" style="width:65%; max-width:700px;">
</p>

To train the text-only benchmark with DeCLUTR-small backbone, run:

- Specify the GPU to use. `CUDA_VISIBLE_DEVICES=0`¸ means only GPU 0 will be used.
- `batch_size`: Set the batch size for training. Larger values use more memory but may speed up training.
- `geography`: Specify the geographical subset of the dataset to train on. This could be "south", "midwest", "west", or "northeast". 
- `model_name_or_path`: Define the pretrained model for classification. The implementation is tested for "johngiorgi/declutr-small" and "AnnaWegmann/Style-Embedding" models.
- `tokenizer_name_or_path`: Specify the tokenizer to use. It should match the model to ensure compatibility.
- `seed`: Set the random seed for the reproducibility of the results.
- `logged_entry_name`: Provide a log entry name, helping identify the experiment configuration. Please set up a [weights and biases](https://wandb.ai/site/) account first. 
- `learning_rate`: Specify the learning rate for the optimizer. 
- `save_dir`: Directory for models to be saved.

```python
CUDA_VISIBLE_DEVICES=0 python textClassifier.py \
    --batch_size 32 \
    --geography south \
    --model_name_or_path johngiorgi/declutr-small \
    --tokenizer_name_or_path johngiorgi/declutr-small \
    --seed 1111 \
    --logged_entry_name declutr-text-only-seed:1111-bs:32-loss:CE-south \
    --learning_rate 0.0001 \
    --save_dir models/text-baseline/
```
