# BERT-Conditioned Text-to-Image Synthesis Using Conditional GANs

A PyTorch implementation of text-to-image synthesis using a BERT-based text encoder and a conditional Generative Adversarial Network (cGAN). The project explores how natural-language descriptions can condition a generator to produce corresponding images.

## Project Overview

This project implements a text-conditioned image-generation pipeline. A pretrained BERT model converts captions into semantic representations, which are projected into a lower-dimensional embedding space and supplied to the GAN.

The generator combines text embeddings with random noise to synthesize images. The discriminator evaluates images jointly with their corresponding text representations to distinguish real image-caption pairs from generated or mismatched pairs.

The implementation also includes optional attention, conditioning augmentation, and CLIP-based guidance.

## Architecture

### 1. Text Encoder
- Uses `bert-base-uncased` from Hugging Face Transformers.
- Extracts pooled sentence representations from BERT.
- Projects the 768-dimensional BERT representation into a 256-dimensional conditioning vector.
- Supports frozen BERT weights by default, with a trainable projection layer.
- Optionally exposes contextual token features for word-level attention.

### 2. Generator
- Combines text embeddings with a 100-dimensional random noise vector.
- Uses a fully connected projection followed by transposed convolutional upsampling.
- Generates 64 × 64 RGB images.
- Uses Batch Normalization and ReLU in intermediate layers, with Tanh at the output.
- Supports optional conditioning augmentation and spatial word-level attention.

### 3. Discriminator
- Extracts visual features through convolutional layers.
- Combines image features with the projected text embedding.
- Learns to distinguish real matching image-caption pairs from generated and, optionally, mismatched pairs.
- Produces logits for adversarial training using `BCEWithLogitsLoss`.

### 4. Optional CLIP Guidance
- Uses a pretrained CLIP ViT-B/32 model as a frozen semantic guidance network.
- Computes image-text contrastive alignment loss during training when enabled.
- Keeps CLIP parameters frozen while allowing gradients to flow through the image encoder into the generator.

### 5. Evaluation
- Calculates cosine similarity between generated-image and caption embeddings using CLIP.
- Records generator and discriminator losses.
- Supports training-loss plots and evaluation reports.

## Dataset

The project uses the Hugging Face `text-to-image-2M` dataset, stored in WebDataset TAR shards.

Dataset: https://huggingface.co/datasets/jackyhate/text-to-image-2M

The experiment uses a selected subset rather than downloading the complete dataset:

| Split | Number of examples |
|---|---:|
| Training | 10,000 |
| Testing | 2,000 |
| Total | 12,000 |

Images are decoded, converted to RGB, resized to 64 × 64, converted to tensors, and normalized to the range [-1, 1].

The data pipeline supports streaming shard access, deterministic buffered sampling, local caching of the selected subset, sample identity tracking, and checks for duplicate images across the training and test partitions.

**Note:** The subset is selected from a shuffled stream across the configured shards; it should not be interpreted as a uniformly random sample of the entire two-million-example dataset. The full dataset is not included in this repository.

## Training

The training pipeline includes:

- Separate generator and discriminator optimization.
- Two generator updates per discriminator update by default.
- Adam optimizers with configurable learning rates and beta values.
- One-sided label smoothing for real discriminator examples.
- Optional mismatched-caption negatives.
- Optional image noise during discriminator training.
- Optional conditioning-augmentation regularization.
- Optional CLIP alignment loss.
- Gradient clipping and finite-value checks.
- Learning-rate scheduling.
- Periodic checkpoints and generated sample grids.
- Checkpoint resumption with saved model, optimizer, scheduler, and random-number-generator states.

## Repository Structure

```text
Text_to_Image_Synthesis_Using_cGAN_and_BERT/
│
├── README.md
├── config.yaml
├── pyproject.toml
├── requirements.txt
├── train.py
├── generate.py
├── evaluate.py
│
├── src/
│   └── text_to_image_gan/
│       ├── __init__.py
│       ├── alignment.py
│       ├── data.py
│       ├── evaluation.py
│       ├── inference.py
│       ├── training.py
│       ├── utils.py
│       │
│       └── models/
│           ├── __init__.py
│           ├── text_encoder.py
│           ├── generator.py
│           └── discriminator.py
│
└── tests/
    ├── test_models.py
    └── test_training_finite_guards.py
```

## Installation

Clone the repository:

```bash
git clone https://github.com/jahnavireddy-27/Text_to_Image_Synthesis_Using_cGAN_and_BERT.git
cd Text_to_Image_Synthesis_Using_cGAN_and_BERT
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -e ".[dev]"
```

The first run may download pretrained BERT and CLIP model weights from Hugging Face.

## Configuration

Model architecture, dataset settings, training hyperparameters, evaluation settings, and output paths are configured in `config.yaml`.

Important configurable parameters include:

- Image resolution and dataset subset sizes.
- BERT model and text embedding dimensions.
- Generator and discriminator channel configurations.
- Noise-vector dimensions.
- Generator and discriminator learning rates.
- Batch size and number of epochs.
- Conditioning augmentation and word-level attention.
- Optional CLIP alignment weight.
- Checkpoint and sample-output locations.

## Running the Project

### Train

```bash
python train.py --config config.yaml
```

To explicitly select a device:

```bash
python train.py --config config.yaml --device mps
```

Other supported device choices depend on the installed PyTorch environment, including `cpu` and `cuda`.

### Generate Images

After training and saving a checkpoint:

```bash
python generate.py \
  --checkpoint outputs/checkpoints/checkpoint_epoch_0020.pt \
  --captions "A red bird sitting on a tree branch" \
              "A small dog running through a green field" \
  --output-dir outputs/generated \
  --images-per-caption 2
```

### Evaluate

```bash
python evaluate.py --help
```

Use the available evaluation arguments to load a checkpoint and evaluate generated image-caption pairs.

### Run Tests

```bash
pytest
```

## Reproducibility and Limitations

- Random seeds and dataset-selection metadata are recorded to support reproducibility.
- Checkpoints preserve model and optimizer states for resuming training.
- Exact stochastic resumption may vary across hardware backends.
- CLIP alignment and attention mechanisms are configurable rather than mandatory.
- CLIP cosine similarity measures semantic image-text alignment; it is not a direct measure of visual realism.
- The repository provides the implementation and training pipeline. It does not include pretrained project checkpoints, downloaded datasets, or generated output images.
- Image-generation quality depends on training configuration, compute resources, and optimization stability. The implementation should not be interpreted as a guarantee of photorealistic output.

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- BERT
- Conditional Generative Adversarial Networks
- Convolutional Neural Networks
- CLIP
- NumPy
- Pillow
- Matplotlib
- PyYAML

## Acknowledgements

The dataset is provided by its contributors through Hugging Face. This project uses pretrained BERT and CLIP models from the Hugging Face ecosystem.

