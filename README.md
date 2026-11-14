# Text-to-Image Synthesis Using GANs and BERT-Based Text Encoding

This repository is a modular reconstruction from the project specification. The
original source code is unavailable, so choices not fixed by the report are
identified as reconstruction defaults rather than claims about the original
implementation.

## What is report-confirmed

- Python, PyTorch, Hugging Face Transformers, pretrained `bert-base-uncased`,
  Torchvision, and CLIP ViT-B/32 for evaluation.
- Tokenized captions with padding/truncation, maximum length 32, BERT pooler
  output, and a 768-to-256 projection.
- A generator conditioned on the 256-dimensional text vector and 100-dimensional
  noise, projected to 512 x 4 x 4 and upsampled to 64 x 64 RGB with BatchNorm,
  ReLU, and Tanh.
- A discriminator with four image convolution blocks, image channels
  3 -> 64 -> 128 -> 256 -> 512, a 4 x 4 final image feature map, expanded
  256-dimensional text features, and a joint image/text head.
- BCE training, Adam, generator/discriminator learning rates 1.5e-4/5e-5,
  betas (0.5, 0.999), batch size 32, real label 0.9, image noise sigma 0.05,
  gradient clipping at 1.0, two generator updates per discriminator update,
  StepLR halving every 10 epochs, and checkpoints every 5 epochs.
- The experiment sizes listed in the report: 10,000 training examples and 2,000
  test examples. These are selected subsets; the whole 2M-example dataset is not
  treated as the training set. `drop_last` is disabled so the final partial
  training batch is included.
- 64 x 64 images normalized to [-1, 1], image-caption pairs, generated samples,
  loss plots, and CLIP image/text cosine similarity. The historical score 0.2556
  is stored only as a reference, never as an expected test threshold.

The active experiment now uses Oxford Flowers 102 to keep the image domain
focused. This is a deliberate dataset change from the original report's broad
2M-source sample, so its reported 10,000/2,000 split sizes do not describe the
current flower run.

The flower experiment now adds two explicit text-to-image improvements beyond
the report reconstruction: token-level BERT conditioning and a frozen CLIP
image-text alignment loss. BERT sentence features still provide the global
condition; BERT token features guide generator locations through attention at
4x4, 8x8, 16x16, and 32x32 feature resolutions. During generator updates, a
frozen CLIP model supplies a symmetric in-batch image-text contrastive loss.
CLIP's parameters are not optimized; gradients pass through its image encoder
to the generated pixels. These additions are a new experiment and are not yet
evidence that image quality has improved.

## Reconstruction defaults and assumptions

The report does not determine every implementation detail. This project makes
the following choices explicit and configurable in `config.yaml`:

- BERT weights are frozen by default (`freeze_text_encoder: true`); the
  sentence and token projections remain trainable. Set the flag to `false` to
  fine-tune BERT.
- Generator channels are 512 -> 256 -> 128 -> 64 -> 3 over four
  stride-2, 4 x 4 transposed convolutions. Hidden activation is ReLU and output
  activation is Tanh. Generator output BatchNorm is disabled by default.
- Discriminator image blocks use stride-2, 4 x 4 convolutions. Image-block
  BatchNorm is enabled except in the first block. The joint head is
  768 x 4 x 4 -> 512 x 4 x 4 -> 1 x 1 x 1; intermediate joint BatchNorm is
  enabled, and final-output BatchNorm is disabled. The model returns logits;
  training uses `BCEWithLogitsLoss`. `DeeperDiscriminator.probabilities`
  converts logits to probabilities when needed.
- Seed 42, 100 epochs, four map-style data workers, and per-epoch sample grids
  are reproducibility/usability defaults, not reported historical settings.
- Generator targets use 1.0 separately from the discriminator's smoothed real
  target 0.9. Discriminator-input noise is disabled during generator updates by
  default and can be enabled using `training.noise_on_generator_updates`.
- `image` and `text` are starter column names only. Dataset ID, splits, columns,
  and streaming mode must be checked/set for the selected Hugging Face dataset.
  No dataset schema is presumed.

The dimensional path in the default configuration is:

```text
Text: BERT pooler (B x 768) -> sentence projection (B x 256)
Token path: BERT hidden states (B x L x 768) -> projection (B x L x 256)
Generator input: text (B x 256) + noise (B x 100) -> FC (B x 8192)
  -> 512 x 4 x 4 -> 256 x 8 x 8 -> 128 x 16 x 16
  -> 64 x 32 x 32 -> 3 x 64 x 64
Word attention: token features are fused at 4x4, 8x8, 16x16, and 32x32.
Discriminator image: 3 x 64 x 64 -> 64 x 32 x 32 -> 128 x 16 x 16
  -> 256 x 8 x 8 -> 512 x 4 x 4
Joint head: image 512 x 4 x 4 + text 256 x 4 x 4 -> 768 x 4 x 4
  -> 512 x 4 x 4 -> 1 x 1 x 1 raw logit
```

## Setup

Use Python 3.10 or newer in a virtual environment. Install the project and test
dependencies with:

```bash
python -m pip install -r requirements.txt
```

The active dataset is `kaiyuyue/oxford-flowers-102-captions`, loaded from
Hugging Face Datasets. Its `train` split contains 7,169 image-caption records
and its `val` split contains 1,020. The images are Oxford Flowers 102; each has
20 machine-written captions generated with Qwen2.5-VL. The captions range from
short phrases to detailed descriptions. The configuration samples from the
detailed half on each access so the generator sees varied descriptions over
training. The captions are CC0; the images retain the original Oxford dataset
terms. The image and caption fields are `image` and `captions`, with `image_id`
as the sample identifier. [Dataset card](https://huggingface.co/datasets/kaiyuyue/oxford-flowers-102-captions)

The current configuration uses all 7,169 training images and the 1,020-image
validation split as the held-out evaluation set. Hugging Face Datasets caches
its downloaded files under `data.cache_dir` (`outputs/flowers/dataset_cache`).
The train and validation splits are distinct in the dataset source, with one
identical image explicitly allowed by `data.allow_identical_images`. Images
are resized directly to square 64 x 64 and normalized to [-1, 1].

The active configuration uses only the flower dataset. Generated training
artifacts and downloaded dataset caches are kept out of the source bundle; the
Colab notebook includes the current code for a fresh GPU run.

## Run

Train with the flower-only configuration:

```bash
python train.py --config config.yaml
```

Training saves the complete YAML configuration in the checkpoint directory and
alongside each checkpoint. Checkpoints include all three model states, both
optimizer and scheduler states, epoch/loss history, and Python/NumPy/PyTorch
random-number-generator states. Resume by setting `training.resume_from` to a
checkpoint path in `config.yaml`. Resume rejects configuration differences by
default; only an increased epoch target and changed `paths.*` destinations are
allowed. Each allowed difference is printed. A checkpoint is saved at the final
completed epoch even if that epoch is not an interval multiple.

Generate one image for a caption:

```bash
python generate.py \
  --checkpoint checkpoints/checkpoint_epoch_0005.pt \
  --caption "A small bird perched on a branch"
```

Evaluate generated images against the held-out test captions with CLIP:

```bash
python evaluate.py --checkpoint checkpoints/checkpoint_epoch_0005.pt
```

Training writes periodic image grids under
`outputs/flowers/word_attention_clip/samples/` and a loss plot to
`outputs/flowers/word_attention_clip/training_losses.png`. Evaluation writes the measured average
generated-image/text cosine similarity to `outputs/clip_evaluation.json`.
CLIP's 0.2556 historical report value is included in that JSON as a reference
only; it is not used to pass/fail evaluation. Evaluation noise is seeded by
`evaluation.seed`, and the JSON records the CLIP model/processor identifiers,
effective image preprocessing settings, tokenizer settings, test split, and
similarity metric.

With `training.clip_alignment_weight` above zero, each generator update also
downloads/uses the configured frozen CLIP teacher and logs its contrastive loss
as `CLIP` in the epoch output. The first run therefore downloads both BERT and
CLIP weights. Checkpoints from the earlier sentence-only generator cannot be
resumed into the word-attention architecture; this experiment starts a fresh
checkpoint series.

## Tests

Run the CPU-friendly synthetic forward-pass and configuration tests with:

```bash
python -m pytest
```

The text-encoder test injects a mock tokenizer and BERT module, verifies frozen
BERT weights and the trainable projection, and does not download pretrained
weights. The model tests use synthetic tensors. No training or dataset download
is started by the tests.
