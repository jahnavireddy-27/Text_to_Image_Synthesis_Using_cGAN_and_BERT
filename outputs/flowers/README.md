# Flower-only GAN run

The active experiment uses Oxford Flowers 102 image-caption pairs from
`kaiyuyue/oxford-flowers-102-captions`: 7,169 training images and 1,020
validation images. Each image has 20 Qwen2.5-VL-generated captions. The loader
randomly chooses among the more detailed half of the captions during training.
The caption text is CC0; the underlying images retain the original Oxford
Flowers dataset terms.

The config keeps BERT as the text encoder, adds token-level attention at four
generator resolutions, disables discriminator BatchNorm, and enables
matching-aware training and conditioning augmentation. A frozen CLIP teacher
adds an in-batch image-text contrastive loss to generator updates. This is a
fresh architecture; sentence-only checkpoints from the earlier run are not
compatible. The run saves grids every epoch and checkpoints every five epochs.

## Run on Google Colab

Open `Flower_GAN_Colab.ipynb` in Colab, select a GPU runtime, and run the cells
in order. The notebook contains the project source bundle, so no source ZIP or
Google Drive connection is needed. Training outputs and dataset cache stay in
temporary Colab storage. The final notebook cell can download a results ZIP to
your computer if you want to keep checkpoints for a later session.

For a later session, open the same notebook and set its restore option to use
the results ZIP you downloaded. The dataset cache may download again if Colab
discards the runtime.

Look for object shape and caption-specific color/petal changes in the fixed
sample grids. The first run is a 20-epoch check; losses alone do not establish
that generation has succeeded.
