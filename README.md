# CILP — CLIP-style Vision–Language Alignment

A minimal, educational **CLIP-style** dual-encoder implementation using a pretrained **ResNet18 image encoder** and **DistilBERT text encoder**. Learn visual-language alignment end to end: image/text embeddings → linear projection → L2 normalization → scaled cosine similarity → symmetric InfoNCE → retrieval evaluation.

> This is **not** an exact reproduction of OpenAI CLIP. The independently pretrained encoders have not been jointly aligned before this project. No retrieval results or trained weights are included.

## Repository structure

```text
CILP/
├── clip_align/
│   ├── __init__.py       # Public model/dataset exports
│   ├── model.py          # Encoders, Projection Heads, similarity and InfoNCE
│   ├── data.py           # Paired image-caption Dataset
│   └── evaluation.py     # Bidirectional Recall@K
├── examples/
│   └── pairs.csv         # Illustrative CSV format (images not included)
├── train.py              # Freeze encoders, train projection heads
├── evaluate.py           # Evaluate saved checkpoint
├── requirements.txt
└── .gitignore
```

## Setup (Conda / CUDA)

```bash
conda create -n cilp python=3.11 -y
conda activate cilp
# Install a CUDA-enabled PyTorch build appropriate to your driver / GPU.
# For CUDA 12.8, for example:
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"
```

`transformers` does **not** have a separate `transformers-gpu` package. `torch` provides GPU support.

## Prepare the data

Create **disjoint train and validation splits** with one row per image and exactly one caption per image. Both CSV files must contain columns `image,caption`:

```csv
image,caption
dog.jpg,a dog running on grass
car.jpg,a red car on the road
```

Place the referenced images in a folder, e.g. `images/`. `examples/pairs.csv` is a **format illustration**, not a real dataset; no JPEG images are bundled.

For this minimal symmetric InfoNCE implementation, avoid multiple captions per image or duplicated image-text positives within the same training batch. Flickr8k, Flickr30k, and COCO often have multiple captions per image: construct a one-caption-per-image split or implement multi-positive loss and evaluation first.

## Train

```bash
python train.py --train-csv data/train.csv --val-csv data/val.csv --image-dir images --batch-size 16 --epochs 5 --output checkpoints/best.pt
```

By default only the two linear Projection Heads and the trainable logit-scale are optimized. Both pretrained encoders are frozen and kept in evaluation mode. Training selects the best checkpoint by the average of validation I2T and T2I Recall@1.

## Evaluate

```bash
python evaluate.py --checkpoint checkpoints/best.pt --val-csv data/val.csv --image-dir images
```

Metrics: `I2T_R@1`, `I2T_R@5`, `T2I_R@1`, `T2I_R@5`. For **N < 5** validation pairs, R@5 is trivially 100%, so use a meaningful validation set. The reference evaluator builds an `[N,N]` similarity matrix in RAM, which is only appropriate for modest datasets.

## Mathematical summary

For batch size B and embedding size D, the normalized image and text projections `I,T` have shape `[B,D]`. Their similarity logits are `exp(logit_scale) * I @ T.T`, of shape `[B,B]`. The target index for each row is the diagonal, `arange(B)`. Loss = average of image-to-text and text-to-image cross entropy.

## Project status and next experiments

- [x] Paired image/text Dataset and tokenization
- [x] Separate pretrained image/text encoders
- [x] Projection Heads, normalized embeddings, scaled similarity
- [x] Symmetric InfoNCE training
- [x] One-to-one retrieval Recall@K evaluation
- [ ] Download and validate a real dataset
- [ ] Record baseline retrieval numbers (including pretrained CLIP)
- [ ] Compare frozen vs partially unfrozen encoders, linear vs MLP heads
- [ ] Add multi-caption/multi-positive support
- [ ] Add test automation and large-scale chunked retrieval

**Validation note:** The code was organized and reviewed for tensor shapes, but has not been executed with real image data or a GPU in this upload. Numerical performance must be measured after running training.
