# CycleGAN - Unpaired Image-to-Image Translation

## Overview

This is an implementation of **CycleGAN**, a deep learning model for unpaired image-to-image translation. CycleGAN can learn to transform images from one domain to another without paired training data, using cycle consistency loss to maintain image content while changing style.

### Key Features
- 🎨 **Unpaired Learning**: No need for paired training images
- 🔄 **Cycle Consistency**: Ensures content preservation (A→B→A and B→A→B)
- ⚡ **Fast Training**: Efficient implementation with PyTorch
- 🎯 **Flexible Architecture**: Supports ResNet-based generators and PatchGAN discriminators
- 📊 **Easy Configuration**: YAML-based configuration system

## Project Structure

```
CYCLEGAN/
├── config.yaml              # Configuration file
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore rules
├── README.md               # This file
├── datasets/
│   └── horse2zebra/        # Sample dataset (Horse2Zebra)
│       ├── trainA/         # Domain A training images
│       └── trainB/         # Domain B training images
├── src/
│   ├── main.py             # Main training script
│   ├── config.py           # Configuration loader
│   ├── models/
│   │   ├── generator.py    # ResNet Generator
│   │   ├── discriminator.py # PatchGAN Discriminator
│   │   ├── blocks.py       # ResNet blocks
│   │   └── losses.py       # Loss functions (GAN, Cycle, Identity)
│   ├── data/
│   │   ├── dataloader.py   # Data loading utilities
│   │   └── __init__.py
│   ├── train/
│   │   ├── trainer.py      # Training loop
│   │   └── __init__.py
│   └── __init__.py
└── assets/
    ├── checkpoints/        # Saved model weights
    ├── samples/            # Generated samples
    └── logs/               # Training logs
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (optional, for GPU acceleration)

### Setup

1. **Clone or download the project:**
```bash
cd CYCLEGAN
```

2. **Create a virtual environment (optional but recommended):**
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Quick Start

### Training

To train the CycleGAN model:

```bash
python -m src.main
```

Or from the src directory:
```bash
cd src
python main.py
```

### Configuration

Edit `config.yaml` to customize training:

```yaml
training:
  device: "cuda"        # or "cpu"
  epochs: 200           # Number of training epochs
  batch_size: 1         # Batch size
  lr: 0.0002           # Learning rate
  
data:
  image_size: 256      # Input image size
  
paths:
  trainA: "datasets/horse2zebra/trainA"
  trainB: "datasets/horse2zebra/trainB"
```

## Model Architecture

### Generator (ResNet-based)
- Input: RGB image (3 channels)
- Downsampling: 2 convolutional layers with stride 2
- Residual Blocks: 9 ResNet blocks for feature transformation
- Upsampling: 2 transpose convolutional layers
- Output: RGB image (3 channels)
- Activation: Tanh

### Discriminator (PatchGAN)
- Input: RGB image (3 channels)
- Architecture: Multi-layer convolutional network
- Output: 1-channel prediction map (per-patch classification)
- Activation: Leaky ReLU

## Loss Functions

### 1. **Adversarial Loss (LSGAN)**
Encourages generators to produce realistic images:
```
L_GAN = E[(D(G(x)) - 1)²]
```

### 2. **Cycle Consistency Loss**
Ensures content preservation during translation:
```
L_cycle = ||G_B(G_A(x)) - x|| + ||G_A(G_B(y)) - y||
```

### 3. **Identity Loss**
Prevents color shifting when images are already in target domain:
```
L_identity = ||G_A(y) - y|| + ||G_B(x) - x||
```

### 4. **Total Loss**
```
L_total = L_GAN + λ_cycle * L_cycle + λ_identity * L_identity
```

## Training

The training loop performs:

1. **Generator Training**:
   - Generate fake images (A→B and B→A)
   - Compute adversarial loss
   - Compute cycle consistency loss
   - Compute identity loss
   - Backpropagation and optimization

2. **Discriminator Training**:
   - Discriminate real vs fake images
   - Update discriminator weights

## Dependencies

Key packages:
- **torch** (>=2.0.0) - Deep learning framework
- **torchvision** (>=0.15.0) - Computer vision utilities
- **PyYAML** (>=6.0) - Configuration management
- **Pillow** (>=9.0.0) - Image processing
- **numpy** (>=1.24.0) - Numerical computations
- **matplotlib** (>=3.7.0) - Visualization
- **tensorboard** (>=2.12.0) - Training monitoring

See `requirements.txt` for complete list.

## Dataset

### Preparing Your Dataset

1. Create directory structure:
```
datasets/
└── your_dataset/
    ├── trainA/    # Domain A images
    ├── trainB/    # Domain B images
    ├── valA/      # Domain A validation (optional)
    └── valB/      # Domain B validation (optional)
```

2. Update `config.yaml` with dataset paths:
```yaml
paths:
  trainA: "datasets/your_dataset/trainA"
  trainB: "datasets/your_dataset/trainB"
```

### Popular Datasets
- **Horse2Zebra**: Convert horses to zebras and vice versa
- **Photo2Painting**: Photos to artistic paintings
- **Summer2Winter**: Seasonal style transfer

## Results

Training outputs:
- **Checkpoints**: Saved model weights
- **Samples**: Generated images at intervals
- **Logs**: Training metrics and losses

## Configuration Options

### Model Parameters
```yaml
model:
  generator:
    type: "resnet"
    in_channels: 3
    out_channels: 3
    ngf: 64           # Number of generator filters
    n_blocks: 9       # Number of residual blocks
    norm: "instance"  # Normalization type
    use_reflection_pad: true
    upsample: "deconv"
  
  discriminator:
    type: "patchgan"
    in_channels: 3
    ndf: 64           # Number of discriminator filters
    norm: "instance"
```

### Training Parameters
```yaml
training:
  lr: 0.0002
  beta1: 0.5        # Adam optimizer beta1
  beta2: 0.999      # Adam optimizer beta2
  lr_decay_start: 100
  use_image_pool: true
  pool_size: 50
```

## Troubleshooting

### GPU Out of Memory
- Reduce `batch_size` in `config.yaml`
- Reduce `image_size`
- Use gradient accumulation

### Poor Results
- Increase training `epochs`
- Adjust loss weights (`lambda_cyc`, `lambda_id`)
- Ensure dataset images are aligned
- Check image quality and diversity

### Import Errors
Ensure you're running from the project root:
```bash
python -m src.main
```

## Performance Tips

1. **GPU Training**: Set `device: "cuda"` for faster training
2. **Data Loading**: Adjust `num_workers` based on CPU cores
3. **Batch Size**: Larger batches may improve stability
4. **Learning Rate**: Start with 0.0002, adjust if training is unstable

## References

- **Original Paper**: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
- **Authors**: Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros
- **Official Implementation**: [junyanz/CycleGAN](https://github.com/junyanz/CycleGAN)

## License

This project is provided for educational purposes.

## Contributing

Feel free to submit issues and enhancement requests!

## Support

For questions or issues, please check:
1. Configuration in `config.yaml`
2. Dataset paths and image format
3. CUDA/GPU availability
4. Installed dependencies

---

**Last Updated**: November 2025  
**Python Version**: 3.8+  
**Framework**: PyTorch 2.0+
