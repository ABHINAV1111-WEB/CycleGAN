import torch
import yaml
import os
from src.train.trainer import CycleGANTrainer
from src.data.dataloader import get_dataloader

# Get the path to config.yaml (one level up from src/)
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")

# Load config from YAML
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Set device
device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")

# Load dataloader
dataloader = get_dataloader(config, train=True)

# Initialize trainer
trainer = CycleGANTrainer(device=device, dataloader=dataloader, config=config)

# Training loop
for epoch in range(config["training"]["epochs"]):
    print(f"\nEpoch [{epoch + 1}/{config['training']['epochs']}]")
    for i, batch in enumerate(dataloader):
        real_A, real_B = batch["A"].to(device), batch["B"].to(device)
        losses = trainer.train_step(real_A, real_B)
        print(f"Step [{i+1}] | "
              f"Generator Loss: {losses['loss_G']:.4f} | "
              f"D_A Loss: {losses['loss_D_A']:.4f} | "
              f"D_B Loss: {losses['loss_D_B']:.4f}")