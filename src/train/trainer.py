import torch
import itertools
from torch import nn, optim
from ..models.generator import Generator 
from ..models.discriminator import Discriminator 
from ..models.losses import GANLoss, cycle_consistency_loss, identity_loss

# CycleGANTrainer Class
class CycleGANTrainer:
    def __init__(self, device, dataloader, config):
        self.device= device
        self.dataloader= dataloader
        self.config= config

        # Initialize models
        self.G_A = Generator().to(device) # A -> B
        self.G_B = Generator().to(device) # B -> A
        self.D_A = Discriminator().to(device)
        self.D_B = Discriminator().to(device)

        # Loss functions
        self.criterionGAN = GANLoss(use_lsgan=True)
        self.criterionCycle = cycle_consistency_loss
        self.criterionIdentity = identity_loss

        # Optimizers
        lr = config.get('training', {}).get('lr', 0.0002)
        self.optimizer_G = optim.Adam(itertools.chain(self.G_A.parameters(), self.G_B.parameters()),
                                      lr=lr, betas=(0.5, 0.999)
                                      )
        self.optimizer_D_A = optim.Adam(self.D_A.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer_D_B = optim.Adam(self.D_B.parameters(), lr=lr, betas=(0.5, 0.999))


    # training step
    def train_step(self, real_A, real_B):
        # move to device
        real_A = real_A.to(self.device)
        real_B = real_B.to(self.device)

        # Train generator
        self.optimizer_G.zero_grad()

        # Identity loss
        idt_A= self.G_A(real_B)
        idt_B= self.G_B(real_A)
        loss_idt_A= self.criterionIdentity(idt_A, real_B)
        loss_idt_B= self.criterionIdentity(idt_B, real_A)

        # GAN loss
        fake_B = self.G_A(real_A)
        pred_fake_B = self.D_B(fake_B)
        loss_G_A = self.criterionGAN(pred_fake_B, True)

        fake_A = self.G_B(real_B)
        pred_fake_A = self.D_A(fake_A)
        loss_G_B = self.criterionGAN(pred_fake_A, True)

        # Cycle loss
        rec_A = self.G_B(fake_B)
        rec_B = self.G_A(fake_A)
        loss_cycle_A = self.criterionCycle(rec_A, real_A)
        loss_cycle_B = self.criterionCycle(rec_B, real_B)

        # Total generator loss
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
        loss_G.backward()
        self.optimizer_G.step()

        # Train Discriminator A
        self.optimizer_D_A.zero_grad()
        loss_D_A_real = self.criterionGAN(self.D_A(real_A), True)
        loss_D_A_fake = self.criterionGAN(self.D_A(fake_A.detach()), False)
        loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5
        loss_D_A.backward()
        self.optimizer_D_A.step()

        # Train Discriminator B
        self.optimizer_D_B.zero_grad()
        loss_D_B_real = self.criterionGAN(self.D_B(real_B), True)
        loss_D_B_fake = self.criterionGAN(self.D_B(fake_B.detach()), False)
        loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5
        loss_D_B.backward()
        self.optimizer_D_B.step()

        return{
            'loss_G' : loss_G.item(),
            'loss_D_A' : loss_D_A.item(),
            'loss_D_B' : loss_D_B.item()
        }