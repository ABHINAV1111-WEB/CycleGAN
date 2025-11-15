import torch
import torch.nn as nn

# Adversoal loss: Makes generated images look real to the discriminator

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True):
        """
        Args:
            use_lsgan(bool): If True, uses Least Squares GAN loss(recommende for stability).
            If False, use standard Binary Cross Entropy loss.
        
        """

        super(GANLoss, self).__init__()
        if use_lsgan:
            self.loss = nn.MSELoss()  # LSGAN: real=1.0, fake= 0.0
        
        else:
            self.loss = nn.BCEWithLogitsLoss()  # Vanilla GAN

    def get_target_tensor(self, prediction, target_is_real):
        # creates a tensor filled with real or fake labels.
        if target_is_real:
            return torch.ones_like(prediction)
        else:
            return torch.zeros_like(prediction)
            
    def forward(self, prediction, target_is_real):
        # compute GAN loss between  prediction and target label.
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)
        
# Cycle-Consistency Loss: Ensures F(G(x)) = x and G(F(y)) = y
def cycle_consistency_loss(reconstructed, original, lambda_cycle=10.0):
    """
    L1 loss between original and reconstructed image.
    Args:
        reconstructed(Tensor): Reconstructed image( e.g. F(G(x)))
        origianl (tensor): Original image (e.g., x)
        lambda_cycle (float): weight for cycle loss
    """

    return lambda_cycle * torch.nn.functional.l1_loss(reconstructed, original)

# Identity Loss: Encourages G(x) = x when x is already in target domain
def identity_loss(identity_output, real_image, lambda_identity=5.0):
    """
    L1 loss between generator and real image when input is already in target domain.
    Args:
        identity_output(tensor): G(y) when y belong to target domain
        real_image(Tensor): y
        lambda_identity (float): Weight for identity loss
    """

    return lambda_identity * torch.nn.functional.l1_loss(identity_output, real_image)