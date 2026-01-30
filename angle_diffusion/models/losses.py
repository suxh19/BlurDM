import torch
import torch.nn as nn
import numpy as np

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


class MSELoss(nn.Module):
    """Mean Squared Error Loss (L2)"""

    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, x, y):
        return torch.mean((x - y) ** 2)



class PSNRLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
          super(GANLoss, self).__init__()
          self.real_label = target_real_label
          self.fake_label = target_fake_label
          self.gan_mode = gan_mode
    
    def get_zero_tensor(self, input):
        return torch.zeros_like(input).requires_grad_(False)

    def forward(self, input, target_is_real, for_discriminator=False):
        if self.gan_mode == "hinge":
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -input.mean()
        else:
            raise ValueError("'Unexpected gan_mode {}'.format(gan_mode)")
        
        return loss


def wgan_gp_loss(D, real_data, fake_data, batch_size, device):
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    alpha = alpha.expand_as(real_data)

    interpolated_data = alpha * real_data + (1 - alpha) * fake_data

    interpolated_data = interpolated_data.requires_grad_(True)

    interpolated_output = D(interpolated_data)

    gradients = torch.autograd.grad(
        outputs=interpolated_output,
        inputs=interpolated_data,
        grad_outputs=torch.ones_like(interpolated_output).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)

    gradients_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradients_penalty






