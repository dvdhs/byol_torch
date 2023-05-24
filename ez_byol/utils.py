import torch
import torch.nn.functional as F
import torchvision.transforms as T

def set_grad(model, bool):
    for param in model.parameters():
        param.requires_grad = bool

# See SIMCLR paper for details on augmentations
# https://arxiv.org/pdf/2002.05709.pdf (Appendix A)
def get_simclr_augments(image_size):
    return torch.nn.Sequential(
    T.RandomApply([
        T.ColorJitter(0.8, 0.8, 0.8, 0.2)],
        p = 0.3
    ),
    T.RandomGrayscale(p=0.2),
    T.RandomHorizontalFlip(),
    T.RandomApply([
        T.GaussianBlur((3, 3), (1.0, 2.0))],
        p = 0.2
    ),
    T.RandomResizedCrop((image_size, image_size)),
    T.Normalize(
        mean=torch.tensor([0.485, 0.456, 0.406]),
        std=torch.tensor([0.229, 0.224, 0.225])),
  )

# See p.4 of BYOL paper
def BYOLLoss(x1, x2):
  return 2 - 2 * F.cosine_similarity(x1,x2)