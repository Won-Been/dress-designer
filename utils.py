import torch
from torch import nn
from torchvision.utils import make_grid
from pathlib import Path
from PIL import Image
import imageio

def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)

def save_model(generator: torch.nn.Module,
               target_dir: str,
               model_name: str):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' of '.pth'"
    model_save_dir = target_dir / model_name

    torch.save(obj=generator.state_dict(),
               f=model_save_dir)


def create_gif(image_batches,
               save_dir: str, 
               save_name: str):

    save_dir = Path(save_dir)
    assert save_name.endswith(".gif"), "file name should be end with '.gif'"
    imageio.mimsave(f"{save_dir / save_name}", image_batches)