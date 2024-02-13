import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm.auto import tqdm
from dataset import *
from model_structure import *
from utils import *

def train_step(generator: torch.nn.Module,
          discriminator: torch.nn.Module,
          dataloader: torch.utils.data.DataLoader,
          lossfn: torch.nn.Module,
          gen_optimizer: torch.optim.Optimizer,
          disc_optimizer: torch.optim.Optimizer,
          device: torch.device):
    generator.train()
    discriminator.train()
    
    real = 1.
    fake = 0.
    gen_losses = 0
    disc_losses = 0

    for batch in tqdm(dataloader):
        # Send train image to device
        train_batch = batch[0].to(device)

        ###### Train Distcriminator ######
        # Set zero_grad
        disc_optimizer.zero_grad()
        
        # Put true image to the discriminator
        true_output = discriminator(train_batch)
        # Create answer tensor
        label = torch.full(size=true_output.shape, fill_value=real, device=device)
        # Calculate the loss of true data
        disc_loss_real = lossfn(true_output, label)
        # Calaulate gradient and backward pass
        disc_loss_real.backward()
        
        # Create noize
        dim_z = torch.randn(size=(train_batch.shape[0], 100, 1, 1), device=device)
        # Create fake image from noize
        generated = generator(dim_z)
        # Put fake image to the discriminator
        fake_output = discriminator(generated)
        # Calculate the loss of fake data
        disc_loss_fake = lossfn(fake_output, label.fill_(fake))

        # Calculate fianl loss of discriminator
        disc_loss = disc_loss_real + disc_loss_fake
        disc_losses+= disc_loss
        # Calculate gradient and backward pass
        disc_loss_fake.backward()

        # Update Discriminator
        disc_optimizer.step()

        ###### Train Generator ######
        gen_optimizer.zero_grad()
        generated = generator(dim_z)
        gen_output = discriminator(generated)
        gen_loss = lossfn(gen_output, label.fill_(real))
        gen_losses+= gen_loss
        gen_loss.backward()
        gen_optimizer.step()

    disc_losses /= len(dataloader)
    gen_losses /= len(dataloader)
    return disc_losses, gen_losses

def test_step(generator: torch.nn.Module,
              batch_size: int,
              device: torch.device):
    generator.eval()
    with torch.inference_mode():
        # Create noize
        dim_z = torch.randn(size=(batch_size, 100, 1, 1), device=device)
        generated_img = generator(dim_z)
    return generated_img

def trainer(epochs: int,
            generator: torch.nn.Module,
            discriminator: torch.nn.Module,
            train_dataloader: torch.utils.data.DataLoader,
            loss_fn: torch.nn.Module,
            gen_optimizer: torch.optim.Optimizer,
            disc_optimizer: torch.optim.Optimizer,
            batch_size: int,
            device: torch.device):
    
    results = {"disc_loss" : [],
               "gen_loss": []}
    
    writer = SummaryWriter()
    test_images = []
    for epoch in range(epochs):
        disc_loss, gen_loss = train_step(generator=generator,
                                         discriminator=discriminator,
                                         dataloader=train_dataloader,
                                         lossfn=loss_fn,
                                         gen_optimizer=gen_optimizer,
                                         disc_optimizer=disc_optimizer,
                                         device=device)

        generated_image = test_step(generator=generator,
                                    batch_size=batch_size,
                                    device=device)
        
        results["disc_loss"].append(disc_loss)
        results["gen_loss"].append(gen_loss)
        writer.add_scalars(main_tag="Loss",
                          tag_scalar_dict={"disc_loss": disc_loss,
                                           "gen_loss": gen_loss},
                          global_step=epoch)
        img_grid = torchvision.utils.make_grid(generated_image)
        test_image = img_grid.detach().cpu().numpy().transpose(1,2,0) * 255
        test_images.append(test_image.astype(np.uint8))
        writer.add_image("Test Genorator", 
                         img_grid, 
                         global_step=epoch)
        print(f"[{epoch+1} / {epochs}] ---------------- disc_loss: {disc_loss}, gen_loss: {gen_loss}")
    writer.close()
    return test_images
