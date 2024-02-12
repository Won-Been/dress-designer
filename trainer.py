import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from dataset import *
from model_structure import *
from utils import save_model

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

    for batch in dataloader:
        # Send train image to device
        train_batch = batch[0]
        train_batch.to(device)

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
        writer.add_scalar(main_tag="Loss",
                          tag_scalar_dict={"disc_loss": disc_loss,
                                           "gen_loss": gen_loss},
                          global_step=epoch)
        writer.add_image("Test Genorator", 
                         generated_image, 
                         global_step=epoch)
    writer.close()
    return results


if __name__=="__main__":
    gen = Generator(z=100, img_size=128, img_channel=3)
    disc = Discriminator(img_size=128, img_channel=3)
    data_dir = Path("dress_dataset")
    # data_list = glob.glob(f"{data_dir}/*.png")
    # print(len(data_list))
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    dataloader, _ = DressDataset(data_path=data_dir, transform=transform, train_range=0.7, batch_size=1, num_worker=0)
    lossfn = torch.nn.BCELoss()
    gen_optimizer = torch.optim.Adam(params=gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
    disc_optimizer = torch.optim.Adam(params=gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
    test_image = test_step(gen, batch_size=1, device="cpu")
    print(test_image)
    