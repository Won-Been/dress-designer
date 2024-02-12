import torch
from torchvision import transforms
from pathlib import Path

import dataset
import model_structure
import utils
import trainer
from config import args


device = "cuda" if torch.cuda.is_available() else "cpu"

def main(mode: str):
    if mode == "train":
        generator = model_structure.Generator(z=args.z_dimension, 
                                              img_size=args.img_size, 
                                              img_channel=args.img_channel)
        discriminator = model_structure.Discriminator(img_size=args.img_size, 
                                                      img_channel=args.img_channel)
        
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        train_dataloader, test_dataloader = dataset.DressDataset(data_path=args.data_path, 
                                                                 transform=transform, 
                                                                 train_range=args.train_range, 
                                                                 batch_size=args.batch_size, 
                                                                 num_worker=args.num_worker)
        loss_fn = torch.nn.BCELoss()
        gen_optimizer = torch.optim.Adam(params=generator.parameters(), 
                                         lr=args.lr,
                                         betas=(args.momentum, 0.999))
        disc_optimizer = torch.optim.Adam(params=discriminator.parameters(), 
                                         lr=args.lr,
                                         betas=(args.momentum, 0.999))
        
        trainer.trainer(epochs=args.epochs, 
                        generator=generator, 
                        discriminator=discriminator, 
                        train_dataloader=train_dataloader,
                        loss_fn=loss_fn,
                        gen_optimizer=gen_optimizer,
                        disc_optimizer=disc_optimizer,
                        batch_size=args.batch_size,
                        device=device)
        
        utils.save_model(generator=generator,
                         target_dir=args.target_dir,
                         model_name=args.model_name)
    else:
        generator_structure = model_structure.Generator(z=args.z_dimension, 
                                              img_size=args.img_size, 
                                              img_channel=args.img_channel)
        model_path = Path(args.save_path) / args.model_name
        generator = generator_structure.load_state_dict(torch.load(model_path))
        trainer.test_step(generator=generator,
                          batch_size=args.batch_size,
                          device=device)