import argparse

parser = argparse.ArgumentParser(description="Hyperparameters for DCGAN")
parser.add_argument("--mode", default="train", type=str, choices=["train", "generate"], help="Put 'train' or 'test'")
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--gen_lr", default=2e-4, type=float)
parser.add_argument("--disc_lr", default=2e-4, type=float)
parser.add_argument("--momentum", default=0.5, type=float)

parser.add_argument("--z_dimension", default=100, type=int)
parser.add_argument("--img_size", default=128, type= int)
parser.add_argument("--img_channel", default=3, type=int)

parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--num_worker", default=2, type=int)
parser.add_argument("--train_range", default=1.0, type=float)

parser.add_argument("--data_path", default="/dress_dataset", type=str, help="Directory to train datasets")
parser.add_argument("--save_path", default="model", type=str, help="Directory to save the checkpoint file")
parser.add_argument("--model_name", default="generator.pth", type=str, help="The checkpoint file name with extension, ex: model.pth")

parser.add_argument("--model_path", type=str, help="Path to the checkpoint to generate the dress image")
parser.add_argument("--generated_img_path", type=str, help="The image file name to save the generated image")
args = parser.parse_args()