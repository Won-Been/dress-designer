import argparse

parser = argparse.ArgumentParser(description="Hyperparameters for DCGAN")
parser.add_argument("--mode", default="train", type=str, choices=["train", "test"], help="Put 'train' or 'test'")
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--lr", default=2e-4, type=float)
parser.add_argument("--momentum", default=0.5, type=float)

parser.add_argument("--z_dimension", default=100, type=int)
parser.add_argument("--img_size", default=128, type= int)
parser.add_argument("--img_channel", default=3, type=int)

parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--num_worker", default=2, type=int)
parser.add_argument("--train_range", default=0.7, type=float)

parser.add_argument("--data_path", default="/dress_dataset", type=str)
parser.add_argument("--save_path", default="/model", type=str)
parser.add_argument("--model_name", default="generator.pth", type=str)

args = parser.parse_args()