import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid
from torchvision import transforms, datasets
from pathlib import Path
from typing import Tuple


def DressDataset(data_path: str, 
                 transform: torchvision.transforms,
                 train_range: float,
                 batch_size: int,
                 num_worker: int,
                 ) -> Tuple:
    """Creates training and testing DataLoaders.
    
    Args:
        data_path: Path to data directory.
        transform: torchvision.transforms to transform the data.
        train_ragne: Ratio of train data.
        batch_size: Number of sample per batch in each of the DataLoaders.
        num_worker: Number of workers per DataLoader.
    
    Returns:
        A tuple of (train_dataloader, test_dataloader)
    """
    dataset = datasets.ImageFolder(data_path, transform=transform)
    train = int(len(dataset) * train_range)
    train_dataset = Subset(dataset, range(train))
    test_dataset = Subset(dataset, range(train, len(dataset)))

    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_worker
                                  )
    
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_worker
                                 )
    return train_dataloader, test_dataloader

if __name__=="__main__":
    data_dir = Path("dress_dataset")
    # data_list = glob.glob(f"{data_dir}/*.png")
    # print(len(data_list))
    transform = transforms.Compose([
        transforms.Resize((640, 480)),
        transforms.ToTensor()
    ])
    a, b = DressDataset(data_path=data_dir,
                        transform=transform,
                        train_range=0.7,
                        batch_size=16,
                        num_worker=0)
    print(a, b)
    print(next(iter(a))[0].shape)
    sampel = next(iter(a))[0]
    # print(make_grid(sampel))

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(make_grid(sampel).permute(1,2,0))