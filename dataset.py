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
