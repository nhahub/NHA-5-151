from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "raw" / "EuroSAT_RGB"
MEAN, STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)


class TransformSubset(Dataset):
    def __init__(self, base, indices, transform):
        self.base, self.indices, self.transform = base, indices, transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.base[self.indices[i]]
        return self.transform(img), label


def get_loaders(data_dir=DEFAULT_DATA, batch_size=64, seed=42, num_workers=2):
    base = datasets.ImageFolder(str(data_dir))
    targets = base.targets
    idx = list(range(len(base)))
    train_idx, rest_idx = train_test_split(idx, test_size=0.3, stratify=targets, random_state=seed)
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=0.5, stratify=[targets[i] for i in rest_idx], random_state=seed
    )

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def loader(indices, tf, shuffle):
        return DataLoader(
            TransformSubset(base, indices, tf),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return (
        loader(train_idx, train_tf, True),
        loader(val_idx, eval_tf, False),
        loader(test_idx, eval_tf, False),
        base.classes,
    )
