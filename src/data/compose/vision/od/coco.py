import os
import urllib
import zipfile
import torch

from torchvision.datasets import CocoDetection
from lightning import pytorch as pl
from torchvision import transforms

from torch.utils.data import DataLoader


class COCODataModule(pl.LightningDataModule):
    COCO_URLS = {
        "train2017": "http://images.cocodataset.org/zips/train2017.zip",
        "val2017": "http://images.cocodataset.org/zips/val2017.zip",
        "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    }

    def __init__(
        self, data_dir: str = "./data", batch_size=1000, num_workers=5
    ) -> None:
        super().__init__()
        self.year = None
        self.data_dir = os.path.join(data_dir, "COCO")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = False
        self.img_size = 640

        self.transform_train = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
            ]
        )

        self.transform_test = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
            ]
        )

    def prepare_data(self):
        os.makedirs(self.data_dir, exist_ok=True)

        for key, url in self.COCO_URLS.items():
            file_path = os.path.join(self.data_dir, f"{key}.zip")
            if not os.path.exists(file_path):
                print(f"Downloading COCO {key}...")
                urllib.request.urlretrieve(url, file_path)

            extract_path = os.path.join(self.data_dir, key)
            if not os.path.exists(extract_path):
                print(f"Extracting COCO {key}...")
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    zip_ref.extractall(self.data_dir)

    def setup(self, stage=None):
        self.train_dataset = CocoDetection(
            root=os.path.join(self.data_dir, "train2017"),
            annFile=os.path.join(
                self.data_dir, "annotations", "instances_train2017.json"
            ),
            transform=self.transform_train,
        )

        self.val_dataset = CocoDetection(
            root=os.path.join(self.data_dir, "val2017"),
            annFile=os.path.join(
                self.data_dir, "annotations", "instances_val2017.json"
            ),
            transform=self.transform_test,
        )

    def collate_fn(self, batch):
        images, targets = zip(*batch)
        images = list(images)
        targets = [{k: torch.tensor(v) for k, v in t.items()} for t in targets]

        return images, targets

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
