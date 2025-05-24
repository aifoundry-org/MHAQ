import os
import urllib
import zipfile
import torch

from torchvision.datasets import CocoDetection
from torchvision.datasets import wrap_dataset_for_transforms_v2
from torchvision.transforms import v2
from lightning import pytorch as pl
from torchvision import transforms, tv_tensors
from typing import Tuple, Dict

from torch.utils.data import DataLoader


class BBoxNormalizationTransform(torch.nn.Module):
    def forward(self, img, label):
        if not "boxes" in label:
            label.update(
                {
                    "boxes": tv_tensors.BoundingBoxes(
                        [],
                        device=img.device,
                        format="XYXY",
                        canvas_size=(tuple(img.size()[1::])),
                    )
                }
            )
            return img, label

        label["boxes"][:, [0, 2]] /= label["boxes"].canvas_size[0]
        label["boxes"][:, [1, 3]] /= label["boxes"].canvas_size[1]

        return img, label

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
        self.cat2idx = []

        self.transform_train = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize((self.img_size, self.img_size)),
                v2.ConvertBoundingBoxFormat("CXCYWH"),
                BBoxNormalizationTransform(),
            ]
        )

        self.transform_test = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize((self.img_size, self.img_size)),
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
            transforms=self.transform_train,
        )
        self.train_dataset = wrap_dataset_for_transforms_v2(
            self.train_dataset,
            target_keys=["boxes", "labels", "image_id", "category_id", "bbox"],
        )

        self.val_dataset = CocoDetection(
            root=os.path.join(self.data_dir, "val2017"),
            annFile=os.path.join(
                self.data_dir, "annotations", "instances_val2017.json"
            ),
            transforms=self.transform_test,
        )
        self.val_dataset = wrap_dataset_for_transforms_v2(
            self.val_dataset, target_keys=["boxes", "labels"]
        )

        self.cat2idx = self.val_dataset.coco.getCatIds()
    
    def collate_fn(self, batch):
        images, targets = zip(*batch)
        images = torch.stack(images, dim=0)           # assumes equal H×W

        out = []
        for idx, t in enumerate(targets):
            if len(t.get("labels", [])) == 0:         # nothing annotated => skip
                continue

            labels = torch.tensor([self.cat2idx.index(l) for l in t["labels"]],
                                dtype=torch.long)

            out.append({
                "boxes":  t["boxes"],
                "labels": labels,
                "idx":    torch.full((labels.size(0),), idx, dtype=torch.long)
            })

        return images, out

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
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
