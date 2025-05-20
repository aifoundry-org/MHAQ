import torch
from typing import Tuple, List, Dict
from torch import Tensor
from lightning import pytorch as pl

from torchvision.datasets import VOCDetection
from torchvision import transforms

from torch.utils.data import DataLoader

VOC_CLASSES = (
    "aeroplane","bicycle","bird","boat","bottle","bus",
    "car","cat","chair","cow","diningtable","dog","horse",
    "motorbike","person","pottedplant","sheep","sofa",
    "train","tvmonitor"
)
CLASS_TO_IDX = {c:i for i,c in enumerate(VOC_CLASSES)}

def collate_fn(batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
    """
    Stacks images into (B,C,H,W).
    Concatenates targets into a (M,6) tensor of [img_idx, cls, x, y, w, h].
    """
    images = torch.stack([b[0] for b in batch], dim=0)
    all_targets = []
    for img_idx, (_, boxes) in enumerate(batch):
        if boxes.numel() == 0:
            continue
        idx_col = torch.full((boxes.size(0), 1), img_idx, dtype=boxes.dtype)
        all_targets.append(torch.cat([idx_col, boxes], dim=1))
    if all_targets:
        targets = torch.cat(all_targets, dim=0)
    else:
        targets = torch.zeros((0, 6), dtype=torch.float32)
    return images, targets


class YOLOTargetTransform:
    def __init__(self, img_size: Tuple[int, int]):
        self.img_size = img_size

    def __call__(self, target: Dict) -> torch.Tensor:
        anns = target["annotation"].get("object")
        if isinstance(anns, dict):
            anns = [anns]
        boxes = []
        w_img, h_img = self.img_size
        for obj in anns:
            cls_idx = CLASS_TO_IDX[obj["name"]]
            b = obj["bndbox"]
            xmin = float(b["xmin"])
            ymin = float(b["ymin"])
            xmax = float(b["xmax"])
            ymax = float(b["ymax"])
            # center, width, height
            xc = ((xmin + xmax) / 2) / w_img
            yc = ((ymin + ymax) / 2) / h_img
            bw = (xmax - xmin) / w_img
            bh = (ymax - ymin) / h_img
            boxes.append([cls_idx, xc, yc, bw, bh])
        if not boxes:
            # no objects => return an empty tensor
            return torch.zeros((0,5), dtype=torch.float32)
        return torch.tensor(boxes, dtype=torch.float32)

class YOLOVOCDataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir: str = "./data", batch_size=1000, num_workers=5
    ) -> None:
        super().__init__()
        self.year = None
        self.data_dir = data_dir
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

        self.target_transform = YOLOTargetTransform((self.img_size, self.img_size))

    def prepare_data(self):
        VOCDetection(
            root=self.data_dir, year=self.year, image_set="train", download=True
        )
        VOCDetection(root=self.data_dir, year=self.year, image_set="val", download=True)

    def setup(self, stage: str):
        self.voc_train = VOCDetection(
            root=self.data_dir,
            year=self.year,
            image_set="train",
            download=False,
            transform=self.transform_train,
            target_transform=self.target_transform
        )

        self.voc_test = VOCDetection(
            root=self.data_dir,
            year=self.year,
            image_set="val",
            download=False,
            transform=self.transform_train,
        )

    def train_dataloader(self):
        return DataLoader(
            self.voc_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=False,
            pin_memory=False,
            prefetch_factor=None,
            shuffle=self.shuffle,
        )

    def val_dataloader(self):
        return DataLoader(
            self.voc_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=5,
            shuffle=self.shuffle,
        )

    def test_dataloader(self):
        return DataLoader(
            self.voc_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=5,
            shuffle=self.shuffle,
        )

class YOLOVOCDataModule2012(YOLOVOCDataModule):
    def __init__(self, data_dir = "./data", batch_size=1000, num_workers=5):
        super().__init__(data_dir, batch_size, num_workers)
        self.year = "2012"

class YOLOVOCDataModule2007(YOLOVOCDataModule):
    def __init__(self, data_dir = "./data", batch_size=1000, num_workers=5):
        super().__init__(data_dir, batch_size, num_workers)
        self.year = "2007"