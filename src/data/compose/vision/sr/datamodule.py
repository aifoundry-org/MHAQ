from typing import Iterable, List, Optional, Sequence, Dict
from lightning import pytorch as pl
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from .datasets import B100, Div2K, Set5, Set14, Urban100
from .transforms import (
    AdjustToScale,
    Compose,
    RandomCrop,
    RandomFlipTurn,
    ToTensor,
)


_BENCHMARK_REGISTRY = {
    "Set5": Set5,
    "Set14": Set14,
    "B100": B100,
    "Urban100": Urban100,
}


class _SRPairDataset(Dataset):
    """Wraps the FolderByDir datasets to return (lr, hr) tensors."""

    def __init__(self, dataset: Dataset, dataset_name: str | None = None) -> None:
        self._dataset = dataset
        self.dataset_name = dataset_name

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int):
        sample = self._dataset[index]
        if not isinstance(sample, (list, tuple)) or len(sample) < 2:
            raise ValueError(
                "Super-resolution datasets are expected to return a sequence "
                "of [HR, LR] images."
            )
        hr = sample[0]
        lr = sample[1]
        if self.dataset_name is not None:
            return lr, hr, self.dataset_name
        return lr, hr


class SuperResolutionDataModule(pl.LightningDataModule):
    """
    LightningDataModule that wires the super-resolution datasets.

    DIV2K is used for training while Set5, Set14, B100 and Urban100 are
    concatenated for validation/test by default.
    """

    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 16,
        num_workers: int = 4,
        scale: int = 4,
        track: str = "bicubic",
        eval_batch_size: int = 1,
        patch_size_train: int = 96,
        patch_size_val: int = 384,
        benchmark_sets: Optional[Iterable[str]] = None,
        div2k_splits: Optional[Iterable[str]] = None,
        benchmark_split: str = "val",
        # download: bool = False,
        download: bool = True,
        preload: bool = False,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.scale = scale
        self.track = track
        self.patch_size_train = patch_size_train
        self.patch_size_val = patch_size_val
        self.download = download
        self.preload = preload
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory

        if benchmark_sets is None:
            benchmark_sets = ("Set5", "Set14", "B100", "Urban100")
        self.benchmark_sets: List[str] = [name for name in benchmark_sets]
        if not div2k_splits:
            div2k_splits = ("train", "val")
        self.div2k_splits: Sequence[str] = tuple(div2k_splits)
        self.benchmark_split = benchmark_split

        self.train_dataset: Optional[Dataset] = None
        self._benchmark_dataset: Optional[Dataset] = None

    def prepare_data(self) -> None:
        if not self.download:
            return

        for split in self.div2k_splits:
            Div2K(
                root=self.data_dir,
                scale=self.scale,
                track=self.track,
                split=split,
                transform=None,
                download=True,
                preload=self.preload,
                predecode=not self.preload,
            )

        # for _, ds in self._benchmark_classes():
        #     ds(
        #         root=self.data_dir,
        #         scale=self.scale,
        #         split=self.benchmark_split,
        #         transform=None,
        #         download=True,
        #         preload=self.preload,
        #         predecode=not self.preload,
        #     )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            if self.train_dataset is None:
                self.train_dataset = self._build_train_dataset()

        # if stage in (None, "fit", "validate", "test", "predict"):
            # self._ensure_benchmark_dataset()

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup('fit') before requesting train dataloader.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def val_dataloader(self) -> Dict:
        return self._build_val_datasets()

    def test_dataloader(self) -> Dict:
        return self._build_val_datasets()

    def predict_dataloader(self) -> Dict:
        return self._build_val_datasets()

    def _build_train_dataset(self) -> Dataset:
        datasets: List[Dataset] = []
        for split in self.div2k_splits:
            dataset = Div2K(
                root=self.data_dir,
                scale=self.scale,
                track=self.track,
                split=split,
                transform=self._build_train_transform(),
                download=self.download,
                preload=self.preload,
                predecode=not self.preload,
            )
            datasets.append(_SRPairDataset(dataset))
        if not datasets:
            raise RuntimeError("No DIV2K splits configured for training.")
        if len(datasets) == 1:
            return datasets[0]
        return ConcatDataset(datasets)

    def _build_val_datasets(self) -> dict:
        datasets = {
            "Set5": DataLoader(_SRPairDataset(Set5(root=self.data_dir, scale=self.scale, download=self.download, transform=self._build_eval_transform()), "Set5"), batch_size=1),
            "Set14": DataLoader(_SRPairDataset(Set14(root=self.data_dir, scale=self.scale, download=self.download, transform=self._build_eval_transform()), "Set14"), batch_size=1),
            "B100": DataLoader(_SRPairDataset(B100(root=self.data_dir, scale=self.scale, download=self.download, transform=self._build_eval_transform()), "B100"), batch_size=1),
            "Urban100": DataLoader(_SRPairDataset(Urban100(root=self.data_dir, scale=self.scale, download=self.download, transform=self._build_eval_transform()), "Urban100"), batch_size=1)
        }
        # for dataset_name, dataset_cls in self._benchmark_classes():
        #     ds = dataset_cls(
        #         root=self.data_dir,
        #         scale=self.scale,
        #         split=split,
        #         transform=self._build_eval_transform(),
        #         download=self.download,
        #         preload=self.preload,
        #         predecode=not self.preload,
        #     )
        #     datasets[dataset_name] = ds
        if not datasets:
            raise RuntimeError("No benchmark datasets configured for SR evaluation.")
        return datasets

    def _build_eval_loader(self, dataset: Dataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def _build_train_transform(self) -> Compose:
        return Compose(
            [
                RandomCrop(
                    self.patch_size_train,
                    scales=[1, self.scale],
                    margin=0.5,
                ),
                RandomFlipTurn(),
                ToTensor(),
            ]
        )

    def _build_eval_transform(self) -> Compose:
        return Compose(
            [
                AdjustToScale(scales=[1, self.scale]),
                ToTensor(),
            ]
        )

    # def _ensure_benchmark_dataset(self) -> Dataset:
    #     if self._benchmark_dataset is None:
    #         self._benchmark_dataset = self._build_benchmark_datasets(self.benchmark_split)
    #     return self._benchmark_dataset

    # def _benchmark_classes(self):
    #     for name in self.benchmark_sets:
    #         key = name if isinstance(name, str) else str(name)
    #         dataset_cls = _BENCHMARK_REGISTRY.get(key)
    #         if dataset_cls is None:
    #             raise ValueError(
    #                 f"Unknown super-resolution benchmark dataset '{name}'. "
    #                 f"Available options: {list(_BENCHMARK_REGISTRY.keys())}"
    #             )
    #         yield key, dataset_cls
