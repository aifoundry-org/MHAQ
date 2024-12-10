import os
import torch
import pytorch_lightning as pl
from lightning import pytorch as pl
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy

# Function to save CIFAR10 dataset to disk


def save_cifar10_to_disk(dataset, root_dir):
    from PIL import Image

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # Retrieve classes from the dataset or its underlying dataset
    def get_classes(ds):
        while isinstance(ds, torch.utils.data.Subset):
            ds = ds.dataset
        return ds.classes

    classes = get_classes(dataset)

    for idx, (img, label) in enumerate(dataset):
        class_name = classes[label]
        class_dir = os.path.join(root_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        img_path = os.path.join(class_dir, f"{idx}.png")
        img.save(img_path)

# DALI Pipeline for training and validation


def create_dali_pipeline(batch_size, num_threads, device_id, data_dir, crop, size, shard_id=0, num_shards=1, is_training=True):
    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads,
                    device_id=device_id, seed=12, prefetch_queue_depth=5)
    with pipe:
        jpegs, labels = fn.readers.file(
            file_root=data_dir,
            shard_id=shard_id,
            num_shards=num_shards,
            random_shuffle=is_training,
            pad_last_batch=True,
            name="Reader"
        )
        images = fn.decoders.image(
            jpegs, device="mixed", output_type=types.RGB)
        if is_training:
            images = fn.random_resized_crop(images, size=crop)
            images = fn.flip(images, horizontal=1)
        else:
            images = fn.resize(images, resize_shorter=size)
        images = fn.crop_mirror_normalize(
            images,
            dtype=types.FLOAT,
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            output_layout="CHW",
        )
        pipe.set_outputs(images, labels)
    return pipe

# Wrapper to yield (inputs, targets) tuples


class DALIWrapper:
    def __init__(self, dali_iterator):
        self.dali_iterator = dali_iterator

    def __len__(self):
        return len(self.dali_iterator)

    def __iter__(self):
        for data in self.dali_iterator:
            inputs = data[0]["data"]
            targets = data[0]["label"].squeeze().long()
            yield inputs, targets
        self.dali_iterator.reset()

# Data Module using DALI


class CIFAR10DALIDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir="./data",
                 batch_size=32,
                 num_workers=4,
                 device_id=0):
        super().__init__()  # Ensure you call the parent class constructor
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device_id = device_id  # GPU device ID
        self.num_classes = 10

    def prepare_data(self):
        # Download CIFAR10 dataset
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

        # Prepare data directories
        train_dir = os.path.join(self.data_dir, "train")
        val_dir = os.path.join(self.data_dir, "val")
        test_dir = os.path.join(self.data_dir, "test")

        # Check if data is already prepared
        if not os.path.exists(train_dir) or not os.path.exists(val_dir):
            # Load dataset
            full_train = CIFAR10(self.data_dir, train=True, download=False)
            # Split into train and validation sets
            train_size = int(0.9 * len(full_train))
            val_size = len(full_train) - train_size
            cifar_train, cifar_val = random_split(
                full_train,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )
            # Save datasets to disk
            save_cifar10_to_disk(cifar_train, train_dir)
            save_cifar10_to_disk(cifar_val, val_dir)
            print("Training and validation data saved to disk.")
        else:
            print("Training and validation data already prepared.")

        if not os.path.exists(test_dir):
            cifar_test = CIFAR10(self.data_dir, train=False, download=False)
            save_cifar10_to_disk(cifar_test, test_dir)
            print("Test data saved to disk.")
        else:
            print("Test data already prepared.")

    def setup(self, stage=None):
        self.val_pipeline = create_dali_pipeline(
            batch_size=self.batch_size,
            num_threads=self.num_workers,
            device_id=self.device_id,
            data_dir=os.path.join(self.data_dir, "val"),
            crop=(32, 32),
            size=32,
            is_training=False,
        )
        self.val_pipeline.build()

        if stage == "fit" or stage is None:
            self.train_pipeline = create_dali_pipeline(
                batch_size=self.batch_size,
                num_threads=self.num_workers,
                device_id=self.device_id,
                data_dir=os.path.join(self.data_dir, "train"),
                crop=(32, 32),
                size=32,
                is_training=True,
            )
            self.train_pipeline.build()

        if stage == "test" or stage is None:
            self.test_pipeline = create_dali_pipeline(
                batch_size=self.batch_size,
                num_threads=self.num_workers,
                device_id=self.device_id,
                data_dir=os.path.join(self.data_dir, "test"),
                crop=(32, 32),
                size=32,
                is_training=False,
            )
            self.test_pipeline.build()

    def train_dataloader(self):
        dali_iter = DALIClassificationIterator(
            self.train_pipeline,
            reader_name="Reader",
            last_batch_policy=LastBatchPolicy.PARTIAL,
            auto_reset=True,
        )
        return DALIWrapper(dali_iter)

    def val_dataloader(self):
        dali_iter = DALIClassificationIterator(
            self.val_pipeline,
            reader_name="Reader",
            last_batch_policy=LastBatchPolicy.PARTIAL,
            auto_reset=True,
        )
        return DALIWrapper(dali_iter)

    def test_dataloader(self):
        dali_iter = DALIClassificationIterator(
            self.test_pipeline,
            reader_name="Reader",
            last_batch_policy=LastBatchPolicy.PARTIAL,
            auto_reset=True,
        )
        return DALIWrapper(dali_iter)
