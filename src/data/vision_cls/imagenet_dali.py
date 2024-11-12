import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
import pytorch_lightning as pl

class ImageNetPipeline(Pipeline):
    def __init__(self, data_dir, batch_size, num_threads, device_id, training=True):
        super().__init__(batch_size, num_threads, device_id, seed=12)
        self.input = fn.readers.file(
            file_root=data_dir, random_shuffle=training, name="Reader"
        )
        self.decode = fn.decoders.image(self.input, device="mixed")
        self.res = fn.resize(
            self.decode, resize_shorter=256, interp_type=types.INTERP_LINEAR
        )
        if training:
            self.crop = fn.random_resized_crop(self.res, size=(224, 224))
            self.flip = fn.flip(self.crop, horizontal=1)
            self.hsv = fn.hsv(self.flip, hue=0.2, saturation=1.5, brightness=0.5)
            output = self.hsv
        else:
            self.crop = fn.crop(self.res, crop=(224, 224))
            output = self.crop

        self.cmnp = fn.crop_mirror_normalize(
            output,
            dtype=types.FLOAT,
            output_layout="CHW",
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        )

    def define_graph(self):
        images = self.cmnp
        labels = self.input.label
        return images, labels

class ImageNetDALIDataloader:
    def __init__(self, data_dir, batch_size, num_threads, device_id, training=True):
        pipeline = ImageNetPipeline(
            data_dir=data_dir,
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            training=training,
        )
        pipeline.build()
        self.dali_iterator = DALIClassificationIterator(
            pipeline,
            size=pipeline.epoch_size("Reader"),
            auto_reset=True,
            fill_last_batch=False,
        )

    def __iter__(self):
        for data in self.dali_iterator:
            images = data[0]["data"]
            labels = data[0]["label"].squeeze().long()
            yield images, labels

    def __len__(self):
        return self.dali_iterator._size // self.dali_iterator.batch_size

    def reset(self):
        self.dali_iterator.reset()

class ImageNetDALIDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="./data", batch_size=32, num_threads=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_threads = num_threads

    def setup(self, stage=None):
        self.device_id = self.trainer.local_rank if self.trainer else 0

    def train_dataloader(self):
        return ImageNetDALIDataloader(
            data_dir=f"{self.data_dir}/train",
            batch_size=self.batch_size,
            num_threads=self.num_threads,
            device_id=self.device_id,
            training=True,
        )

    def val_dataloader(self):
        return ImageNetDALIDataloader(
            data_dir=f"{self.data_dir}/val",
            batch_size=self.batch_size,
            num_threads=self.num_threads,
            device_id=self.device_id,
            training=False,
        )