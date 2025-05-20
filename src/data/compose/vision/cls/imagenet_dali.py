import os
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
import pytorch_lightning as pl

class ImageNetPipeline(Pipeline):
    def __init__(self, data_dir, batch_size, num_threads, device_id, training=True, shard_id=0, num_shards=1):
        super().__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = fn.readers.file(
            file_root=data_dir,
            random_shuffle=training,
            shard_id=shard_id,
            num_shards=num_shards,
            name="Reader"
        )
        self.decode = fn.decoders.image(self.input, device="mixed")
        if training:
            self.res = fn.random_resized_crop(self.decode, size=224)
            self.hsv = fn.hsv(self.res, hue=0.2, saturation=1.5)
            self.cmnp = fn.crop_mirror_normalize(
                self.hsv,
                dtype=types.FLOAT,
                output_layout="CHW",
                mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                mirror=fn.random.coin_flip(probability=0.5)
            )
        else:
            self.res = fn.resize(self.decode, resize_shorter=256)
            self.cmnp = fn.crop_mirror_normalize(
                self.res,
                dtype=types.FLOAT,
                output_layout="CHW",
                crop=(224, 224),
                mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            )

    def define_graph(self):
        images = self.cmnp
        labels = self.input.label
        return images, labels

class ImageNetDALIDataloader:
    def __init__(self, data_dir, batch_size, num_threads, device_id, training=True, world_size=1, local_rank=0):
        pipeline = ImageNetPipeline(
            data_dir=data_dir,
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            training=training,
            shard_id=local_rank,
            num_shards=world_size,
        )
        pipeline.build()
        self.dali_iterator = DALIClassificationIterator(
            pipelines=pipeline,
            size=pipeline.epoch_size("Reader") // world_size,
            auto_reset=True,
            fill_last_batch=False,
        )
        self.batch_size = batch_size

    def __iter__(self):
        for data in self.dali_iterator:
            images = data[0]["data"]
            labels = data[0]["label"].squeeze().long()
            yield images, labels

    def __len__(self):
        return self.dali_iterator._size // self.batch_size

    def reset(self):
        self.dali_iterator.reset()

class ImageNetDALIDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_threads=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.world_size = 1
        self.local_rank = 0
        self.device_id = 0

    def setup(self, stage=None):
        if "LOCAL_RANK" in os.environ:
            self.local_rank = int(os.environ["LOCAL_RANK"])
        else:
            self.local_rank = 0

        if "WORLD_SIZE" in os.environ:
            self.world_size = int(os.environ["WORLD_SIZE"])
        else:
            self.world_size = 1

        self.device_id = self.local_rank

    def train_dataloader(self):
        return ImageNetDALIDataloader(
            data_dir=os.path.join(self.data_dir, "train"),
            batch_size=self.batch_size,
            num_threads=self.num_threads,
            device_id=self.device_id,
            training=True,
            world_size=self.world_size,
            local_rank=self.local_rank,
        )

    def val_dataloader(self):
        return ImageNetDALIDataloader(
            data_dir=os.path.join(self.data_dir, "val"),
            batch_size=self.batch_size,
            num_threads=self.num_threads,
            device_id=self.device_id,
            training=False,
            world_size=self.world_size,
            local_rank=self.local_rank,
        )