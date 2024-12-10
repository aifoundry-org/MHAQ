from src.data.compose import vision
from src.aux.types import DType
from lightning import pytorch as pl


class DatasetComposer:
    def __init__(self, config=None) -> None:
        self.config = config
        self.dataset_type = DType
        self.batch_size: int
        self.num_workers: int
        self.dataset_name: str

    def compose(self) -> pl.LightningDataModule:
        if self.config:
            data_config = self.config.data
            self.dataset_name = data_config.dataset_name
            self.batch_size = data_config.batch_size
            self.num_workers = data_config.num_workers
        else:
            assert self.dataset_name
            assert self.batch_size
            assert self.num_workers
        
        
        try:
            dataset = getattr(vision, self.dataset_name)(
            batch_size=self.batch_size, num_workers=self.num_workers
            )
        except AttributeError as e:
            raise AttributeError(f"{e}, available datasets are {vision.__all__}")

        return dataset
