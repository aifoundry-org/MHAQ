from typing import Callable, List, Optional, Union

from torch.utils.data import ConcatDataset, Dataset

from .div2k import Div2K
from .flickr2k import Flickr2K
from .common import pil_loader


class DF2K(Dataset):
    """DF2K Superresolution Dataset (DIV2K + Flickr2K concatenation)."""

    def __init__(
            self,
            root: str,
            scale: Union[int, List[int], None] = None,
            track: Union[str, List[str]] = 'bicubic',
            split: str = 'train',
            transform: Optional[Callable] = None,
            loader: Callable = pil_loader,
            download: bool = False,
            predecode: bool = False,
            preload: bool = False):
        if split != 'train':
            raise ValueError("DF2K only supports the 'train' split.")
        div2k = Div2K(
            root=root,
            scale=scale,
            track=track,
            split=split,
            transform=transform,
            loader=loader,
            download=download,
            predecode=predecode,
            preload=preload,
        )
        flickr2k = Flickr2K(
            root=root,
            scale=scale,
            track=track,
            split=split,
            transform=transform,
            loader=loader,
            download=download,
            predecode=predecode,
            preload=preload,
        )
        self._dataset = ConcatDataset([div2k, flickr2k])

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int):
        return self._dataset[index]
