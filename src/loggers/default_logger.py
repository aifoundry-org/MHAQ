import logging
import lightning.pytorch as pl

class Logger(logging.Logger):
    def __init__(self, name: str = "lightning.pytorch", level=0) -> None:
        super().__init__(name, level)

logger = Logger()
pl_logger = pl._logger