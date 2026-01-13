import logging
from lightning.pytorch.utilities.rank_zero import rank_zero_only

_logger = logging.getLogger("lightning.pytorch")
_logger.setLevel(logging.INFO)

if not _logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("[%(levelname)s] %(message)s")
    h.setFormatter(fmt)
    _logger.addHandler(h)

class _RankZeroLogger:
    @rank_zero_only
    def info(self, msg, *args, **kwargs):
        return _logger.info(msg, *args, **kwargs)

    @rank_zero_only
    def warning(self, msg, *args, **kwargs):
        return _logger.warning(msg, *args, **kwargs)

    @rank_zero_only
    def error(self, msg, *args, **kwargs):
        return _logger.error(msg, *args, **kwargs)

    @rank_zero_only
    def debug(self, msg, *args, **kwargs):
        return _logger.debug(msg, *args, **kwargs)

logger = _RankZeroLogger()