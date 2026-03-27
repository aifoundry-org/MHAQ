from lightning.pytorch.callbacks import Callback
from typing_extensions import override
from pathlib import Path
from typing import Iterable, Optional, Any, Dict, Tuple
import time
import math

try:
    import torch
except Exception:
    torch = None

def _is_global_zero(trainer) -> bool:
    return getattr(trainer, "is_global_zero", True)

def _to_float(x: Any) -> float:
    try:
        return float(getattr(x, "item", lambda: x)())
    except Exception:
        return float("nan")


def _first_present(d: dict, candidates: Iterable[str]) -> Optional[float]:
    for k in candidates:
        if k in d:
            return _to_float(d[k])
    return None

def _fmt(x: Optional[float], fmt: str) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nan"
    return format(float(x), fmt)

class PrintClassificationMetrics(Callback):
    def __init__(self, filename: str = "task2_logs/logs.txt", rank_zero_only: bool = True):
        super().__init__()
        self.filename = filename
        self.rank_zero_only = rank_zero_only
        self._path = Path(self.filename)
        self._epoch_start_time = None

        self.base_loss_keys = (
            "Loss/Base train loss",
            "Base Loss",
            "Loss/Base loss",
            "Loss/Base",
        )
        self.top1_keys = (
            "Accuracy_top1",
            "Metric/Accuracy_top1",
            "Metric/ns_Accuracy_top1",
            "Accuracy/top1",
            "acc1",
        )
        self.mean_w_bits_keys = (
            "Mean weights bit width",
            "Weights/mean_bitwidth",
        )
        self.mean_a_bits_keys = (
            "Mean activations bit width",
            "Activations/mean_bitwidth",
        )
        self.val_loss_keys = ("Loss/Validation loss", "val_loss", "val_loss_epoch")
        self.lr_keys = ("LR",)
        self.max_w_bits_keys = ("Actual weights max bit width", "BitW/max_w")
        self.max_a_bits_keys = ("Actual activations max bit widths", "BitW/max_a")

    def _write(self, text: str):
        assert self._path is not None, "log path is not initialized"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(text.rstrip() + "\n")

    @override
    def on_fit_start(self, trainer, pl_module):
        if self.rank_zero_only and not _is_global_zero(trainer):
            return
        self._write(
            "# epoch | BaseLoss | ValLoss | Acc@1 | MeanWBits | MeanABits | MaxWBits | MaxABits | "
            "LR | EpochTime(s)"
        )

    @override
    def on_train_epoch_start(self, trainer, pl_module):
        if self.rank_zero_only and not _is_global_zero(trainer):
            return
        self._epoch_start_time = time.time()

    @override
    def on_validation_epoch_end(self, trainer, pl_module):
        if self.rank_zero_only and not _is_global_zero(trainer):
            return super().on_validation_epoch_end(trainer, pl_module)

        cbs = trainer.callback_metrics
        epoch = int(getattr(trainer, "current_epoch", -1))

        base_loss = _first_present(cbs, self.base_loss_keys)
        acc1 = _first_present(cbs, self.top1_keys)
        mean_w = _first_present(cbs, self.mean_w_bits_keys)
        mean_a = _first_present(cbs, self.mean_a_bits_keys)
        val_loss = _first_present(cbs, self.val_loss_keys)
        lr = _first_present(cbs, self.lr_keys)
        max_w = _first_present(cbs, self.max_w_bits_keys)
        max_a = _first_present(cbs, self.max_a_bits_keys)

        t_epoch = None
        if self._epoch_start_time is not None:
            t_epoch = time.time() - self._epoch_start_time

        base_loss = base_loss if base_loss is not None else float("nan")
        acc1 = acc1 if acc1 is not None else float("nan")
        mean_w = mean_w if mean_w is not None else float("nan")
        mean_a = mean_a if mean_a is not None else float("nan")

        self._write(
            f"{epoch:6d} | base={base_loss:.6f} | val={val_loss or float('nan'):.6f} | "
            f"acc1={acc1:.4f} | Wμ={mean_w:.3f} | Aμ={mean_a:.3f} | Wmax={max_w or float('nan'):.3f} | Amax={max_a or float('nan'):.3f} |"
            f"LR={lr or float('nan'):.5g} | "
            f"t={t_epoch or float('nan'):.2f}s"
        )

        return super().on_validation_epoch_end(trainer, pl_module)


def _collect_srbench_metrics(callback_metrics: dict,
                             dataset_names: Tuple[str, ...] = ("set5", "set14", "b100", "urban100", "manga109")) -> Dict[str, float]:
    out = {}
    for k, v in callback_metrics.items():
        lk = str(k).lower()
        val = _to_float(v)

        if "psnr" in lk or "ssim" in lk or "lpips" in lk:
            for ds in dataset_names:
                if ds in lk:
                    if "psnr" in lk:
                        out[f"psnr_{ds}"] = val
                    elif "ssim" in lk:
                        out[f"ssim_{ds}"] = val
                    elif "lpips" in lk:
                        out[f"lpips_{ds}"] = val
    return out


def _collect_bitwidth_metrics(
        callback_metrics: dict,
        dataset_names: Tuple[str, ...] = ("set5", "set14", "b100", "urban100", "manga109"),
) -> Dict[str, float]:
    out = {}

    for k, v in callback_metrics.items():
        lk = str(k).lower()
        val = _to_float(v)

        ds_found = None
        for ds in dataset_names:
            if ds in lk:
                ds_found = ds
                break
        if not ds_found:
            continue

        if "mean weights bit width" in lk:
            out[f"w_mean_{ds_found}"] = val
        elif "mean activations bit width" in lk:
            out[f"a_mean_{ds_found}"] = val
        elif "actual weights max bit width" in lk:
            out[f"w_max_{ds_found}"] = val
        elif "actual weights bit width" in lk:
            out[f"w_actual_{ds_found}"] = val
        elif "actual activations max bit widths" in lk:
            out[f"a_max_{ds_found}"] = val
        elif "actual activations bit widths" in lk:
            out[f"a_actual_{ds_found}"] = val

    return out



class PrintSrMetrics(Callback):
    def __init__(self, filename: str = "task2_logs/logs.txt", rank_zero_only: bool = True):
        super().__init__()
        self.filename = filename
        self.rank_zero_only = rank_zero_only
        self._path = Path(self.filename)

        self._train_epoch_start = None
        self._val_epoch_start = None

        self.train_loss_keys = ("train/loss", "Loss/Train", "Loss/Train loss", "loss/train", "loss")
        self.val_loss_keys = ("val/loss", "Loss/Validation loss", "val_loss", "val_loss_epoch")

        self.lr_keys = ("LR", "lr", "train/lr", "optimizer/lr")

        self.mean_w_bits_keys = ("Mean weights bit width", "Weights/mean_bitwidth")
        self.mean_a_bits_keys = ("Mean activations bit width", "Activations/mean_bitwidth")

    def _write(self, text: str):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(text.rstrip() + "\n")

    def _gpu_mem_mb(self) -> Optional[float]:
        if torch is None:
            return None
        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / (1024 ** 2))
        return None

    @override
    def on_fit_start(self, trainer, pl_module):
        if self.rank_zero_only and not _is_global_zero(trainer):
            return

        self._write(
            "# epoch | train_loss | val_loss | "
            "LR | t_train(s) | t_val(s) | gpu_mem(MB) | SRBench(psnr/ssim...)"
        )

    @override
    def on_train_epoch_start(self, trainer, pl_module):
        if self.rank_zero_only and not _is_global_zero(trainer):
            return
        self._train_epoch_start = time.perf_counter()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @override
    def on_validation_epoch_start(self, trainer, pl_module):
        if self.rank_zero_only and not _is_global_zero(trainer):
            return
        self._val_epoch_start = time.perf_counter()

    def _dump_keys_once(self, cbs: dict):
        if getattr(self, "_did_dump_keys", False):
            return
        self._did_dump_keys = True
        keys = sorted([str(k) for k in cbs.keys()])

    @override
    def on_validation_epoch_end(self, trainer, pl_module):
        if self.rank_zero_only and not _is_global_zero(trainer):
            return super().on_validation_epoch_end(trainer, pl_module)

        cbs = trainer.callback_metrics
        self._dump_keys_once(cbs)

        epoch = int(getattr(trainer, "current_epoch", -1))

        train_loss = _first_present(cbs, self.train_loss_keys)
        val_loss = _first_present(cbs, self.val_loss_keys)
        lr = _first_present(cbs, self.lr_keys)

        t_train = None
        if self._train_epoch_start is not None:
            t_train = time.perf_counter() - self._train_epoch_start

        t_val = None
        if self._val_epoch_start is not None:
            t_val = time.perf_counter() - self._val_epoch_start

        mem_mb = self._gpu_mem_mb()

        srbench = _collect_srbench_metrics(cbs)
        srbench_str = ""
        if srbench:
            parts = []
            for k in sorted(srbench.keys()):
                parts.append(f"{k}={_fmt(srbench[k], '.3f')}")
            srbench_str = " | " + ", ".join(parts)

        bw = _collect_bitwidth_metrics(cbs)
        bw_str = ""
        if bw:
            parts = []
            for k in sorted(bw.keys()):
                parts.append(f"{k}={_fmt(bw[k], '.3f')}")
            bw_str = " | " + ", ".join(parts)

        self._write(
            f"{epoch:6d} | "
            f"train={_fmt(train_loss, '.6f')} | val={_fmt(val_loss, '.6f')} | "
            f"lr={_fmt(lr, '.5g')} | "
            f"t_train={_fmt(t_train, '.2f')} | t_val={_fmt(t_val, '.2f')} | "
            f"mem={_fmt(mem_mb, '.1f')}"
            f"{srbench_str} | "
            f"{bw_str}"
        )

        return super().on_validation_epoch_end(trainer, pl_module)