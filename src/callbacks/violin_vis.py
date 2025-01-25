import lightning.pytorch as pl
import torch
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import seaborn as sns

from src.aux.loss.hellinger import HellingerLoss
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch import Trainer, LightningModule

class DistillViolinVis(Callback):
    def __init__(self, name) -> None:
        self.defined_metrics = {
            "MAE": F.l1_loss,
            "MSE": F.mse_loss,
            "Hellinger": HellingerLoss._distance,
            "Cross-Entropy": F.cross_entropy,
            }
        self.metric_outputs = {metric: [] for metric in self.defined_metrics}
            
    def on_validation_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        for metric_name, metric_func in self.defined_metrics.items():
            metric_value = metric_func(pl_module.tmodel(batch[0]), pl_module(batch[0])).item()
            self.metric_outputs[metric_name].append(metric_value)
        return super().on_validation_batch_start(trainer, pl_module, batch, batch_idx, dataloader_idx)
    
    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._create_violin_plots()
        self._reset_metrics()
        
    def _create_violin_plots(self) -> None:
        data = []
        metric_names = list(self.defined_metrics.keys())

        # Prepare data for seaborn plot
        for metric_name in metric_names:
            mean = np.mean(self.metric_outputs[metric_name])
            std = np.std(self.metric_outputs[metric_name])
            for value in self.metric_outputs[metric_name]:
                data.append({"Metric": metric_name, "Value": (value - mean)/std})

        # Convert to DataFrame for seaborn
        import pandas as pd
        df = pd.DataFrame(data)

        plt.figure(figsize=(8, 6))
        sns.violinplot(data=df, x="Value", y="Metric", inner=None, palette="muted")
        sns.stripplot(data=df, x="Value", y="Metric", color="black", size=5, jitter=True, alpha=0.6)

        plt.title("Violin Plot for Teacher vs. Student Metrics")
        plt.xlabel("Metric Value")
        plt.tight_layout()
        plt.savefig("Violin_plot_distill.png")
    
    # def _create_violin_plots(self) -> None:
    #     num_metrics = len(self.defined_metrics)
    #     fig, ax = plt.subplots(figsize=(7, 5))

    #     data = [self.metric_outputs[metric_name] for metric_name in self.defined_metrics]
    #     ax.violinplot(data, vert=False, showmeans=True)
    #     ax.set_title("Violin Plot for Teacher vs. Student Metrics")
    #     ax.set_yticks(range(1, num_metrics + 1))
    #     ax.set_yticklabels(list(self.defined_metrics.keys()))
    #     ax.set_xlabel("Metric Value")

    #     plt.tight_layout()
    #     plt.savefig("Violin_plot_distill.png")
    #     # plt.show()

    def _reset_metrics(self) -> None:
        self.metric_outputs = {metric: [] for metric in self.defined_metrics}