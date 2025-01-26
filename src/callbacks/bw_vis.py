import wandb
import warnings
import io
import wandb.plot
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch import Trainer, LightningModule

from src.quantization.rniq.layers.rniq_conv2d import NoisyConv2d
from src.quantization.rniq.layers.rniq_linear import NoisyLinear
from src.quantization.rniq.layers.rniq_act import NoisyAct

from src.quantization.rniq.utils.model_stats import get_true_weights_width
from src.quantization.rniq.utils.model_stats import get_true_activations_width
from src.loggers import WandbLogger

warnings.filterwarnings("ignore", category=FutureWarning)

class LayersWidthVis(Callback):
    def __init__(self):
        super().__init__()
    

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule):
        data = []
        
        for (name, module) in pl_module.model.named_modules():
            if isinstance(module, (NoisyConv2d, NoisyLinear)):
                data.append((name, get_true_weights_width(module, max=False)))
            elif isinstance(module, NoisyAct):
                data.append((name, get_true_activations_width(module, max=False)))
            
        color_map = {True: "red", False: "blue"}
        
        fig_height = len(data) * 0.2 # Set 0.2 inch per bar
        fig_width = 10  # Set a fixed width
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        df = pd.DataFrame(data=data, columns=["Name", "Value"])      
        df["act"] = df["Name"].str.contains("activations_quantizer", case=False)
        
        palette = df["act"].map(color_map)
        
        sns.barplot(data=df, x="Value", y="Name", ax=ax, palette=palette.values.tolist())
        
        plt.yticks(fontsize=8)
        plt.xticks(fontsize=8) 
        plt.tight_layout()
        trainer.logger.log_image("layer_dist", [wandb.Image(fig)])
        trainer.logger.log_table(key="widths", columns=["Name", "Value"], data=data)

        return super().on_validation_end(trainer, pl_module)
