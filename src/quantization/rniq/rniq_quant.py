import lightning.pytorch as pl
import torch.nn.functional as F
import torch

from src.quantization.abc.abc_quant import BaseQuant
from src.quantization.rniq.layers.rniq_conv2d import NoisyConv2d
from src.quantization.rniq.layers.rniq_linear import NoisyLinear
from src.quantization.rniq.layers.rniq_act import NoisyAct
from src.quantization.rniq.utils.model_helper import ModelHelper
from src.quantization.rniq.rniq_loss import PotentialLoss
from src.quantization.rniq.utils import model_stats
from src.aux.qutils import attrsetter, is_biased
from src.aux.loss.hellinger import HellingerLoss
from src.aux.loss.symm_ce_loss import SymmetricalCrossEntropyLoss
from src.aux.loss.distill_ce import CrossEntropyLoss
from src.aux.loss.symm_kl_loss import SymmetricalKL
from src.aux.loss.kl_loss import KL
from src.aux.loss.jsdloss import JSDLoss

from torch import nn
from copy import deepcopy
from operator import attrgetter
from collections import OrderedDict


class RNIQQuant(BaseQuant):
    def module_mappings(self):
        return {
            nn.Conv2d: NoisyConv2d,
            nn.Linear: NoisyLinear,
        }

    def get_distill_loss(self, qmodel):
        if self.config.quantization.distillation:
            config_loss = self.config.quantization.distillation_loss
            if config_loss == "Cross-Entropy":
                return CrossEntropyLoss()
            elif config_loss == "Symmetrical Cross-Entropy":
                return SymmetricalCrossEntropyLoss()
            elif config_loss == "L1":
                return torch.nn.L1Loss()
            elif config_loss == "L2":
                return torch.nn.MSELoss()
            elif config_loss == "KL":
                return KL()
            elif config_loss == "Hellinger":
                return HellingerLoss()
            elif config_loss == "Symmetrical KL":
                return SymmetricalKL()
            elif config_loss == "JSD":
                return JSDLoss()
            else:
                raise NotImplementedError("Loss type are invalid! \
                                          Valid options are: \
                                            [Cross-Entropy,Symmetrical Cross-Entropy, L1, L2, KL, Hellinger]")
        else:
            return qmodel.criterion

    def quantize(self, lmodel: pl.LightningModule, in_place=False):
        if self.config.quantization.distillation:
            tmodel = deepcopy(lmodel).eval()
        if in_place:
            qmodel = lmodel
        else:
            qmodel = deepcopy(lmodel)

        layer_names, layer_types = zip(
            *[(n, type(m)) for n, m in qmodel.model.named_modules()]
        )

        # The part where original LModule structure gets changed
        qmodel._noise_ratio = torch.tensor(1.0)
        qmodel.qscheme = self.qscheme

        if self.config.quantization.distillation:
            qmodel.tmodel = tmodel.requires_grad_(False)

        qmodel.wrapped_criterion = PotentialLoss(
            criterion=self.get_distill_loss(qmodel=qmodel),
            alpha=(1, 0.1, 1),
            # alpha=self.alpha,
            lmin=0,
            p=1,
            a=self.act_bit,
            w=self.weight_bit,
            scale_momentum=0.9,
        )

        qmodel.noise_ratio = RNIQQuant.noise_ratio.__get__(
            qmodel, type(qmodel))

        # Important step. Replacing training and validation steps
        # with alternated ones.
        if self.config.quantization.distillation:
            qmodel.training_step = RNIQQuant.distillation_noisy_training_step.__get__(
                qmodel, type(qmodel)
            )
        else:
            qmodel.training_step = RNIQQuant.noisy_training_step.__get__(
                qmodel, type(qmodel)
            )

        qmodel.validation_step = RNIQQuant.noisy_validation_step.__get__(
            qmodel, type(qmodel)
        )
        qmodel.test_step = RNIQQuant.noisy_test_step.__get__(
            qmodel, type(qmodel))

        # Replacing layers directly
        qlayers = self._get_layers(
            lmodel.model, exclude_layers=self.excluded_layers)
        for layer in qlayers.keys():
            module = attrgetter(layer)(lmodel.model)
            if module.kernel_size != (1,1):
                print(layer + " " + repr(module.kernel_size))
                preceding_layer_type = layer_types[layer_names.index(layer) - 1]
                if issubclass(preceding_layer_type, nn.ReLU):
                    qmodule = self._quantize_module(
                        module, signed_Activations=False)
                else:
                    qmodule = self._quantize_module(
                        module, signed_Activations=False)

                attrsetter(layer)(qmodel.model, qmodule)

        return qmodel

    @staticmethod
    def noise_ratio(self, x=None):
        if x != None:
            for module in self.modules():
                if hasattr(module, "_noise_ratio"):
                    module._noise_ratio.data = x.clone().detach()
        return self._noise_ratio

    @staticmethod  # yes, it's a static method with self argument
    def noisy_step(self, x):
        # now that we set qmodule.qscheme, we can address it in replaced step
        return (self.model(x), *ModelHelper.get_model_values(self.model, self.qscheme))

    @staticmethod
    def distillation_noisy_training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = RNIQQuant.noisy_step(self, inputs)

        self.tmodel.eval()
        fp_outputs = self.tmodel(inputs)
        loss = self.wrapped_criterion(outputs, fp_outputs)

        self.log("Loss/FP loss", F.cross_entropy(fp_outputs, targets))
        self.log("Loss/Train loss", loss, prog_bar=True)
        self.log(
            "Loss/Base train loss", self.wrapped_criterion.base_loss, prog_bar=True
        )
        self.log("Loss/Wloss", self.wrapped_criterion.wloss, prog_bar=False)
        self.log("Loss/Aloss", self.wrapped_criterion.aloss, prog_bar=False)
        self.log(
            "Loss/Weight reg loss",
            self.wrapped_criterion.weight_reg_loss,
            prog_bar=False,
        )
        self.log("LR", self.lr, prog_bar=True)

        return loss

    @staticmethod
    def noisy_training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = RNIQQuant.noisy_step(self, inputs)
        loss = self.wrapped_criterion(outputs, targets)

        self.log("Loss/Train loss", loss, prog_bar=True)
        self.log(
            "Loss/Base train loss", self.wrapped_criterion.base_loss, prog_bar=True
        )
        self.log("Loss/Wloss", self.wrapped_criterion.wloss, prog_bar=False)
        self.log("Loss/Aloss", self.wrapped_criterion.aloss, prog_bar=False)
        self.log(
            "Loss/Weight reg loss",
            self.wrapped_criterion.weight_reg_loss,
            prog_bar=False,
        )
        self.log("LR", self.lr, prog_bar=True)

        return loss

    @staticmethod
    def noisy_validation_step(self, val_batch, val_index):
        inputs, targets = val_batch

        # targets = self.tmodel(inputs)
        # self.noise_ratio(0.0)
        outputs = RNIQQuant.noisy_step(self, inputs)

        val_loss = self.criterion(outputs[0], targets)
        for name, metric in self.metrics:
            metric_value = metric(outputs[0], targets)
            # metric_value = metric(outputs, targets)
            self.log(f"Metric/{name}", metric_value, prog_bar=False)
            self.log(f"Metric/ns_{name}", metric_value * (self.noise_ratio()==0), prog_bar=False) 

        # Not very optimal approach. Cycling through model two times..
        self.log(
            "Mean weights bit width",
            model_stats.get_weights_bit_width_mean(self.model),
            prog_bar=False,
        )
        self.log(
            "Actual weights bit width",
            model_stats.get_true_weights_width_mean(self.model),
            prog_bar=False
        )
        self.log(
            "Mean activations bit width",
            model_stats.get_activations_bit_width_mean(self.model),
            prog_bar=False,
        )
        self.log(
            "Actual activations bit widths",
            model_stats.get_true_activations_width_mean(self.model),
            prog_bar=False
        )

        self.log("Loss/Validation loss", val_loss, prog_bar=False)
        # idea is to modify val loss during the stage when model is not converged 
        # to use this metric later for the chckpoint callback
        self.log("Loss/ns_val_loss", val_loss + (10 * self.noise_ratio()), prog_bar=False) 

    @staticmethod
    def noisy_test_step(self, test_batch, test_index):
        inputs, targets = test_batch
        # self.noise_ratio(0.0)
        outputs = RNIQQuant.noisy_step(self, inputs)

        test_loss = self.criterion(outputs[0], targets)
        for name, metric in self.metrics:
            metric_value = metric(outputs[0], targets)
            self.log(f"{name}", metric_value, prog_bar=False)

        self.log("test_loss", test_loss, prog_bar=True)

    def _init_config(self):
        if self.config:
            self.quant_config = self.config.quantization
            self.act_bit = self.quant_config.act_bit
            self.weight_bit = self.quant_config.weight_bit
            self.excluded_layers = self.quant_config.excluded_layers
            self.qscheme = self.quant_config.qscheme

    def _quantize_module(self, module, signed_Activations):
        if isinstance(module, nn.Conv2d):
            qmodule = self._quantize_module_conv2d(module)
        elif isinstance(module, nn.Linear):
            qmodule = self._quantize_module_linear(module)
        else:
            raise NotImplementedError(f"Module not supported {type(module)}")

        qmodule.weight = module.weight

        if is_biased(module):
            qmodule.bias = module.bias

        qmodule = self._get_quantization_sequence(qmodule, signed_Activations)

        return qmodule

    def _get_quantization_sequence(self, qmodule, signed_activations):
        disabled = False
        if self.config.quantization.act_bit == -1 or self.config.quantization.act_bit > 20:
            disabled = True
        sequence = nn.Sequential(
            OrderedDict(
                [
                    ("activations_quantizer", NoisyAct(signed=signed_activations, disable=disabled)),
                    ("0", qmodule),
                ]
            )
        )

        return sequence

    def _quantize_module_conv2d(self, module: nn.Conv2d):
        return NoisyConv2d(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
            is_biased(module),
            module.padding_mode,
            qscheme=self.qscheme,
            log_s_init=-12,
        )

    def _quantize_module_linear(self, module: nn.Linear):
        return NoisyLinear(
            module.in_features,
            module.out_features,
            is_biased(module),
            qscheme=self.qscheme,
            log_s_init=-12,
        )
