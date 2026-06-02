import lightning.pytorch as pl
import torch.nn.functional as F
import torchmetrics.detection
import torchmetrics
import torch

from src.models.cls.resnet.resnet_cifar import resnet20_cifar10_new
from src.quantization.abc.abc_quant import BaseQuant
from src.quantization.gdnsq.layers.gdnsq_conv2d import NoisyConv2d
from src.quantization.gdnsq.layers.gdnsq_linear import NoisyLinear
from src.quantization.gdnsq.utils.model_helper import ModelHelper
from src.quantization.gdnsq.gdnsq_loss import PotentialLoss
from src.quantization.gdnsq.gdnsq_utils import QNMethod
from src.quantization.gdnsq.utils import model_stats
from src.quantization.gdnsq.utils.fuse_conv_bn import fuse_batchnorm_and_normalize_activation_scales
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


class GDNSQQuant(BaseQuant):
    def __init__(self, config):
        super().__init__(config)

    def module_mappings(self):
        return {
            nn.Conv2d: NoisyConv2d,
            nn.Linear: NoisyLinear,
        }

    def get_loss(self, qmodel):
        if self.config.quantization.params.distillation:
            config_loss = self.config.quantization.params.distillation_loss
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
                raise NotImplementedError(
                    "Loss type are invalid! \
                                          Valid options are: \
                                            [Cross-Entropy,Symmetrical Cross-Entropy, L1, L2, KL, Hellinger]"
                )
        else:
            return qmodel.criterion

    def quantize(self, lmodel: pl.LightningModule, in_place=False):
        self.fusebn = self.config.quantization.fuse_batchnorm
        curriculum = self.config.quantization.params.curriculum
        if self.config.quantization.params.distillation:
            if not self.config.quantization.params.distillation_teacher:
                tmodel = deepcopy(lmodel).eval()
            else:  # XXX fix me
                tmodel = resnet20_cifar10_new(pretrained=True)
        if in_place:
            qmodel = lmodel
        else:
            qmodel = deepcopy(lmodel)

        layer_names, layer_types = zip(
            *[(n, type(m)) for n, m in qmodel.model.named_modules()]
        )

        # The part where original LModule structure gets changed
        qmodel.qscheme = self.qscheme

        if self.config.quantization.params.distillation:
            qmodel.tmodel = tmodel.requires_grad_(False)
        qmodel.wrapped_criterion = PotentialLoss(
            criterion=self.get_loss(qmodel=qmodel),
            p=1,
            a=self.act_bit,
            w=self.weight_bit,
            curriculum_enable=curriculum.enable,
            curriculum_mean=curriculum.mean,
            curriculum_std=curriculum.std,
        )

        # Important step. Replacing training and validation steps
        # with alternated ones.
        if self.config.quantization.params.distillation:
            qmodel.training_step = GDNSQQuant.distillation_noisy_training_step.__get__(
                qmodel, type(qmodel)
            )
            # qmodel.validation_step = GDNSQQuant.noisy_validation_step.__get__(
                # qmodel, type(qmodel)
            # )
        else:
            qmodel.training_step = GDNSQQuant.noisy_train_decorator(qmodel.training_step)
            # qmodel.validation_step = GDNSQQuant.noisy_val_decorator(qmodel.validation_step)

        qmodel.validation_step = GDNSQQuant.noisy_val_decorator(qmodel.validation_step)
        qmodel.test_step = GDNSQQuant.noisy_test_decorator(qmodel.test_step)

        # Replacing layers directly
        qlayers = self._get_layers(lmodel.model, exclude_layers=self.excluded_layers)
        skip_1x1 = self.quant_config.params.skip_1x1_conv
        for layer in qlayers.keys():
            module = attrgetter(layer)(lmodel.model)
            if (not skip_1x1) or module.kernel_size != (1, 1):
                # print(layer + " " + repr(module.kernel_size))
                # preceding_layer_type = layer_types[layer_names.index(layer) - 1]
                # following_layer_type = layer_types[layer_names.index(layer) + 1]
                qmodule = self._quantize_module(module)

                attrsetter(layer)(qmodel.model, qmodule)

        if self.config.quantization.freeze_batchnorm:
            GDNSQQuant.freeze_all_batchnorm_layers(qmodel)

        return qmodel

    def freeze_all_batchnorm_layers(model, freeze=True):
        # Freezes all batch normalization layers in the model.
        # This means they won't update running means/variances
        # during training and their parameters won't receive gradients.
        for module in model.modules():
            # Check for any batch norm variant
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                # Switch to evaluation mode (affects running stats)
                module.eval()
                # Freeze BN params
                module.weight.requires_grad = not freeze
                module.bias.requires_grad = not freeze

    def fuse_conv_bn(self, model: nn.Module):
        if torch.cuda.is_available():
            model = model.cuda()
        n_fused_batchnorm = fuse_batchnorm_and_normalize_activation_scales(model)
        model.validation_step = type(model).validation_step.__get__(model, type(model))

        return n_fused_batchnorm


    @staticmethod
    def noisy_train_decorator(train_step):
        self = train_step.__self__

        def wrapper(batch, batch_idx):
            outputs = (
                train_step(batch, batch_idx),
                *ModelHelper.get_model_values(self.model, self.qscheme),
            )
            loss = self.wrapped_criterion(outputs)

            self.log("Loss/Train loss", loss, prog_bar=True, sync_dist=True)
            self.log(
                "Loss/Base train loss",
                self.wrapped_criterion.base_loss,
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                "Loss/Wloss",
                self.wrapped_criterion.wloss,
                prog_bar=False,
                sync_dist=True,
            )
            self.log(
                "Loss/Aloss",
                self.wrapped_criterion.aloss,
                prog_bar=False,
                sync_dist=True,
            )
            self.log(
                "Loss/Weight reg loss",
                self.wrapped_criterion.weight_reg_loss,
                prog_bar=False,
                sync_dist=True,
            )
            self.log("LR", self.lr, prog_bar=True, sync_dist=True)
            return loss

        return wrapper

    @staticmethod
    def noisy_val_decorator(val_step):
        self = val_step.__self__
        # self._batch_size = self.trainer.config.data.batch_size

        def wrapper(*args):
            loss = val_step(*args)
            # for metric in self.trainer.model.metrics:
            for metric in self.metrics:
                if key := [key for key in self.trainer.logged_metrics.keys() if  metric[0] in key][0]:
                    metric_value = self.trainer.logged_metrics[key]
                # if metric_value := metric[1]._forward_cache:
                # if metric_value := :
                    if isinstance(metric, list):
                        metric_name = metric[0]
                    else:
                        metric_name = metric
                    # Log both gated (`ns_`) and raw metrics.
                    converged = model_stats.is_converged(self)
                    self.log(
                        f"Metric/{metric_name}",
                        metric_value,
                        prog_bar=False,
                        sync_dist=True,
                    )
                    self.log(
                        f"Metric/ns_{metric_name}",
                        metric_value * converged,
                        prog_bar=False,
                        sync_dist=True,
                    )


            self.log("Loss/Validation loss", loss, prog_bar=False, sync_dist=True)

            self.log(
                "Mean weights bit width",
                model_stats.get_weights_bit_width_mean(self.model),
                prog_bar=False,
                # batch_size=self._batch_size,
                sync_dist=True,
            )
            self.log(
                "Actual weights bit width",
                model_stats.get_true_weights_width(self.model, max=False),
                prog_bar=False,
                # batch_size=self._batch_size,
                sync_dist=True,
            )
            self.log(
                "Actual weights max bit width",
                model_stats.get_true_weights_width(self.model),
                prog_bar=False,
                # batch_size=self._batch_size,
                sync_dist=True,
            )
            self.log(
                "Mean activations bit width",
                model_stats.get_activations_bit_width_mean(self.model),
                prog_bar=False,
                # batch_size=self._batch_size,
                sync_dist=True,
            )
            self.log(
                "Actual activations bit widths",
                model_stats.get_true_activations_width(self.model, max=False),
                prog_bar=False,
                # batch_size=self._batch_size,
                sync_dist=True,
            )
            self.log(
                "Actual activations max bit widths",
                model_stats.get_true_activations_width(self.model),
                prog_bar=False,
                # batch_size=self._batch_size,
                sync_dist=True,
            )

        return wrapper

    @staticmethod
    def noisy_test_decorator(test_step):
        self = test_step.__self__

        def wrapper(*args):
            return test_step(*args)

        return wrapper

    @staticmethod  # yes, it's a static method with self argument
    def noisy_step(self, x):
        # now that we set qmodule.qscheme, we can address it in replaced step
        return (self.forward(x), *ModelHelper.get_model_values(self.model, self.qscheme))

    @staticmethod
    def distillation_noisy_training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = GDNSQQuant.noisy_step(self, inputs)

        self.tmodel.eval()
        fp_outputs = self.tmodel.predict_step(inputs, batch_idx)
        loss = self.wrapped_criterion(outputs, fp_outputs)

        self.log("Loss/FP loss", F.cross_entropy(fp_outputs, targets), sync_dist=True)
        self.log(
            "Loss/Base train loss",
            self.wrapped_criterion.base_loss,
            prog_bar=True,
            sync_dist=True,
        )

        self.log("Loss/Train loss", loss, prog_bar=True, sync_dist=True)
        self.log(
            "Loss/Wloss", self.wrapped_criterion.wloss, prog_bar=False, sync_dist=True
        )
        self.log(
            "Loss/Aloss", self.wrapped_criterion.aloss, prog_bar=False, sync_dist=True
        )
        self.log(
            "Loss/Weight reg loss",
            self.wrapped_criterion.weight_reg_loss,
            prog_bar=False,
            sync_dist=True,
        )
        self.log("LR", self.lr, prog_bar=True, sync_dist=True)

        return loss

    # TODO LEGACY
    @staticmethod
    def noisy_training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = GDNSQQuant.noisy_step(self, inputs)
        loss = self.wrapped_criterion(outputs, targets)

        self.log("Loss/Train loss", loss, prog_bar=True, sync_dist=True)
        self.log(
            "Loss/Base train loss",
            self.wrapped_criterion.base_loss,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "Loss/Wloss", self.wrapped_criterion.wloss, prog_bar=False, sync_dist=True
        )
        self.log(
            "Loss/Aloss", self.wrapped_criterion.aloss, prog_bar=False, sync_dist=True
        )
        self.log(
            "Loss/Weight reg loss",
            self.wrapped_criterion.weight_reg_loss,
            prog_bar=False,
            sync_dist=True,
        )
        self.log("LR", self.lr, prog_bar=True, sync_dist=True)

        return loss

    # TODO LEGACY
    @staticmethod
    def noisy_validation_step(self, val_batch, val_index):
        inputs, targets = val_batch

        # targets = self.tmodel(inputs)
        outputs = GDNSQQuant.noisy_step(self, inputs)

        # Oh, I hate this, but here we goo
        try:
            val_loss = self.criterion(outputs[0], targets)
            self.log("Loss/Validation loss", val_loss, prog_bar=False, sync_dist=True)
        except:
            pass
        
        for name, metric in self.metrics:
            metric_value = metric(outputs[0], targets)
            # metric_value = metric(outputs, targets)
            if issubclass(
                metric.__class__, torchmetrics.detection.MeanAveragePrecision
            ):
                self.log(f"Metric/mAP@[.5:.95]", metric_value["map"], prog_bar=False, sync_dist=True)
                self.log(
                    f"Metric/ns_mAP@[.5:.95]",
                    metric_value["map"] * model_stats.is_converged(self),
                    prog_bar=False,
                    sync_dist=True
                )
            else:
                self.log(f"Metric/{name}", metric_value, prog_bar=False, sync_dist=True)
                self.log(
                    f"Metric/ns_{name}",
                    metric_value * model_stats.is_converged(self),
                    prog_bar=False,
                    sync_dist=True
                )

        # Not very optimal approach. Cycling through model two times..
        self.log(
            "Mean weights bit width",
            model_stats.get_weights_bit_width_mean(self.model),
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "Actual weights bit width",
            model_stats.get_true_weights_width(self.model, max=False),
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "Actual weights max bit width",
            model_stats.get_true_weights_width(self.model),
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "Mean activations bit width",
            model_stats.get_activations_bit_width_mean(self.model),
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "Actual activations bit widths",
            model_stats.get_true_activations_width(self.model, max=False),
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "Actual activations max bit widths",
            model_stats.get_true_activations_width(self.model),
            prog_bar=False,
            sync_dist=True,
        )

        # self.log("Loss/Validation loss", val_loss, prog_bar=False)

    @staticmethod
    def noisy_test_step(self, test_batch, test_index):
        inputs, targets = test_batch
        outputs = GDNSQQuant.noisy_step(self, inputs)

        test_loss = self.criterion(outputs[0], targets)
        for name, metric in self.metrics:
            metric_value = metric(outputs[0], targets)
            self.log(f"{name}", metric_value, prog_bar=False, sync_dist=True)

        self.log("test_loss", test_loss, prog_bar=True, sync_dist=True)

    def _init_config(self):
        if self.config:
            self.quant_config = self.config.quantization
            self.act_bit = self.quant_config.act_bit
            self.weight_bit = self.quant_config.weight_bit
            self.weight_guard_bit = self.quant_config.weight_guard_bit
            self.act_guard_bit = self.quant_config.act_guard_bit
            self.excluded_layers = self.quant_config.excluded_layers
            self.qscheme = self.quant_config.qscheme
            # Bias quantization has been removed; always disable it.
            self.quant_bias = False

    def _quantize_module(self, module):
        self.qnmethod = QNMethod[self.quant_config.params.qnmethod]
        disable_activations = self.config.quantization.act_bit == -1
        if isinstance(module, nn.Conv2d):
            qmodule = self._quantize_module_conv2d(
                module,
                disable_activations=disable_activations,
            )
        elif isinstance(module, nn.Linear):
            qmodule = self._quantize_module_linear(
                module,
                disable_activations=disable_activations,
            )
        else:
            raise NotImplementedError(f"Module not supported {type(module)}")

        qmodule.weight = module.weight

        if is_biased(module):
            qmodule.bias = module.bias

        return qmodule

    def _get_quantization_sequence(self, qmodule):
        return qmodule

    def _quantize_module_conv2d(
        self,
        module: nn.Conv2d,
        *,
        disable_activations: bool,
    ):
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
            quant_bias=self.quant_bias,
            disable=disable_activations,
            qnmethod=self.qnmethod,
            weight_guard_bit=self.weight_guard_bit,
            act_guard_bit=self.act_guard_bit,
        )

    def _quantize_module_linear(
        self,
        module: nn.Linear,
        *,
        disable_activations: bool,
    ):
        return NoisyLinear(
            module.in_features,
            module.out_features,
            is_biased(module),
            qscheme=self.qscheme,
            log_s_init=-12,
            disable=disable_activations,
            qnmethod=self.qnmethod,
            weight_guard_bit=self.weight_guard_bit,
            act_guard_bit=self.act_guard_bit,
        )
