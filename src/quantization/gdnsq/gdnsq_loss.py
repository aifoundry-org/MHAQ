import torch.nn as nn
import torch


class PotentialLoss(nn.Module):
    def __init__(self, criterion,
                 p=1,
                 a=8,
                 w=4,
                 lossless=False,
                 curriculum_enable=False,
                 curriculum_mean=0.9,
                 curriculum_std=0.5,
                 ) -> None:
        super().__init__()
        self.criterion = criterion
        self.s_weight_loss = torch.tensor(0)
        self.s_act_loss = torch.tensor(0)
        self.weight_reg_loss = torch.tensor(0)
        self.p = torch.tensor(p)
        self.at = a 
        self.wt = w 
        self.lossless = lossless
        self.l_eps = torch.tensor(1e-3)
        self.r_eps = torch.tensor(1e-3)
        self.aloss = torch.tensor(1.0)
        self.wloss = torch.tensor(1.0)
        self.loss_sum = 0.0
        self.cnt = 1

        self.t = 0.0
        self.curriculum_enable = curriculum_enable
        self.curriculum_mean = curriculum_mean
        self.curriculum_std = curriculum_std

    def _get_base_loss(self, output, target=None):
        if target is None:
            return output[0]
        return self.criterion(output[0], target)

    def _get_curriculum_factor(self, ref_tensor):
        if not self.curriculum_enable:
            return ref_tensor.new_tensor(1.0)

        mean = ref_tensor.new_tensor(self.curriculum_mean)
        std = ref_tensor.new_tensor(self.curriculum_std)
        return torch.normal(mean, std).clamp(0, 1)

    def forward(self, output, target=None):
        """Forward method to wrap the main loss.

        Args:
            output (tuple[torch.tensor]): Quantized model outputs as
                `(base_or_pred, log_act_s, log_act_q, log_wght_s, log_w)`.
            target (torch.tensor, optional): Ground truth or distillation target.
                If omitted, `output[0]` is treated as the already-computed base loss.

        Returns:
            torch.tensor: Potential loss result value
        """
        las = output[1]  # log_act_s
        laq = output[2]  # log_act_q
        lws = output[3]  # log_wght_s
        lwq = output[4]  # log_w

        self.base_loss = self._get_base_loss(output, target)
        loss = self.base_loss

        z = loss.new_tensor(0.0)
        wloss0 = (torch.max(z, (lwq - lws) -
                 (self.wt - self.l_eps)).pow(self.p))
        wloss = wloss0.mean()
        wact = (wloss0 > 0).sum() # number of active constraints on weights

        aloss0 = (torch.max(z, (laq - las) -
                 (self.at - self.l_eps)).pow(self.p))
        aloss = aloss0.mean()
        aact = (aloss0 > 0).sum() # number of active constraints on activations

        rloss = loss.pow(self.p)
        cirr = self._get_curriculum_factor(loss)

        calib_mul = self.loss_sum / self.cnt * cirr
        wmul = (wact + self.l_eps) / (wact + aact + self.l_eps)
        amul = (aact + self.l_eps) / (wact + aact + self.l_eps)

        l1, l2 = (1.0, self.t) if self.lossless else (self.t, 1.0)

        ploss = calib_mul * l1 * (wmul * wloss + amul * aloss) + l2 * rloss

        if self.training:
            self.loss_sum += rloss.detach()
            self.cnt += 1

        self.wloss = wloss
        self.aloss = aloss
        self.rloss = rloss
        self.s_weight_loss = -lws.mean()
        self.q_weight_loss = lwq.mean()
        self.s_act_loss = -las.mean()
        self.q_act_loss = laq.mean()
        self.weight_reg_loss = (lwq-lws).max()

        return ploss

class PotentialLossNoPred(PotentialLoss):
    def forward(self, output):
        return super().forward(output, target=None)