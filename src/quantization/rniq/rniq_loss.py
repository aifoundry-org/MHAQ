import numpy as np
import torch.nn as nn
import torch


class PotentialLoss(nn.Module):
    def __init__(self, criterion,
                 p=1,
                 a=8,
                 w=4,
                 lossless=False
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


    def forward(self, output, target):
        """Forward method to wrapping main loss

        Args:
            output (tuple[torch.tensor]): Output for main loss
            stated as (x ,log_act_s, log_act_q, log_wght_s, log_w)
            target (torch.tensor): ground truth to calculate loss for

        Returns:
            torch.tensor: Potential loss result value
        """
        prd = output[0]  # prediction
        las = output[1]  # log_act_s
        laq = output[2]  # log_act_q
        lws = output[3]  # log_wght_s
        lwq = output[4]  # log_w

        self.base_loss = self.criterion(prd, target)
        loss = self.base_loss

        z = torch.tensor(0)
        wloss0 = (torch.max(z, (lwq - lws) -
                 (self.wt - self.l_eps)).pow(self.p))
        wloss = wloss0.mean()
        wact = (wloss0 > 0).sum() # number of active constraints on weights

        aloss0 = (torch.max(z, (laq - las) -
                 (self.at - self.l_eps)).pow(self.p))
        aloss = aloss0.mean()
        aact = (aloss0 > 0).sum() # number of active constraints on activations

        rloss = loss.pow_(self.p)

        calib_mul = self.loss_sum / self.cnt
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
