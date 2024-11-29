import numpy as np
import torch.nn as nn
import torch


class PotentialLoss(nn.Module):
    def __init__(self, criterion, alpha=(1, 1, 1),
                 step_size=10,
                 eps=0.02,
                 lmin=0,
                 p=1,
                 a=8,
                 w=4,
                 scale_momentum=0.9,
                 scale_coeff=1.1,
                 w_scale_m=1.0,
                 a_scale_m=1.0) -> None:
        super().__init__()
        self.alpha = torch.tensor(alpha)
        self.register_buffer("a_scale", torch.tensor(0))
        self.register_buffer("w_scale", torch.tensor(0))
        self.scale_momentum = scale_momentum
        self.criterion = criterion
        self.lmin = torch.log2(torch.tensor(lmin + 1))
        self.eps = torch.tensor(eps)
        self.s_weight_loss = torch.tensor(0)
        self.s_act_loss = torch.tensor(0)
        self.weight_reg_loss = torch.tensor(0)
        self.p = torch.tensor(p)
        self.at = torch.log2((2**torch.tensor(a)).sub(1.0))
        self.wt = torch.log2((2**torch.tensor(w-1)).sub(1.0))
        self.l_eps = torch.tensor(1e-3)
        self.r_eps = torch.tensor(1e-3)
        self.scale_coeff = scale_coeff
        self.rloss_sm = None
        self.wloss_sm = None
        self.aloss_sm = None
        self.w_scale = torch.tensor(1.0)
        self.a_scale = torch.tensor(1.0)
        self.r_scale = torch.tensor(1.0)

        self.raise_a_power = 0.9


    def forward(self, output, target):
        """Forward method to wrapping main loss

        Args:
            output (tuple[torch.tensor]): Output for main loss
            stated as (x ,log_act_s, log_act_q, log_wght_s, log_w)
            target (torch.tensor): ground truth to calculate loss for

        Returns:
            torch.tensor: Potential loss result value
        """
        out_0 = output[0]  # prediction
        out_1 = output[1]  # log_act_s
        out_2 = output[2]  # log_act_q
        out_3 = output[3]  # log_wght_s
        out_4 = output[4]  # log_w

        self.base_loss = self.criterion(out_0, target.softmax(-1))
        # self.base_loss = self.criterion(out_0.softmax(-1), target.softmax(-1))
        # self.base_loss = self.criterion(out_0, target)
        loss = self.base_loss

        z = torch.tensor(0)
        x = torch.max(z, loss - self.lmin * (1 + self.eps))


        wloss = (torch.max(z, (out_4 - out_3) -
                 (self.wt - self.l_eps)).pow(self.p)).mean()
        aloss = (torch.max(z, (out_2 - out_1) -
                 (self.at - self.l_eps)).pow(self.p)).mean()

        rloss = x.pow_(self.p)

        if self.training:

            if self.rloss_sm is not None:
                self.aloss_sm = self.scale_momentum * self.aloss_sm.detach() + \
                    (1.0 - self.scale_momentum) * aloss
                self.wloss_sm = self.scale_momentum * self.wloss_sm.detach() + \
                    (1.0 - self.scale_momentum) * wloss
                self.rloss_sm = self.scale_momentum * self.rloss_sm.detach() + \
                    (1.0 - self.scale_momentum) * rloss
            else:
                self.aloss_sm = aloss
                self.wloss_sm = wloss
                self.rloss_sm = rloss

            self.w_scale = (1.0 / (self.wloss_sm + self.r_eps)).detach()
            self.a_scale = (1.0 / (self.aloss_sm + self.r_eps)).detach()
            self.r_scale = (1.0 / (self.rloss_sm + self.r_eps)).detach()

        ploss = self.alpha[0] * wloss * self.w_scale.clone() + \
            self.alpha[1] * aloss * self.a_scale.clone() + \
            self.alpha[2] * rloss * self.r_scale.clone()


        self.wloss = wloss
        self.aloss = aloss
        self.rloss = rloss
        self.s_weight_loss = -out_3.mean()
        self.q_weight_loss = out_4.mean()
        self.s_act_loss = -out_1.mean()
        self.q_act_loss = out_2.mean()
        self.weight_reg_loss = (out_4-out_3).max() + 1


        return ploss
