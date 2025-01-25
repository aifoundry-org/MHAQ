import numpy as np
import torch.nn as nn
import torch


class PotentialLoss(nn.Module):
    def __init__(self, criterion, alpha=(1, 1, 1),
                 step_size=10,
                 eps=0,
                 lmin=0,
                 p=1,
                 a=8,
                 w=4,
                 scale_momentum=0.99,
                 scale_coeff=1.1,
                 w_scale_m=1.0,
                 a_scale_m=1.0) -> None:
        super().__init__()
        self.alpha = torch.tensor(alpha)
        self.scale_momentum = scale_momentum
        self.criterion = criterion
        self.lmin = torch.log2(torch.tensor(lmin + 1))
        self.eps = torch.tensor(eps)
        self.s_weight_loss = torch.tensor(0)
        self.s_act_loss = torch.tensor(0)
        self.weight_reg_loss = torch.tensor(0)
        self.p = torch.tensor(p)
        self.at = a 
        self.wt = w 
        self.l_eps = torch.tensor(1e-3)
        self.r_eps = torch.tensor(1e-3)
        self.scale_coeff = scale_coeff
        self.aloss = torch.tensor(1.0)
        self.wloss = torch.tensor(1.0)

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
        out_0 = output[0]  # prediction
        out_1 = output[1]  # log_act_s
        out_2 = output[2]  # log_act_q
        out_3 = output[3]  # log_wght_s
        out_4 = output[4]  # log_w

        # self.base_loss = self.criterion(out_0, target.softmax(-1))
        # self.base_loss = self.criterion(out_0.softmax(-1), target.softmax(-1))
        self.base_loss = self.criterion(out_0, target)
        loss = self.base_loss

        z = torch.tensor(0)
        x = torch.max(z, loss - self.lmin * (1 + self.eps))


        wloss = (torch.max(z, (out_4 - out_3) -
                 (self.wt - self.l_eps)).pow(self.p)).mean()
        aloss = (torch.max(z, (out_2 - out_1) -
                 (self.at - self.l_eps)).pow(self.p)).mean()

        rloss = x.pow_(self.p)

        ploss = self.t * (self.alpha[0] * wloss + self.alpha[1] * aloss) + self.alpha[2] * rloss


        self.wloss = wloss - self.l_eps
        self.aloss = aloss - self.l_eps
        self.rloss = rloss
        self.s_weight_loss = -out_3.mean()
        self.q_weight_loss = out_4.mean()
        self.s_act_loss = -out_1.mean()
        self.q_act_loss = out_2.mean()
        self.weight_reg_loss = (out_4-out_3).max()


        return ploss
