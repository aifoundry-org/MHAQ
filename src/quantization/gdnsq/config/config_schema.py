from pydantic import BaseModel

from typing import Optional

class CurriculumConfig(BaseModel):
    enable: bool = True
    mean: float = 0.9
    std: float = 0.5


class GDNSQQuantizerParams(BaseModel):
    distillation: Optional[bool] = False    
    distillation_loss: Optional[str] = "Cross-Entropy"
    distillation_teacher: Optional[str] = None
    qnmethod: str = "STE"
    curriculum: CurriculumConfig = CurriculumConfig()
    # When True, Conv2d layers with kernel_size (1, 1) are left as nn.Conv2d (not NoisyConv2d).
    skip_1x1_conv: bool = True

