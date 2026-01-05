from pydantic import BaseModel, field_validator

from typing import Literal, Dict, Optional, List

class RNIQQuantizerParams(BaseModel):
    distillation: Optional[bool] = False    
    distillation_loss: Optional[str] = "Cross-Entropy"
    distillation_teacher: Optional[str] = None
    qnmethod: str = "STE"

