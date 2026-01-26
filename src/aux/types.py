from enum import Enum

class DType(Enum):
    VISION_CLS = 1
    VISION_SR = 2
    VISION_DNS = 3
    VISION_OD = 4

class MType(Enum):
    # Vision-based model types
    VISION_CLS = 1
    VISION_SR = 2
    VISION_DNS = 3
    VISION_OD = 4
    
    # Language-based model types
    LM = 10

class QScheme(Enum):
    PER_TENSOR = 0
    PER_CHANNEL = 1

class QMethod(Enum):
    GDNSQ = 0
    