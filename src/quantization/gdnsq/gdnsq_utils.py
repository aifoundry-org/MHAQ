from enum import Enum

class QMode(Enum):
        NOISE_VAL = 1
        ROUND_VAL = 2
        SOURCE_VAL = 3
        FLOAT_TRAIN_VAL = 4

class QNMethod(Enum):
        STE = 0
        EWGS = 1 
        AEWGS = 2
        LSQ = 3

class GradNoiseType(Enum):
        BER3     = "BER3"    
        BER1     = "BER1"     
        NORM3    = "NORM3"    
        NORM1    = "NORM1"    
        UNIFORM  = "UNIFORM"  
        ROUNDING = "ROUNDING" 
