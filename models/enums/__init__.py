from enum import Enum

class PredictionClass(Enum):
    basketball = 0
    redball = 1
    UNKNOWN = 99

class BlobProcessStatus(Enum):
    ReadyToProcess = 0
    InProgress = 1
    Processed = 2
    ErrorOccurred = 3

class DetectorType(Enum):
    YoloV5 = 0
    SSD = 1
    Mask_R_CNN = 2
    UNKNOWN = 3
