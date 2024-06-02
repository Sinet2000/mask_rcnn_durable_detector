from dataclasses import dataclass
from decimal import Decimal

@dataclass
class MaskRCnnDetectorResult:
    label: str = "UNKNOWN"
    value: Decimal = Decimal(0.0)
    error_message: str = ""
    det_img_filename: str = ""
    det_img_path: str = ""