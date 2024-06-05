import json
from enum import Enum
from dataclasses import dataclass
from typing import Optional
from utils import string_to_enum
from .enums import DetectorType, PredictionClass

@dataclass
class ImagePredictionResult:
    image_name: str
    detector_type: DetectorType
    prediction: float = 0
    classification: Optional[str] = None
    result_img_name: Optional[str] = None
    result_img_path: Optional[str] = None
    errors: str = None
    has_errors: bool = False
    time_taken: float = 0
    
    def __str__(self):
        return f"Image Name: {self.image_name}, Detector Type: {self.detector_type}, Prediction: {self.prediction}, Classification: {self.classification}, Result Image Name: {self.result_img_name}, Result Image Path: {self.result_img_path}, Errors: {self.errors}, Has Errors: {self.has_errors}, Time Taken: {self.time_taken}"

    def to_json_dict(self) -> dict:
        # Check if self.classification is None
        if self.classification is None or self.classification == "" or self.classification.lower() == "null":
            prediction_class_value = "UNKNOWN"
        else:
            # Use string_to_enum to convert self.classification to the corresponding enum value
            prediction_class_value = string_to_enum(PredictionClass, self.classification).value
        
        return {
            "imageName": self.image_name,
            "resultImgName": self.result_img_name,
            "resultImgPath": self.result_img_path,
            "detectorType": self.detector_type.value,
            "predictionClass": prediction_class_value,
            "prediction": self.prediction,
            "errors": self.errors,
            "hasErrors": self.has_errors,
            "timeTaken": self.time_taken
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_json_dict())