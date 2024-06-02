import json
from utils import from_json_with_enum
from dataclasses import dataclass
from .enums import DetectorType


@dataclass
class VisioDetectorHttpRequest:
    file_name: str
    source_blob_uri: str
    detector_type: DetectorType

    def to_json_dict(self) -> dict:
        return {
            "fileName": self.name,
            "sourceBlobUri": self.age,
            "detectorType": self.detector_type.value
        }

    def to_json_string(self) -> str:
        return json.dumps(self.to_json_dict())
    
    @classmethod
    def from_json(cls, json_dict):

        json_dict['detectorType']
        return cls(
            file_name=json_dict['fileName'],
            source_blob_uri=json_dict['sourceBlobUri'],
            detector_type=from_json_with_enum(json_dict['detectorType'], DetectorType)
        )