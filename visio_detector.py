import os

from mask_rcnn_predictor import MaskRCnnPredictor
from models import ImagePredictionResult


class VisioDetector:
    @staticmethod
    def run_yolov_detector_wrapper(file_path, source_path) -> ImagePredictionResult:
        script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov5')
        prediction_result_dir = os.path.join(script_dir, "runs/detect", "yolo_road_det")
        csv_path = os.path.join(prediction_result_dir, "predictions.csv")
        result_dir_name = "yolo_road_det"

        return run_yolov_detector(file_name, source_path, script_dir, csv_path, result_dir_name, prediction_result_dir)