# It must be executed using conda env, that contains configured packages from Setup.ipynb (used python 3.9)

# Import Python Standard Library dependencies
import json
from pathlib import Path
import random
from decimal import Decimal

# Import utility functions
from cjm_psl_utils.core import download_file
from cjm_pil_utils.core import resize_img
from utils import draw_masks_pil, draw_bboxes_pil, save_detection_result
from models import MaskRCnnDetectorResult
from models.enums import DetectorType

# Import numpy
import numpy as np

# Import the pandas package
import pandas as pd

# Import PIL for image manipulation
from PIL import Image, ImageDraw, ImageFont

# Import PyTorch dependencies
import torch

# Import Mask R-CNN
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# Import ONNX dependencies
import onnx # Import the onnx module
from onnxsim import simplify # Import the method to simplify ONNX models
import onnxruntime as ort # Import the ONNX Runtime

# /home/midtempo/Mask-RCNN/trained-model/content/pytorch-mask-r-cnn-instance-segmentation
# 2024-04-27_18-10-33
class MaskRCnnPredictor:
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.font_file = 'KFOlCnqEu92Fr1MmEU9vAw.ttf'

        # Check if the model directory exists and is not empty
        if not self.model_dir.exists() or not any(self.model_dir.iterdir()):
            raise FileNotFoundError("Model directory does not exist or is empty.")

        # Check if the font file exists, if not, download it
        font_path = Path(self.font_file)
        if not font_path.exists():
            download_file(f"https://fonts.gstatic.com/s/roboto/v30/{font_file}", "./")

    def load_colormap(self):
        # The colormap path
        self.colormap_path = list(self.model_dir.glob('*colormap.json'))[0]

        # Load the JSON colormap data
        with open(self.colormap_path, 'r') as file:
            colormap_json = json.load(file)

        # Convert the JSON data to a dictionary        
        colormap_dict = {item['label']: item['color'] for item in colormap_json['items']}

        # Extract the class names from the colormap
        self.class_names = list(colormap_dict.keys())

        # Make a copy of the colormap in integer format
        self.int_colors = [tuple(int(c*255) for c in color) for color in colormap_dict.values()]
    
    def load_model_and_set_session(self):
        # The model checkpoint path
        checkpoint_path = list(self.model_dir.glob('*.pth'))[0]

        # Load the model checkpoint onto the CPU
        model_checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Initialize a Mask R-CNN model without loading any weights
        model = maskrcnn_resnet50_fpn_v2(pretrained=False)

        # Get the number of input features for the classifier
        in_features_box = model.roi_heads.box_predictor.cls_score.in_features
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

        # Replace the box predictor
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, len(self.class_names))

        # Replace the mask predictor
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels=in_features_mask, dim_reduced=256, num_classes=len(self.class_names))

            # Initialize the model with the checkpoint parameters and buffers
        model.load_state_dict(model_checkpoint)

        ## Exporting the Model to ONNX
        model.eval()

        input_tensor = self.__prepare_input_tensor()

        # Set a filename for the ONNX model
        onnx_file_path = f"{self.model_dir}/{self.colormap_path.stem.removesuffix('-colormap')}-{checkpoint_path.stem}.onnx"

        # Export the PyTorch model to ONNX format
        torch.onnx.export(model.cpu(),
                        input_tensor.cpu(),
                        onnx_file_path,
                        export_params=True,
                        do_constant_folding=False,
                        input_names = ['input'],
                        output_names = ['boxes', 'labels', 'scores', 'masks'],
                        dynamic_axes={'input': {2 : 'height', 3 : 'width'}}
                        )
        
        # Load the ONNX model from the onnx_file_name
        onnx_model = onnx.load(onnx_file_path)

        # Simplify the model
        model_simp, check = simplify(onnx_model)

        # Save the simplified model to the onnx_file_name
        onnx.save(model_simp, onnx_file_path)


        ### Create an Inference Session
        self.session = ort.InferenceSession(onnx_file_path, providers=['CPUExecutionProvider'])

    def __prepare_input_tensor(self, input_size=(256, 256)):
        # Prepare the Input Tensor
        input_tensor = torch.randn(1, 3, *input_size)
        return input_tensor
    
    def process_image_and_get_predictions(self, img_path, result_img_dir) -> MaskRCnnDetectorResult:
        try:
            # Check if the session object is set
            if not hasattr(self, 'session') or self.session is None:
                raise ValueError("Session object is not set.")

            # Check if the image exists
            img_path = Path(img_path)
            if not img_path.exists():
                raise FileNotFoundError("Image file not found.")

            # Load the test image
            test_img = Image.open(img_path)

            # Set test image size
            test_sz = 512

            ## Resize the test image
            input_img = resize_img(test_img, target_sz=test_sz, divisor=1)

            # Calculate the scale between the source image and the resized image
            min_img_scale = min(test_img.size) / min(input_img.size)

            ## Prepare the Test Image
            # Convert the input image to NumPy format
            input_tensor_np = np.array(input_img, dtype=np.float32).transpose((2, 0, 1))[None]/255
            # -------------------------------

            # Run inference
            model_output = self.session.run(None, {"input": input_tensor_np})

            # Set the confidence threshold
            threshold = 0.45

            # Filter the output based on the confidence threshold
            scores_mask = model_output[2] > threshold

            bbox_list = (model_output[0][scores_mask])*min_img_scale
            label_list = [self.class_names[int(idx)] for idx in model_output[1][scores_mask]]
            probs_list = model_output[2]

            colors = [self.int_colors[self.class_names.index(i)] for i in label_list]

            annotated_img = draw_masks_pil(input_img, model_output[-1], label_list, colors, alpha=0.3)
            annotated_img = annotated_img.resize(test_img.size)

            annotated_img = draw_bboxes_pil(
                image=annotated_img, 
                boxes=bbox_list, 
                labels=label_list,
                probs=probs_list,
                colors=colors, 
                font=self.font_file,
            )

            det_img_filename, det_img_path = save_detection_result(annotated_img, result_img_dir, img_path.name, DetectorType.Mask_R_CNN)
            # display(annotated_img)

            # Print the prediction data as a Pandas Series for easy formatting
            prediction_result_with_bboxes = pd.Series({
                "Predicted BBoxes:": [f"{label}:{bbox}" for label, bbox in zip(label_list, bbox_list.round(decimals=3))],
                "Confidence Scores:": [f"{label}: {prob}" for label, prob in zip(label_list, probs_list)]
            }).to_frame().style.hide(axis='columns')

            predictions = prediction_result_with_bboxes.data.iloc[1, 0][0].strip("[]")

            label, value = predictions.split(":")
            label = label.strip()
            value = Decimal(value.strip())

            return MaskRCnnDetectorResult(label=label, value=value, error_message="", det_img_filename=det_img_filename, det_img_path=det_img_path)

        except Exception as e:
            return MaskRCnnDetectorResult(error_message=str(e))