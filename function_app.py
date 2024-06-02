## conda activate py-39
## func start --port 7072

import json
import logging
import os

import azure.durable_functions as df
import azure.functions as func

from managers import AzureBlobManager, AzureTableStorageManager
from models import BlobToProcessQueueMessage, VisioDetectorHttpRequest, ImagePredictionResult
from models.enums import BlobProcessStatus, DetectorType
from mask_rcnn_predictor import MaskRCnnPredictor
from visio_detector import VisioDetector
from utils import get_child_directory_path

blob_container_name = os.environ.get("BlobContainerName")
blob_connection_string = os.environ.get("BlobConnectionString")
azure_blob_manager = AzureBlobManager(blob_connection_string, blob_container_name)

predictions_blob_container_name = os.environ.get("ProcessedBlobsContainerName")
predictions_azure_blob_manager = AzureBlobManager(blob_connection_string, predictions_blob_container_name)

table_storage_name = os.environ.get("TableStorageName")
table_connection_string = os.environ.get("TableConnectionString")
azure_table_storage_manager = AzureTableStorageManager(table_connection_string, table_storage_name)

source_img_dir = 'image_set/mask_rcnn'
result_img_dir = 'image_set_res/mask_rcnn'

def delete_file_if_exists(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"File '{file_path}' deleted successfully.")
    else:
        print(f"File '{file_path}' does not exist.")

# We can provide a key, and use function level: https://learn.microsoft.com/en-us/azure/azure-functions/functions-bindings-http-webhook-trigger?tabs=python-v2%2Cisolated-process%2Cnodejs-v4%2Cfunctionsv2&pivots=programming-language-python
app = df.DFApp(http_auth_level=func.AuthLevel.ANONYMOUS)

trained_model_dir = "./trained-model/content/pytorch-mask-r-cnn-instance-segmentation/2024-04-27_18-10-33/"
mask_rcnn_predictor = MaskRCnnPredictor(trained_model_dir)
mask_rcnn_predictor.load_colormap()
mask_rcnn_predictor.load_model_and_set_session()

@app.function_name(name="ObjectDetectionHttpTrigger")
@app.route(route="mask-rcnn/detect", methods=("POST",))
@app.durable_client_input(client_name="client")
async def http_start(req: func.HttpRequest, client: df.DurableOrchestrationClient):
    req_body = req.get_body().decode('utf-8')
    logging.info(f"Started ObjectDetectionHttpTrigger, received: {req_body}")

    instance_id = await client.start_new("image_detection_orchestrator", client_input=req_body)
    
    await client.wait_for_completion_or_create_check_status_response(req, instance_id)     

    # Get orchestration execution status
    status = await client.get_status(instance_id)     

    # Retrieves orchestration execution results and displays them on the screen
    runtime = status.runtime_status
    output = status.output
    logging.info(f"runtime: {runtime}\n\n output:{output}")

    return output

@app.orchestration_trigger(context_name="context")
def image_detection_orchestrator(context: df.DurableOrchestrationContext):
    visio_detector_req_str = context.get_input()
    visio_detector_json = json.loads(visio_detector_req_str)
    logging.info(f"image_detection_orchestrator.visio_detector_json: {visio_detector_json}")

    try:
        visio_detector_req = VisioDetectorHttpRequest.from_json(visio_detector_json)
        logging.info(f"image_detection_orchestrator.detector_type: {visio_detector_req.detector_type}")

        if visio_detector_req.detector_type == DetectorType.Mask_R_CNN:
            result = yield context.call_activity("run_maskrcnn_detection_activity", visio_detector_req_str)
        else:
            result = ImagePredictionResult(
                image_name=visio_detector_req.file_name,
                detector_type=visio_detector_req.detector_type,
                errors= "The detector type is incorrect, must be Mask RCNN",
                has_errors=True
                ).to_json()

        logging.info(f"image_detection_orchestrator.result: {result}")
        return result
    except Exception as ex:
        logging.error(f"An unexpected error occurred: {ex}")
        return ImagePredictionResult(
            image_name=visio_detector_json['fileName'],
            detector_type=DetectorType.UNKNOWN,
            errors= str(ex),
            has_errors=True
            ).to_json()

@app.activity_trigger(input_name="visioDetectorReqStr")
def run_maskrcnn_detection_activity(visioDetectorReqStr: str) -> str:
    visioDetectorModel = VisioDetectorHttpRequest.from_json(json.loads(visioDetectorReqStr))

    logging.info(f"Running Mask RCNN detection for: {visioDetectorModel.file_name}")

    try:
        logging.info(f"Detector Type: {visioDetectorModel.detector_type}")

        file_download_dir = get_child_directory_path(source_img_dir)
        source_img_path = azure_blob_manager.download_and_upload_file(visioDetectorModel.file_name, file_download_dir)

        detector_type = DetectorType.Mask_R_CNN
        maskrcnn_precition_result = mask_rcnn_predictor.process_image_and_get_predictions(source_img_path, get_child_directory_path(result_img_dir))

        # yolov5_precition_result = VisioDetector.run_yolov_detector_wrapper(visioDetectorModel.file_name, downloaded_file_path)

        # predictions_azure_blob_manager.upload_file_to_blob(yolov5_precition_result.result_img_path, yolov5_precition_result.result_img_name)
        if maskrcnn_precition_result.error_message:
            return ImagePredictionResult(
                file_name=visioDetectorModel.file_name,
                detector_type=detector_type,
                errors=maskrcnn_precition_result.error_message,
                has_errors=True
            ).to_json()

        image_prediction_response = ImagePredictionResult(
            image_name=visioDetectorModel.file_name,
            detector_type=detector_type,
            classification = maskrcnn_precition_result.label,
            result_img_name=visioDetectorModel.file_name,
            result_img_path=maskrcnn_precition_result.det_img_path,
            prediction=float(maskrcnn_precition_result.value),
            time_taken=0.5)
        
        predictions_azure_blob_manager.upload_file_to_blob(maskrcnn_precition_result.det_img_path, maskrcnn_precition_result.det_img_filename)
        
        logging.info(f"run_yolov_detection_activity.Result: {image_prediction_response}")

        # delete_file_if_exists(downloaded_file_path)
        return image_prediction_response.to_json()

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return ImagePredictionResult(
            image_name=visioDetectorModel.file_name,
            detector_type=visioDetectorModel.detector_type,
            errors= str(e),
            has_errors=True
            ).to_json()