import uuid
import os

from PIL import Image
from pathlib import Path
from models.enums import DetectorType

def img_rename_to_detection_result(file_path, detector_type: DetectorType) -> tuple:
    """
    Renames a file with a prefix and returns the new filename and path.

    Parameters:
    - file_path (str): The path to the file.
    - prefix (str): The prefix to be added to the filename.

    Returns:
    - tuple: The new filename and path if successful, otherwise None.
    """
    try:
        if os.path.exists(file_path):
            # Split the file path into directory and filename
            directory, filename = os.path.split(file_path)

            # Get the image extension
            img_ext = os.path.splitext(filename)[1].lower()

            valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
            if img_ext not in valid_extensions:
                raise ValueError("Invalid image format. Only JPG, JPEG, PNG, GIF, BMP, and TIFF formats are supported.")

            prefix = build_img_prefix_for_detector(detector_type)
            # Remove the previous extension from the source image name
            source_img_name_without_ext = os.path.splitext(filename)[0]

            # Generate a unique identifier (GUID)
            unique_id = uuid.uuid4()

            # Construct the new filename
            new_filename = f"{prefix}{source_img_name_without_ext}_{unique_id}{img_ext}"

            # Construct the new file path
            new_file_path = os.path.join(directory, new_filename)
            # Rename the file
            os.rename(file_path, new_file_path)
            print(f"File '{file_path}' renamed to '{new_file_path}'.")

            return new_filename, new_file_path
        else:
            print(f"File '{file_path}' does not exist.")
            return None
    except OSError as e:
        print(f"Error renaming file: {e}")
        return None
    
def save_detection_result(image: Image.Image, directory: str, source_img_name: str, detector_type: DetectorType) -> tuple:
    """
    Saves a PIL image to a specified directory with the given filename.
    Ensures the directory exists and handles exceptions.

    Parameters:
    - image (PIL.Image.Image): The image to be saved.
    - directory (str): The directory where the image will be saved.
    - source_img_name (str): The name of the source file.
    - detector_type (DetectorType): The type of detector

    Returns:
    - str: The path to the saved image.
    """
    try:
        # Ensure the directory exists
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        # Check if the source image has a valid image extension
        valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
        img_ext = os.path.splitext(source_img_name)[1].lower()
        if img_ext not in valid_extensions:
            raise ValueError("Invalid image format. Only JPG, JPEG, PNG, GIF, BMP, and TIFF formats are supported.")

        # Build the image prefix based on detector type
        prefix = build_img_prefix_for_detector(detector_type)

        # Remove the previous extension from the source image name
        source_img_name_without_ext = os.path.splitext(source_img_name)[0]

        # Generate a unique identifier (GUID)
        unique_id = uuid.uuid4()

        # Construct the new filename with the original extension at the end
        new_filename = f"{prefix}{source_img_name_without_ext}_{unique_id}{img_ext}"

        # Full path for the image
        image_path = path / new_filename

        # Save the image
        image.save(image_path)
        print(f"Image saved successfully at: {image_path}")

        return new_filename, str(image_path)
    except Exception as e:
        print(f"An error occurred while saving the image: {e}")
        return None, None
    
def save_img_to_directory(image: Image.Image, directory: str, source_img_name: str) -> str:
    """
    Saves a PIL image to a specified directory with the given filename.
    Ensures the directory exists and handles exceptions.

    Parameters:
    - image (PIL.Image.Image): The image to be saved.
    - directory (str): The directory where the image will be saved.
    - source_img_name (str): The name of the source file.

    Returns:
    - str: The path to the saved image.
    """
    try:
        # Ensure the directory exists
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        # Check if the source image has a valid image extension
        valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
        img_ext = os.path.splitext(source_img_name)[1].lower()
        if img_ext not in valid_extensions:
            raise ValueError("Invalid image format. Only JPG, JPEG, PNG, GIF, BMP, and TIFF formats are supported.")

        # Full path for the image
        image_path = path / source_img_name

        # Save the image
        image.save(image_path)
        print(f"Image saved successfully at: {image_path}")

        return source_img_name, str(image_path)
    except Exception as e:
        print(f"An error occurred while saving the image: {e}")
        return None, None
     
def build_img_prefix_for_detector(detector_type: DetectorType) -> str:
    """
    Returns the prefix based on the detector type.

    Parameters:
    - detector_type (DetectorType): The type of detector.

    Returns:
    - str: The corresponding prefix.
    """
    if detector_type == DetectorType.YoloV5:
        return "yolov5_"
    elif detector_type == DetectorType.SSD:
        return "ssd_"
    elif detector_type == DetectorType.Mask_R_CNN:
        return "mask-rcnn_"
    else:
        return "nan_"
    
def get_child_directory_path(child_directory: str) -> str:
    """
    Constructs the full path to a child directory within the current working directory.
    
    Parameters:
    - child_directory (str): The name of the child directory.
    
    Returns:
    - str: The full path to the child directory.
    """
    # Get the current working directory
    current_directory = Path.cwd()
    
    # Construct the full path to the child directory
    child_directory_path = current_directory / child_directory
    
    # Ensure the child directory exists, create if not
    child_directory_path.mkdir(parents=True, exist_ok=True)
    
    return str(child_directory_path)
