# https://gist.github.com/Sinet2000/feab2a06c9bc4f8f04d4ac5f33adbb0e

import logging
import os
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError, AzureError

class AzureBlobManager:
    def __init__(self, connection_string, container_name):
        self.connection_string = connection_string
        self.container_name = container_name
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_client = self.blob_service_client.get_container_client(container_name)

    def upload_file(self, file_path):
        blob_name = os.path.basename(file_path)
        blob_client = self.blob_service_client.get_blob_client(self.container_name, blob_name)

        if blob_client.exists():
            return f"The file '{blob_name}' already exists in the blob container."
        else:
            with open(file_path, "rb") as data:
                upload_stream = data.read()
                blob_client.upload_blob(upload_stream)
            return f"Successfully uploaded the file '{blob_name}' to the blob container."

    def list_blobs(self):
        blob_list = self.container_client.list_blobs()
        for blob in blob_list:
            print(blob.name)

    def delete_all_blobs(self):
        blob_list = self.container_client.list_blobs()
        for blob in blob_list:
            self.container_client.delete_blob(blob)

    def download_and_upload_file(self, file_name, upload_dir) -> str:
        blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=file_name)
        download_file_path = os.path.join(upload_dir, file_name)
        
        # Ensure the upload directory exists
        os.makedirs(upload_dir, exist_ok=True)
        with open(download_file_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
            logging.info(f"Successfully downloaded and uploaded the file '{file_name}'.")

        return os.path.join(upload_dir, file_name)
    
    def upload_file_to_blob(self, file_path, file_name) -> str:
        try:
            # Check if file path and file name are valid
            if not file_path:
                raise ValueError("File path is null or empty.")
            if not file_name:
                raise ValueError("File name is null or empty.")

            # Check if the file exists locally
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File '{file_path}' does not exist.")

            # Upload the file to blob container
            blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=file_name)
            with open(file_path, "rb") as file:
                blob_client.upload_blob(file)
                logging.info(f"File '{file_name}' uploaded to blob container '{self.container_name}' successfully.")
            return f"Uploaded file '{file_name}' to blob container '{self.container_name}'."
        except ValueError as ve:
            logging.error(f"ValueError: {ve}")
            raise  # Re-raise the exception
        except FileNotFoundError as fe:
            logging.error(f"FileNotFoundError: {fe}")
            raise  # Re-raise the exception
        except Exception as e:
            logging.error(f"Error uploading file to blob container: {e}")
            raise  # Re-raise the exception