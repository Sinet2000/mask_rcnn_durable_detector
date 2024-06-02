import json
from azure.data.tables import TableServiceClient
from azure.core.exceptions import ResourceNotFoundError
from enum import Enum
from typing import Optional
from models.blob_to_process_queue_message import BlobProcessStatus

class ProcessedBlobModel:
    def __init__(self,
                 source_uri: str,
                 file_name: str = None,
                 process_status: Optional[BlobProcessStatus] = None):
        
        self.source_uri = source_uri
        self.file_name = file_name
        self.process_status = process_status

class TableBlobModel:
    def __init__(self,
                 partition_key: str,
                 row_key: str,
                 blob_data_json: str):
        
        self.partition_key = partition_key
        self.row_key = row_key
        self.blob_data_json = blob_data_json

    def to_processed_blob_model(self) -> Optional[ProcessedBlobModel]:
        try:
            blob_data = json.loads(self.blob_data_json)
            process_status_str = blob_data.get('ProcessStatus')
            process_status = BlobProcessStatus[process_status_str] if process_status_str else None
            return ProcessedBlobModel(
                source_uri=blob_data.get('SourceUri'),
                file_name=blob_data.get('FileName'),
                process_status=process_status
            )
        except (json.JSONDecodeError, KeyError):
            # Handle invalid JSON data or missing keys
            return None


class AzureTableStorageManager:
    def __init__(self, connection_string, table_name):
        self.table_service_client = TableServiceClient.from_connection_string(conn_str=connection_string)
        self.table_client = self.table_service_client.get_table_client(table_name=table_name)

    def get_blob_data(self, partition_key, row_key):
        try:
            entity = self.table_client.get_entity(partition_key=partition_key, row_key=row_key)
            return TableBlobModel(partition_key=entity['PartitionKey'],
                                  row_key=entity['RowKey'],
                                  blob_data_json=entity['BlobDataJson'])
        
        except ResourceNotFoundError:
            print(f"The entity with PartitionKey: {partition_key} and RowKey: {row_key} does not exist.")
            return None