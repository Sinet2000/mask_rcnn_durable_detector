from .enums import BlobProcessStatus

class BlobToProcessQueueMessage:
    def __init__(self,
                 file_name: str,
                 partition_key: str,
                 row_key: str,
                 source_uri: str,
                 process_status: BlobProcessStatus):
        
        self.file_name = file_name
        self.partition_key = partition_key
        self.row_key = row_key
        self.source_uri = source_uri
        self.process_status = process_status

    def __str__(self):
        return f"FileName: {self.file_name}, PartitionKey: {self.partition_key}, RowKey: {self.row_key}, SourceUri: {self.source_uri}, ProcessStatus: {self.process_status}"