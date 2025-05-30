import logging
import boto3
from botocore.exceptions import ClientError
from typing import Optional, Dict, Any
import traceback

# Set up logging
logger = logging.getLogger("s3_storage")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(handler)

class S3Storage:
    """
    Service for storing and retrieving files from AWS S3
    """
    def __init__(
        self, 
        aws_access_key: str, 
        aws_secret_key: str, 
        bucket_name: str, 
        region: str = "ap-south-1"
    ):
        """
        Initialize S3 client and set bucket name
        
        Args:
            aws_access_key: AWS access key ID
            aws_secret_key: AWS secret access key
            bucket_name: S3 bucket name
            region: AWS region (default: ap-south-1)
        """
        self.aws_access_key = aws_access_key
        self.aws_secret_key = aws_secret_key
        self.bucket_name = bucket_name
        self.region = region
        
        logger.info(f"Initializing S3 storage with bucket: {bucket_name}, region: {region}")
        
        try:
            # Initialize S3 client
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=region
            )
            
            # Initialize S3 resource for higher-level operations
            self.s3_resource = boto3.resource(
                's3',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=region
            )
            
            # Verify bucket exists
            try:
                self.s3_client.head_bucket(Bucket=bucket_name)
                logger.info(f"Successfully connected to S3 bucket: {bucket_name}")
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    logger.warning(f"Bucket {bucket_name} not found, attempting to create it")
                    self._create_bucket()
                else:
                    raise
                    
        except Exception as e:
            logger.error(f"Error initializing S3 client: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _create_bucket(self):
        """
        Create S3 bucket if it doesn't exist
        """
        try:
            if self.region == "us-east-1":
                # us-east-1 is the default region and requires different syntax
                self.s3_client.create_bucket(Bucket=self.bucket_name)
            else:
                # Other regions need the LocationConstraint parameter
                self.s3_client.create_bucket(
                    Bucket=self.bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
                )
            logger.info(f"Created new S3 bucket: {self.bucket_name}")
        except Exception as e:
            logger.error(f"Error creating S3 bucket: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    async def upload_file(
        self, 
        file_content: bytes, 
        file_path: str, 
        content_type: str = "application/octet-stream"
    ) -> Dict[str, Any]:
        """
        Upload a file to S3
        
        Args:
            file_content: Binary content of the file
            file_path: Path to store the file in S3 (including filename)
            content_type: MIME type of the file
        
        Returns:
            Dictionary with upload details including S3 URL
        """
        try:
            logger.info(f"Uploading file to S3: {file_path} ({len(file_content)} bytes)")
            
            # Upload file to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=file_path,
                Body=file_content,
                ContentType=content_type
            )
            
            # Generate file URL
            file_url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{file_path}"
            
            # Generate pre-signed URL (temporary access URL)
            presigned_url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': file_path},
                ExpiresIn=3600  # URL valid for 1 hour
            )
            
            logger.info(f"Successfully uploaded file to S3: {file_path}")
            
            return {
                "success": True,
                "file_path": file_path,
                "bucket": self.bucket_name,
                "region": self.region,
                "file_url": file_url,
                "presigned_url": presigned_url,
                "size": len(file_content)
            }
        except Exception as e:
            logger.error(f"Error uploading file to S3: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    async def download_file(self, file_path: str) -> bytes:
        """
        Download a file from S3
        
        Args:
            file_path: Path to the file in S3
            
        Returns:
            Binary content of the file
        """
        try:
            logger.info(f"Downloading file from S3: {file_path}")
            
            # Create a new S3 object
            s3_object = self.s3_resource.Object(self.bucket_name, file_path)
            
            # Download file from S3
            response = s3_object.get()
            file_content = response['Body'].read()
            
            logger.info(f"Successfully downloaded file from S3: {file_path} ({len(file_content)} bytes)")
            return file_content
        except Exception as e:
            logger.error(f"Error downloading file from S3: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    async def get_file_url(self, file_path: str, expiry: int = 3600) -> str:
        """
        Generate a pre-signed URL for a file in S3
        
        Args:
            file_path: Path to the file in S3
            expiry: URL expiry time in seconds (default: 1 hour)
            
        Returns:
            Pre-signed URL for the file
        """
        try:
            logger.info(f"Generating pre-signed URL for: {file_path}")
            
            # Generate pre-signed URL
            presigned_url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': file_path},
                ExpiresIn=expiry
            )
            
            logger.info(f"Generated pre-signed URL for: {file_path}")
            return presigned_url
        except Exception as e:
            logger.error(f"Error generating pre-signed URL: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    async def delete_file(self, file_path: str) -> bool:
        """
        Delete a file from S3
        
        Args:
            file_path: Path to the file in S3
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Deleting file from S3: {file_path}")
            
            # Delete file from S3
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=file_path
            )
            
            logger.info(f"Successfully deleted file from S3: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error deleting file from S3: {str(e)}")
            logger.error(traceback.format_exc())
            return False
            
    def get_direct_url(self, file_path: str) -> str:
        """
        Get the direct URL for a file in S3 (not pre-signed)
        
        Args:
            file_path: Path to the file in S3
            
        Returns:
            Direct URL to the file
        """
        return f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{file_path}"