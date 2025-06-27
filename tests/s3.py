import boto3
from botocore.exceptions import ClientError

s3 = boto3.client('s3')

try:
    # Try head_bucket (doesn’t need ListBucket, just checks access)
    s3.head_bucket(Bucket='cloneifyai')
    print("You have access to the bucket.")
except ClientError as e:
    print(f"Access issue: {e.response['Error']['Code']} - {e.response['Error']['Message']}")
