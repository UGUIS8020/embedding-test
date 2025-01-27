import boto3
from dotenv import load_dotenv
import os

load_dotenv()

def create_s3_client():
    """
    Creates and returns an S3 client with explicit configuration
    """
    return boto3.client('s3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION")
    )

def upload_folder_to_s3(folder_path, bucket_name, s3_folder=None):
    """
    Uploads a folder and its contents to an S3 bucket
    :param folder_path: Local folder to upload
    :param bucket_name: Bucket to upload to
    :param s3_folder: S3 folder name. If not specified, the local folder name is used
    """
    # Create S3 client with explicit configuration
    s3_client = create_s3_client()

    # If S3 folder is not specified, use local folder name
    if s3_folder is None:
        s3_folder = os.path.basename(os.path.normpath(folder_path))

    # Walk through the folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            
            # Calculate relative path
            relative_path = os.path.relpath(local_file_path, folder_path)
            s3_file_path = os.path.join(s3_folder, relative_path).replace("\\", "/")

            try:
                # Upload the file
                s3_client.upload_file(local_file_path, bucket_name, s3_file_path)
                print(f"Uploaded {local_file_path} to {bucket_name}/{s3_file_path}")
            except Exception as e:
                print(f"Error uploading {local_file_path}: {str(e)}")

# Example usage
if __name__ == "__main__":
    folder_to_upload = "./chroma_db"
    bucket_name = "uguis-chroma"
    s3_folder = "tooth_transplant/"  # Optional: specify the S3 folder

    upload_folder_to_s3(folder_to_upload, bucket_name, s3_folder)