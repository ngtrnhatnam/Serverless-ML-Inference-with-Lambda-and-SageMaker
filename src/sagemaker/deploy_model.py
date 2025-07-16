import boto3    # SDK chính để giao tiếp với AWS.
import sagemaker    # SDK cao cấp gói sẵn thao tác ML.
from sagemaker.pytorch import PyTorchModel    # Lớp helper cho model PyTorch.
import time

# Thiết lập phiên làm việc và role
sagemaker_session = sagemaker.Session()
role = "arn:aws:iam::961880115777:role/service-role/AmazonSageMaker-ExecutionRole-20250716T150326"

# Chuẩn bị đường dẫn model trong S3
bucket = sagemaker_session.default_bucket()
model_key = "resnet18-dogcat/model.tar.gz"
model_uri = f"s3://{bucket}/{model_key}"

s3 = boto3.client("s3")
try:
    s3.head_object(Bucket=bucket, Key=model_key)
    print("Model đã có trong S3, không cần upload lại.")
except:
    s3.upload_file("assets/model/model.tar.gz", bucket, model_key)
    print("Uploaded model.tar.gz lên S3.")

model = PyTorchModel(
    model_data=model_uri,
    role=role,
    entry_point="assets/model/inference.py",  
    framework_version="1.12.1",
    py_version="py38"
)

timestamp = int(time.time())
endpoint_name = f"dogcat-endpoint-{timestamp}"

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",  # Có thể dùng ml.t2.medium để tiết kiệm
    endpoint_name=endpoint_name
)

print(f"\nEndpoint đã deploy: {endpoint_name}")