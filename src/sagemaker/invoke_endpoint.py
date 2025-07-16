import boto3
import json
import os
import time

start = time.time()
def invoke_image(endpoint_name: str, image_path: str):
    # Tạo client để gọi endpoint SageMaker
    runtime = boto3.client('sagemaker-runtime')

    # Đọc file ảnh cần gửi lên
    with open(image_path, "rb") as f:
        payload = f.read()

    # Gửi request tới endpoint
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/x-image",
        Body=payload
    )
    
    # Đọc kết quả
    result = response['Body'].read().decode()
    return result


if __name__ == "__main__":
    # Nhớ sửa lại tên endpoint đúng cái đã deploy
    endpoint_name = "dogcat-endpoint-1752654024"  # đổi lại!
    image_path = "assets/images/test.jpeg"         # ảnh chó/mèo test
    
    result = invoke_image(endpoint_name, image_path)
    print(f"Dự đoán: {result}")
    end = time.time()
    print(f"Time taken for inference: {end - start:.2f} seconds")