import boto3
import base64
import json

runtime = boto3.client("sagemaker-runtime")

ENDPOINT_NAME = "dogcat-endpoint-1752643983"  # Nhớ đổi lại nếu endpoint đổi tên

def lambda_handler(event, context):
    try:
        # Ảnh được gửi base64 encoded từ client
        body = event.get("body")
        if isinstance(body, str):
            body = json.loads(body)

        image_data = base64.b64decode(body["image"])

        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/x-image",
            Body=image_data
        )

        result = response["Body"].read().decode("utf-8")

        return {
            "statusCode": 200,
            "body": json.dumps({"prediction": result})
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }