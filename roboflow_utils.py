import requests
import base64
import os
from PIL import Image
import io

def roboflow_infer(image_file):
    api_key = os.environ.get('ROBOFLOW_API_KEY')
    if not api_key:
        raise ValueError('ROBOFLOW_API_KEY not set in environment')

    # Read the file from the beginning
    image_file.seek(0)
    # Load with PIL and convert to RGB
    img = Image.open(image_file).convert('RGB')
    # Save as JPEG to a BytesIO buffer
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=95)
    buffer.seek(0)
    # Read the buffer and encode as base64 (no newlines, no padding)
    img_bytes = buffer.read()
    img_base64 = base64.b64encode(img_bytes).decode('ascii').replace('\n', '')

    url = "https://detect.roboflow.com/human-and-trash-detection-v1/1"
    params = {
        "api_key": api_key,
        "confidence": 0.03,
    }
    # First try base64 JSON upload
    response = requests.post(
        url,
        params=params,
        json={"image": img_base64}
    )

    if response.status_code == 400:
        # Fallback: send as multipart/form-data
        buffer.seek(0)
        files = {'file': ('image.jpg', buffer, 'image/jpeg')}
        response = requests.post(
            url,
            params=params,
            files=files
        )
    response.raise_for_status()
    return response.json() 