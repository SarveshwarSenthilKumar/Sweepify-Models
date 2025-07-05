import requests
import base64
import os
from PIL import Image
import io
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def compress_image(img, max_size_mb=5):
    """Compress image to ensure it's under max_size_mb"""
    quality = 95
    max_size_bytes = max_size_mb * 1024 * 1024
    
    # First, resize the image if it's too large
    max_dimension = 1920  # Max width or height
    if img.width > max_dimension or img.height > max_dimension:
        # Calculate new dimensions maintaining aspect ratio
        if img.width > img.height:
            new_width = max_dimension
            new_height = int((img.height * max_dimension) / img.width)
        else:
            new_height = max_dimension
            new_width = int((img.width * max_dimension) / img.height)
        
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        print(f"Resized image to: {img.size}")
    
    while True:
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        
        current_size = buffer.tell()
        print(f"Image size: {current_size / (1024*1024):.2f}MB, Quality: {quality}")
        
        if current_size <= max_size_bytes or quality <= 5:
            buffer.seek(0)
            print(f"Final compressed size: {current_size / (1024*1024):.2f}MB")
            return buffer
        
        quality -= 10
        buffer.close()

def roboflow_infer(image_file):
    api_key = os.getenv('ROBOFLOW_API_KEY')
    if not api_key:
        raise ValueError('ROBOFLOW_API_KEY not set in environment')

    # Read the file from the beginning
    image_file.seek(0)
    
    # Get original file size for debugging
    image_file.seek(0, 2)  # Seek to end
    original_size = image_file.tell()
    image_file.seek(0)  # Reset to beginning
    print(f"Original file size: {original_size / (1024*1024):.2f}MB")
    
    # Load with PIL and convert to RGB
    img = Image.open(image_file).convert('RGB')
    print(f"Image dimensions: {img.size}")
    
    # Compress image if it's too large (max 5MB for Roboflow API)
    buffer = compress_image(img, max_size_mb=5)
    
    # Get the compressed image bytes
    img_bytes = buffer.read()
    
    url = "https://detect.roboflow.com/human-and-trash-detection-v1/1"
    params = {
        "api_key": api_key,
        "confidence": 0.20,
    }
    
    # Try multipart/form-data first (more reliable for large files)
    try:
        buffer.seek(0)
        files = {'file': ('image.jpg', buffer, 'image/jpeg')}
        response = requests.post(
            url,
            params=params,
            files=files
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Multipart upload failed: {e}")
        # Fallback: try base64 JSON upload
        try:
            img_base64 = base64.b64encode(img_bytes).decode('ascii').replace('\n', '')
            response = requests.post(
                url,
                params=params,
                json={"image": img_base64}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e2:
            print(f"Base64 upload also failed: {e2}")
            raise e2 