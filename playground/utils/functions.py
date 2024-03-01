import cv2
import base64

def read_image(image_path):
    original_image = cv2.imread(image_path)
    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    return image

def convert_image(output_image):
    is_success, buffer = cv2.imencode(".jpg", output_image)

    if is_success:
        # Convert the byte stream to a Base64 string
        image_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
        return image_base64
    else:
        raise ValueError("Failed to encode the image as Base64")
