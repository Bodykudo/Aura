import cv2
import base64
from matplotlib import pyplot as plt

def read_image(image_path):
    original_image = cv2.imread(image_path)
    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    return image

def convert_image(output_image):
    plt.subplot(1, 3, 1)
    plt.imshow(output_image)
    plt.title('Median Filter')
    plt.show()

    is_success, buffer = cv2.imencode(".jpg", output_image)

    if is_success:
        # Convert the byte stream to a Base64 string
        base64_image = base64.b64encode(buffer).decode('utf-8')
        return base64_image
    else:
        raise ValueError("Failed to encode the image as Base64")
