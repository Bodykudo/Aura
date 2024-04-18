import cv2
import numpy as np

from api.utils import read_image


class Matching:
    @staticmethod
    def ssd_match(image_path: str, template_path: str):
        image = read_image(image_path)
        template = read_image(template_path, grayscale=True)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result_cols = gray.shape[1] - template.shape[1] + 1
        result_rows = gray.shape[0] - template.shape[0] + 1
        output = np.zeros((result_rows, result_cols), dtype=np.int32)

        for i in range(result_rows):
            for j in range(result_cols):
                roi = gray[i : i + template.shape[0], j : j + template.shape[1]]
                difference = roi.astype(np.float32) - template.astype(np.float32)
                result_image = np.square(difference)
                output[i, j] = np.sum(result_image)

        _, _, min_loc, _ = cv2.minMaxLoc(output)
        roi = (min_loc[0], min_loc[1], template.shape[1], template.shape[0])

        output_image = image.copy()
        cv2.rectangle(
            output_image,
            (roi[0], roi[1]),
            (roi[0] + roi[2], roi[1] + roi[3]),
            (0, 255, 0),
            2,
        )
        return output_image

    @staticmethod
    def cross_correlation_match(image_path: str, template_path: str):
        image = read_image(image_path)
        template = read_image(template_path, grayscale=True)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result_cols = gray.shape[1] - template.shape[1] + 1
        result_rows = gray.shape[0] - template.shape[0] + 1
        output = np.zeros((result_rows, result_cols), dtype=np.float32)

        template_mean = np.mean(template)
        template_std = np.std(template)

        for i in range(result_rows):
            for j in range(result_cols):
                roi = gray[i : i + template.shape[0], j : j + template.shape[1]]
                roi_mean = np.mean(roi)
                roi_std = np.std(roi)

                if roi_std > 0:
                    correlation = ((roi - roi_mean) * (template - template_mean)).sum()
                    output[i, j] = correlation / (
                        roi_std * template_std * np.prod(template.shape)
                    )

        _, _, _, max_loc = cv2.minMaxLoc(output)
        roi = (max_loc[0], max_loc[1], template.shape[1], template.shape[0])

        output_image = image.copy()
        cv2.rectangle(
            output_image,
            (roi[0], roi[1]),
            (roi[0] + roi[2], roi[1] + roi[3]),
            (0, 255, 0),
            2,
        )
        return output_image
