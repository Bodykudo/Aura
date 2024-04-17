import cv2
import numpy as np
import math

class FeatureMatching:

    @staticmethod
    def Normalized_Cross_Correlation(image_features, template_features):
        image_mean = sum(image_features) / len(image_features)
        template_mean = sum(template_features) / len(template_features)
        cross_correlation_sum = sum((img_val - image_mean) * (temp_val - template_mean)
                                    for img_val, temp_val in zip(image_features, template_features))

        image_variance_sum = sum((img_val - image_mean) ** 2 for img_val in image_features)
        template_variance_sum = sum((temp_val - template_mean) ** 2 for temp_val in template_features)

        if image_variance_sum == 0.0 or template_variance_sum == 0.0:
            return 0.0
        normalized_cross_correlation = cross_correlation_sum / math.sqrt(image_variance_sum * template_variance_sum)

        return normalized_cross_correlation
    @staticmethod
    def calculate_ssd(img1, img2):
    # Assume img1 is ROI from the original image and img2 is the template.
    # Both must have the same size, but this is naturally enforced by how they are passed to this function,
    # not by asserting their sizes globally.
        diff = img1.astype(np.float32) - img2.astype(np.float32)
        ssd = np.sum(diff**2)
        return ssd
    
    @staticmethod
    # Calculate similarity between original image and template at a given position
    def calculate_similarity(original, template, x, y):
        roi = original[y:y + template.shape[0], x:x + template.shape[1]]
        similarity_score = FeatureMatching.calculate_ssd(roi, template)
        return similarity_score
    
    @staticmethod
    def find_template(original, template):
        rows, cols = original.shape[0] - template.shape[0] + 1, original.shape[1] - template.shape[1] + 1
        best_match_x, best_match_y = 0, 0
        max_similarity_score = float('inf')  # We are looking for minimum SSD

        for y in range(rows):
            for x in range(cols):
                roi = original[y:y + template.shape[0], x:x + template.shape[1]]
                similarity_score = FeatureMatching.calculate_ssd(roi, template)
                if similarity_score < max_similarity_score:
                    max_similarity_score = similarity_score
                    best_match_x = x
                    best_match_y = y

        # Draw rectangle around matched region
        cv2.rectangle(original, (best_match_x, best_match_y),
                    (best_match_x + template.shape[1], best_match_y + template.shape[0]),
                    (255, 0, 0), 2)
        
        # Display result
        cv2.imshow("Result", original)
        cv2.waitKey(0)
        cv2.destroyAllWindows()