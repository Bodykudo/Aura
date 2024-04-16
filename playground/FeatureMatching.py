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

