from Filter import Filter
from matplotlib import pyplot as plt
import numpy as np

image_after_avg_filter=Filter.apply_avg_filter('cat.png',7)
image_after_gaussian_filter=Filter.apply_gaussian_filter('cat.png',Filter.gaussian_kernel(5,10))
image_after_median_filter=Filter.apply_median_filter('cat.png',9)


plt.subplot(1, 3, 1)
plt.imshow(image_after_median_filter)
plt.title('Median Filter')


plt.subplot(1, 3, 2)
plt.imshow(image_after_avg_filter)
plt.title('Average Filter')


plt.subplot(1, 3, 3)
plt.imshow(image_after_gaussian_filter)  # Assuming grayscale, change cmap accordingly
plt.title('Gaussian Filter')


plt.tight_layout()


plt.show()
