from playground.Filter import Filter
from matplotlib import pyplot as plt

image=Filter.apply_avg_filter('cat.png',5)
print("done")
plt.imshow(image)
plt.show()
