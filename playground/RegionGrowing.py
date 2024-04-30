import numpy as np
import cv2

def get_8_connected(x, y, shape):
    xmax = shape[0] - 1
    ymax = shape[1] - 1
    
    connected_pixels = []
    
    for dx in range(3):
        for dy in range(3):
            connected_pixel_x = x + dx - 1
            connected_pixel_y = y + dy - 1
            if (0 <= connected_pixel_x <= xmax) and (0 <= connected_pixel_y <= ymax) and \
               not (connected_pixel_x == x and connected_pixel_y == y):
                connected_pixels.append((connected_pixel_x, connected_pixel_y))
    
    return connected_pixels

def region_growing(img, seed_points):
    processed = np.full((img.shape[0], img.shape[1]), False)
    outimg = np.zeros_like(img)

    for index, pix in enumerate(seed_points):
        processed[pix[0], pix[1]] = True
        outimg[pix[0], pix[1]] = img[pix[0], pix[1]]

    while len(seed_points) > 0:
        pix = seed_points[0]
        
        for coord in get_8_connected(pix[0], pix[1], img.shape):
            if not processed[coord[0], coord[1]]:
                if img[coord[0], coord[1]] != 0:
                    outimg[coord[0], coord[1]] = outimg[pix[0], pix[1]]
                    if not processed[coord[0], coord[1]]:
                        seed_points.append(coord)
                    processed[coord[0], coord[1]] = True

        seed_points.pop(0)

    return outimg, processed


def on_mouse(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Seed: ' + str(x) + ', ' + str(y), img[y, x])
        clicks.append((int(y), int(x)))

clicks = []
image_path = r"C:\College\3rd Year\Second Term\Computer Vision\Aura\playground\lenna.png"
image_org = cv2.imread(image_path)
image_gray = cv2.cvtColor(image_org, cv2.COLOR_BGR2GRAY)
ret, img = cv2.threshold(image_gray, 130, 255, cv2.THRESH_BINARY)

cv2.namedWindow('Input')
cv2.setMouseCallback('Input', on_mouse, 0)
cv2.imshow('Input', image_org)
cv2.waitKey()
cv2.destroyAllWindows()

seed = clicks
region_filled, processed = region_growing(img, seed)

# Find contours of the filled region
contours, _ = cv2.findContours(processed.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
for contour in contours:
    cv2.drawContours(image_org, [contour], -1, (0, 255, 0), 2)

cv2.imshow('Region Edges on Original Image', image_org)
cv2.waitKey()
cv2.destroyAllWindows()
