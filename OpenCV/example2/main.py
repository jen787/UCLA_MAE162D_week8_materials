import cv2
import numpy as np

# read the image
img = cv2.imread("example2.jpg")

img = cv2.resize(img, (500, 500))

# convert the image to grayscale and apply Canny edge detection
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)

# find the contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# draw the contours
result = cv2.drawContours(img, contours, -1, (243, 239, 131), 2)

# show the original image with the largest contour drawn on it
cv2.imwrite("result.jpg", result)
print("[MAIN] Image saved successfully!")