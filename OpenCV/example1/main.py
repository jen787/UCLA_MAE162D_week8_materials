import cv2
import numpy as np

# read the image
img = cv2.imread("example1.jpg")

img = cv2.resize(img, (500, 500))

# create a motion blur kernel
psf = np.zeros((50, 50, 3))
psf = cv2.ellipse(psf, 
                  (25, 25),     # center
                  (22, 0),      # axes -- 22 for blur length, 0 for thin PSF 
                  15,           # angle of motion in degrees
                  0, 360,       # ful ellipse, not an arc
                  (1, 1, 1),    # white color
                  thickness=-1) # filled

# normalize by sum of one channel 
psf /= psf[:,:,0].sum()

# apply the kernel to the image
result = cv2.filter2D(img, -1, psf)

cv2.imwrite("result.jpg", result)
print("[MAIN] Image saved successfully!")