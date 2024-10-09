import cv2
import numpy as np


img = cv2.imread('images/cropped_face_aligned.jpg')

img = img.astype(np.float32)
print(img)

B, G, R= cv2.split(img)
zeros = np.zeros_like(R)

cv2.imwrite("images/red_channel.jpg", cv2.merge([zeros, zeros, R]))
cv2.imwrite("images/green_channel.jpg", cv2.merge([zeros, G, zeros]))
cv2.imwrite("images/blue_channel.jpg", cv2.merge([B, zeros, zeros]))

img[..., 0] -= 93.5940
img[..., 1] -= 104.7624
img[..., 2] -= 129.1863

B, G, R= cv2.split(img)

print(B)
print(G)
print(R)


cv2.imwrite("images/n_red_channel.jpg", cv2.merge([zeros, zeros, R]))
cv2.imwrite("images/n_green_channel.jpg", cv2.merge([zeros, G, zeros]))
cv2.imwrite("images/n_blue_channel.jpg", cv2.merge([B, zeros, zeros]))
cv2.imwrite('images/normalized_image.jpg', img)
