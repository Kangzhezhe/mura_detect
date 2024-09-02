import cv2

img = cv2.imread("gamma_corrected_white.jpg")
ret, thresholded_image = cv2.threshold(img,100, 255, cv2.THRESH_BINARY)

cv2.imwrite('threshold.jpg', thresholded_image)