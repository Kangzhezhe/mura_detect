import numpy as np
import matplotlib.pyplot as plt 
import cv2


def show_hist(img,equ,figname="histogram"):
    plt.figure(figsize=(12,6))
    plt.subplot(2,1,1)
    plt.hist(img.ravel(),256,[0,256])
    plt.title('Histogram of Original Image')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.subplot(2,1,2)
    plt.hist(equ.ravel(),256,[0,256])
    plt.title('Histogram of Enhanced Image')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.savefig(figname)

def piecewise_linear(x, points):
    y = np.zeros_like(x)
    for i in range(len(points) - 1):
        mask = (x >= points[i][0]) & (x <= points[i + 1][0])
        x_step = x[mask]
        slope = (points[i + 1][1] - points[i][1]) / (points[i + 1][0] - points[i][0])
        y[mask] = points[i][1] + slope * (x_step - points[i][0])
    return y 

X = np.linspace(0, 1, 10000) *255

img = cv2.imread('Enhance/色斑/20200623_152207_Sub_1_0_W255_Org.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imwrite('origin.jpg', img)
counts = np.bincount(img.flatten())
most_pixel = np.argmax(counts)

# import ipdb ; ipdb.set_trace()

mura_black = True
y_offset = 5
y_min = 10
y_max = 245
center_percent = 20

percentile_value_r = np.percentile(img, 100-center_percent/2)
percentile_value_l = np.percentile(img, center_percent/2)

points = [(0,0),(percentile_value_l-1, y_min),(percentile_value_l, most_pixel - y_offset), (percentile_value_r, most_pixel + y_offset), (percentile_value_r+1, y_max),(255.0, 255.0)]
equ = piecewise_linear(img , points).astype(np.uint8)

y = piecewise_linear(X,points)
plt.plot(X, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Piecewise Linear Function')
plt.savefig('piecewise_linear.png')

show_hist(img, equ)
cv2.imwrite('equed.jpg', equ)

ret, thresholded_image = cv2.threshold(equ, (percentile_value_l if mura_black else percentile_value_r), 255, cv2.THRESH_BINARY)
cv2.imwrite('threshold.jpg', thresholded_image)
