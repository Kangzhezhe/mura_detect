import cv2
import os
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt 
import numpy as np
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

def threshold(img, debug=False, mura_black=True, y_offset=5, y_min=100, y_max=205, center_percent=15):
    if debug: cv2.imwrite('origin.jpg', img)
    
    counts = np.bincount(img.flatten())
    most_pixel = np.argmax(counts)

    percentile_value_r = np.percentile(img, 100-center_percent/2)
    percentile_value_l = np.percentile(img, center_percent/2)

    # points = [(0,0),(percentile_value_l-1, y_min),(percentile_value_l, most_pixel - y_offset), (percentile_value_r, most_pixel + y_offset), (percentile_value_r+1, y_max),(255.0, 255.0)]
    # equ = piecewise_linear(img , points).astype(np.uint8)

    # if debug:
    #     X = np.linspace(0, 1, 10000) *255
    #     y = piecewise_linear(X,points)
    #     plt.plot(X, y)
    #     plt.xlabel('x')
    #     plt.ylabel('y')
    #     plt.title('Piecewise Linear Function')
    #     plt.savefig('piecewise_linear.png')
    #     show_hist(img, equ)
    #     cv2.imwrite('equed.jpg', equ)

    # thresholded_image = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,31,3)
    ret, thresholded_image = cv2.threshold(img, (percentile_value_l if mura_black else percentile_value_r), 255, cv2.THRESH_BINARY)
    if debug: cv2.imwrite('threshold.jpg', thresholded_image)

    return thresholded_image

debug = True
if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Thresholding images')
    # parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    # args = parser.parse_args()

    # debug = args.debug

    if debug:
        # img = cv2.imread('Enhance/污渍/20210116_163509_Main_0_4_W128_Org.jpg', cv2.IMREAD_GRAYSCALE)
        img = cv2.imread('gamma_corrected_white.jpg', cv2.IMREAD_GRAYSCALE)
        out = threshold(img,debug=True)
    else:
        data_path = "Enhance/"
        problem_list = os.listdir(data_path)
        for problem_name in tqdm(problem_list):
            # import ipdb; ipdb.set_trace()
            img_list = os.listdir(os.path.join(data_path, problem_name))
            os.makedirs("Threshold/{}/".format(problem_name), exist_ok=True)
            for img_name in tqdm(img_list):
                img = cv2.imread(os.path.join(data_path, problem_name, img_name),cv2.IMREAD_GRAYSCALE)
                output = threshold(img)
                cv2.imwrite("Threshold/{}/{}.jpg".format(problem_name, img_name.split(".")[0]), output)

