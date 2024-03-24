import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
import os
from tqdm import tqdm

def enhancement(img,gamma=1.5,debug=False):
    # Load the image
    # img = cv2.imread('Texture/色斑/20200623_152207_Sub_1_0_W255_Org.jpg', cv2.IMREAD_GRAYSCALE)

    # Apply histogram equalization to enhance contrast
    equ = cv2.equalizeHist(img)

    # Display the original and enhanced images
    if debug:
        cv2.imwrite('Original_Image.jpg', img)
        cv2.imwrite('Enhanced_Image.jpg', equ)
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
        plt.savefig('histogram.png')


    def gamma_correction(image, gamma):
        # 标准化像素值到[0, 1]范围
        image = image / 255.0
        # 应用gamma变换
        corrected_image = np.power(image, gamma)
        # 将像素值重新缩放到[0, 255]范围
        corrected_image = (corrected_image * 255).astype(np.uint8)
        return corrected_image

    gamma_corrected = gamma_correction(equ, gamma)
    if debug:cv2.imwrite('gamma_corrected.jpg', gamma_corrected)

    kernel_size = 5
    blurred_image = cv2.medianBlur(gamma_corrected, kernel_size)
    if debug:cv2.imwrite('blurred_image.jpg', blurred_image)

    kernel_size = 5
    aver_blurred_image = cv2.blur(blurred_image, (kernel_size, kernel_size))
    if debug:cv2.imwrite('aver_blurred_image.jpg', aver_blurred_image)

    return blurred_image


debug = False

if __name__ == '__main__':
    if debug:
        img = cv2.imread('Texture/色斑/20200623_152207_Sub_1_0_W255_Org.jpg', cv2.IMREAD_GRAYSCALE)
        out =enhancement(img,debug=True)
    else:
        data_path = "Texture/"
        problem_list = os.listdir(data_path)
        for problem_name in tqdm(problem_list):
            img_list = os.listdir(os.path.join(data_path, problem_name))
            os.makedirs("Enhance/{}/".format(problem_name), exist_ok=True)
            for img_name in tqdm(img_list):
                img = cv2.imread(os.path.join(data_path, problem_name, img_name),cv2.IMREAD_GRAYSCALE)
                output = enhancement(img)
                cv2.imwrite("Enhance/{}/{}.jpg".format(problem_name, img_name.split(".")[0]), output)