import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
import os
from tqdm import tqdm

SCALE = 8

def fit_curve(brightness_values, coordinates):
    # 创建插值网格
    xi = np.linspace(0, coordinates.shape[1]-1, coordinates.shape[1])
    yi = np.linspace(0, coordinates.shape[0]-1, coordinates.shape[0])
    xi, yi = np.meshgrid(xi, yi)

    import ipdb; ipdb.set_trace()

    # import ipdb; ipdb.set_trace()
    # 使用径向基函数对插值函数进行插值
    rbf = Rbf(coordinates[:,:, 0], coordinates[:,:, 1], brightness_values, function='multiquadric')
    interpolated_values = rbf(xi, yi)

    return interpolated_values

def apply_curve(image):
    brightness_values = np.resize(image, (image.shape[0]//SCALE, image.shape[1]//SCALE)).astype(np.float32) / 255.0
    coordinates = np.indices(brightness_values.shape).astype(np.float32).transpose(1,2,0)

    # 对亮度值进行二维曲面拟合
    interpolated_values = fit_curve(brightness_values, coordinates)
    # 将插值值应用于图像的每个像素的亮度调整
    mapped_values = np.interp(interpolated_values, (interpolated_values.min(), interpolated_values.max()), (0, 255)).astype(np.uint8)
    cv2.imwrite('curve_fit.jpg',  mapped_values)
    import ipdb; ipdb.set_trace()
    
    return mapped_values

MURA_TYPE = ['色斑', '条带', '白团', '黑斑', '污渍', '斜纹']

def localEqualHist(image):
    clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(4,4))
    dst = clahe.apply(image)
    return dst

def global_EuqualHist(image,threshold = 50):
    import ipdb; ipdb.set_trace()
    mask = image > threshold
    hist, _ =  np.histogram(image[mask], bins=256, range=[0,256])

     # 计算大于阈值的像素的累积直方图
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    # 对大于阈值的像素进行直方图均衡化
    equalized_image = np.interp(image, range(256), cdf_normalized)
    return equalized_image


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

def frequency_domain_filtering(image, radius = 200):
    
    # 对灰度图像进行傅里叶变换
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    
    # 创建一个高通滤波器，滤除高频部分
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow-radius:crow+radius, ccol-radius:ccol+radius] = 1
    
    # 应用高通滤波器
    fshift = fshift * mask
    
    # 对变换后的结果进行逆变换
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back.astype(np.uint8)

def enhancement(img,gamma_black=1.5,gamma_white=0.5,debug=False, mura_type='黑斑'):
    if mura_type not in MURA_TYPE:
        print("mura type must be in {}".format(MURA_TYPE))
        return img

    # 局部直方图均衡效果更好
    # equ = cv2.equalizeHist(img)
    equ = localEqualHist(img)

    # apply_curve(equ)
    if debug:
        cv2.imwrite('Original_Image.jpg', img)
        cv2.imwrite('Enhanced_Image.jpg', equ)
        show_hist(img,equ)

    # equ1 =global_EuqualHist(equ)
    # equ1 =cv2.equalizeHist(equ)
    equ1 = equ
    # if debug:
    #     cv2.imwrite('Enhanced_Image1.jpg', equ1)
    #     show_hist(equ,equ1,"histogram_1")

    # def gamma_correction(image, gamma):
    #     # 标准化像素值到[0, 1]范围
    #     image = image / 255.0
    #     # 应用gamma变换
    #     corrected_image = np.power(image, gamma)
    #     # 将像素值重新缩放到[0, 255]范围
    #     corrected_image = (corrected_image * 255).astype(np.uint8)
    #     return corrected_image

    # if mura_type in ['色斑', '黑斑', '污渍']:
    #     corrected = gamma_correction(equ1, gamma_white)
    #     if debug:cv2.imwrite('gamma_corrected_black.jpg', corrected)
    # elif mura_type in ['条带', '白团', '斜纹']:
    #     corrected = gamma_correction(equ1, gamma_black)
    #     if debug:cv2.imwrite('gamma_corrected_white.jpg', corrected)
    corrected = equ1

    kernel_size = 5
    blurred_image = cv2.medianBlur(corrected, kernel_size)
    # blurred_image = amp_medianBlur_gray(corrected)
    if debug:cv2.imwrite('blurred_image.jpg', blurred_image)


    # if mura_type is '黑斑':
    # print(mura_type)
    eroded_kernal = 11
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (eroded_kernal, eroded_kernal))
    eroded_image = cv2.erode(blurred_image, kernel)
    if debug:cv2.imwrite('Eroded_Image.jpg', eroded_image)

    dilate_kernel_size = 11
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_kernel_size, dilate_kernel_size))
    dilated_image = cv2.dilate(eroded_image, dilate_kernel)
    if debug:cv2.imwrite('Dilated_image.jpg', dilated_image)

    output = dilated_image
    return output

debug = True

if __name__ == '__main__':
    if debug:
        # img = cv2.imread('Texture/斜纹/20210116_165013_Main_0_4_W128_Org.jpg', cv2.IMREAD_GRAYSCALE) #斜纹，不明显
        img = cv2.imread('Texture/黑斑/20210622_152757_Main_0_0_W255_Org.jpg', cv2.IMREAD_GRAYSCALE) #黑点，效果好
        # img = cv2.imread('Texture/色斑/20200623_152207_Sub_1_0_W255_Org.jpg', cv2.IMREAD_GRAYSCALE) #色斑附带白带
        # img = cv2.imread('Texture/白团/20210116_150440_Main_0_5_W64_Org.jpg', cv2.IMREAD_GRAYSCALE) #包团加噪声
        out = enhancement(img,debug=True,mura_type='白团')
    else:
        data_path = "Texture/"
        problem_list = os.listdir(data_path)
        for problem_name in tqdm(problem_list):
            # import ipdb; ipdb.set_trace()
            img_list = os.listdir(os.path.join(data_path, problem_name))
            os.makedirs("Enhance/{}/".format(problem_name), exist_ok=True)
            for img_name in tqdm(img_list):
                img = cv2.imread(os.path.join(data_path, problem_name, img_name),cv2.IMREAD_GRAYSCALE)
                output = enhancement(img,mura_type=problem_name)
                cv2.imwrite("Enhance/{}/{}.jpg".format(problem_name, img_name.split(".")[0]), output)

