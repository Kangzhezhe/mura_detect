import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
# 绘制频谱图
def show_fft(magnitude_spectrum,figname="fft_magnitude_spectrum.png"):
    plt.figure(figsize=(8, 8))
    plt.imshow(np.log(1 + magnitude_spectrum), cmap='gray')
    plt.title('Magnitude Spectrum of Fourier Transform')
    plt.axis('off')
    plt.savefig(figname)

def texture_suppression(img, max_spec_amp = 0.0001,max_distance_to_center = 300,kernel_ratio = 10, lowpass_radius = 300,debug=False):
    # 将彩色图像转换为灰度图像
    grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 对灰度图像进行傅立叶变换
    fourier_transform = np.fft.fft2(grayscale_image)

    # 低通滤波
    order = 4
    cutoff = 150
    fshift = np.fft.fftshift(fourier_transform)
    rows, cols = fshift.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    rr, cc = np.ogrid[:rows, :cols]
    distance = np.sqrt((rr - crow) ** 2 + (cc - ccol) ** 2)
    # lowpass = 1 / (1 + (distance / cutoff) ** (2 * order))
    lowpass = np.exp(-distance**2 / (2*cutoff**2))
    fshift = fshift * lowpass
    # mask[distance < lowpass_radius] = 1
    # fshift = fshift * mask
    fourier_transform = np.fft.ifftshift(fshift)

    # 计算傅立叶变换后的频谱
    magnitude_spectrum = np.fft.fftshift(np.abs(fourier_transform))

    # 获得纹理频谱特征
    mask = magnitude_spectrum > max_spec_amp*np.max(magnitude_spectrum)
    zeroed_fourier_transform = np.multiply(magnitude_spectrum, mask)
    zeroed_fourier_transform = np.clip(zeroed_fourier_transform, 0, 255).astype(np.uint8)

    # 增强纹理频谱特征
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded_image = cv2.erode(zeroed_fourier_transform, kernel, iterations=1)
    if debug : show_fft(eroded_image,'eroded_magnitude_spectrum.png')
    dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)
    if debug : show_fft(dilated_image,'dilated_magnitude_spectrum.png')

    # 找到高亮纹理频谱中心点的位置
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 计算每个高亮点的中心
    center_points = []
    for contour in contours:
        moments = cv2.moments(contour)
        if moments['m00'] != 0:
            center_x = int(moments['m10'] / moments['m00'])
            center_y = int(moments['m01'] / moments['m00'])
            center_points.append((center_x, center_y))

    # 计算图像中心
    center_X, center_Y = img.shape[1] // 2, img.shape[0] // 2
    # 设置阈值，如果距离中心较近则不去除该中心点
    distances = [np.sqrt((center_x - center_X) ** 2 + (center_y - center_Y) ** 2) for center_x, center_y in center_points]
    # max_distance_to_center = 1000  # 请根据实际情况调整阈值
    center_points = [(x, y) for (x, y), distance in zip(center_points, distances) if distance > max_distance_to_center]

    # 为每个中心点生成掩模
    masks = []
    areas = [cv2.contourArea(contour) for contour in contours]
    average_area = np.mean(areas)
    kernel_size = int (np.sqrt(average_area / np.pi) * kernel_ratio + 0.5)

    # 使用圆形结构元素创建掩模
    for center_x, center_y in center_points:
        center = (center_x, center_y)
        mask = np.zeros_like(eroded_image)
        cv2.circle(mask, center, kernel_size // 2, 255, -1)
        masks.append(mask)

    masked = np.ones_like(eroded_image)
    for mask in masks:
        masked = cv2.bitwise_and(masked, ~mask)

    # 显示掩模
    if debug: cv2.imwrite("spec_mask.jpg",masked*255)
    # zeroed_fourier_transform_masked = cv2.bitwise_and(zeroed_fourier_transform, masked)
    # show_fft(zeroed_fourier_transform_masked,'zeroed_magnitude_spectrum_masked')
    # import ipdb; ipdb.set_trace()

    # 对处理后的频域图像进行反傅立叶变换
    masked_ishift = np.fft.ifftshift(masked)
    if debug : show_fft(np.abs(fourier_transform),'origin')
    reconstructed_image_spec = np.multiply(masked_ishift, fourier_transform)
    if debug : show_fft(np.abs(reconstructed_image_spec),'reconstructed')
    reconstructed_image = np.fft.ifft2(reconstructed_image_spec).real
    if debug: cv2.imwrite("reconstructed_image.jpg",reconstructed_image)
    return reconstructed_image

debug = False
if __name__ == '__main__':
    if debug:
        img = cv2.imread("Visualize/斜纹/20210116_164538_Main_0_4_W128_Org.jpg")
        output = texture_suppression(img,debug=True)
    else:
        data_path = "Visualize/"
        problem_list = os.listdir(data_path)
        for problem_name in tqdm(problem_list):
            img_list = os.listdir(os.path.join(data_path, problem_name))
            os.makedirs("Texture/{}/".format(problem_name), exist_ok=True)
            for img_name in tqdm(img_list):
                img = cv2.imread(os.path.join(data_path, problem_name, img_name))
                output = texture_suppression(img)
                cv2.imwrite("Texture/{}/{}.jpg".format(problem_name, img_name.split(".")[0]), output)