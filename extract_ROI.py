import os
import cv2

def extract_ROI(img, constant_threshold=10):
    # 将图像转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("gray.jpg", gray)
    # 对图像进行自适应阈值处理
    threshshold = gray.min() + constant_threshold
    _, thresh = cv2.threshold(gray, threshshold, 255, cv2.THRESH_BINARY)

    # 查找图像中的轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # import pdb; pdb.set_trace()

    # # 获取最大的轮廓
    largest_contour = max(contours, key=cv2.contourArea)

    # 获取最大轮廓的外接矩形
    x, y, w, h = cv2.boundingRect(largest_contour)

    # 提取ROI
    roi = img[y:y+h, x:x+w]

    return roi


if __name__ == '__main__':
    data_path = "data/mura/"
    problem_list = os.listdir(data_path)
    for problem_name in problem_list:
        img_list = os.listdir(os.path.join(data_path, problem_name))
        os.makedirs("Visualize/{}/".format(problem_name), exist_ok=True)
        for img_name in img_list:
            img = cv2.imread(os.path.join(data_path, problem_name, img_name))
            roi = extract_ROI(img)
            cv2.imwrite("Visualize/{}/{}.jpg".format(problem_name, img_name.split(".")[0]), roi)