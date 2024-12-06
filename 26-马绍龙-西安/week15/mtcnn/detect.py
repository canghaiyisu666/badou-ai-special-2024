import cv2
import numpy as np
from mtcnn import mtcnn

# 读取图片文件
img = cv2.imread('img/timg.jpg')

# 初始化MTCNN模型
model = mtcnn()
# 设置三级网络的置信度阈值，三段网络的置信度阈值不同
threshold = [0.5, 0.6, 0.7]
# 使用模型检测人脸，返回人脸矩形
rectangles = model.detectFace(img, threshold)
# 复制图片以用于绘制检测结果
draw = img.copy()

# 遍历每个人脸矩形
for rectangle in rectangles:
    # 确保矩形信息完整
    if rectangle is not None:
        # 计算矩形宽度和高度
        W = -int(rectangle[0]) + int(rectangle[2])
        H = -int(rectangle[1]) + int(rectangle[3])
        # 计算填充的宽度和高度
        paddingH = 0.01 * W
        paddingW = 0.02 * H
        # 根据矩形和填充值裁剪图片
        crop_img = img[int(rectangle[1] + paddingH):int(rectangle[3] - paddingH),
                   int(rectangle[0] - paddingW):int(rectangle[2] + paddingW)]
        # 确保裁剪后的图片存在且尺寸有效
        if crop_img is None:
            continue
        if crop_img.shape[0] < 0 or crop_img.shape[1] < 0:
            continue
        # 在图片上绘制人脸矩形
        cv2.rectangle(draw, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])), (255, 0, 0),
                      1)

        # 绘制人脸关键点
        for i in range(5, 15, 2):
            cv2.circle(draw, (int(rectangle[i + 0]), int(rectangle[i + 1])), 2, (0, 255, 0))

# 保存绘制结果图片
cv2.imwrite("img/out.jpg", draw)

# 显示绘制结果图片
cv2.imshow("test", draw)
# 等待按键事件
c = cv2.waitKey(0)
