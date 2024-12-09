import sys
from operator import itemgetter
import numpy as np
import cv2
import matplotlib.pyplot as plt


# -----------------------------#
#   计算原始输入图像
#   每一次缩放的比例
# -----------------------------#
def calculateScales(img):
    """
    计算图像金字塔的缩放比例。

    该函数根据输入图像的尺寸，计算出一系列缩放比例，用于生成图像金字塔。    图像金字塔用于在不同尺度下检测人脸。

    参数:    img: 输入的图像，通常为OpenCV格式（numpy数组）。

    返回值:    scales: 一个列表，包含所有计算出的缩放比例。
    """

    copy_img = img.copy()  # 复制图像以避免修改原始图像

    pr_scale = 1.0  # 初始化缩放比例为1.0

    h, w, _ = copy_img.shape  # 获取图像的高、宽

    # 引申优化项  = resize(h*500/min(h,w), w*500/min(h,w))，第一次初始化缩放
    if min(w, h) > 500:  # 如果图像的短边大于500像素，调整缩放比例，使短边为500像素
        pr_scale = 500.0 / min(h, w)
        w = int(w * pr_scale)
        h = int(h * pr_scale)

    elif max(w, h) < 500:  # 如果图像的长边小于500像素，调整缩放比例，使长边为500像素
        pr_scale = 500.0 / max(h, w)
        w = int(w * pr_scale)
        h = int(h * pr_scale)

    scales = []  # 初始化缩放比例列表
    factor = 0.709  # 初始化缩放因子
    factor_count = 0  # 初始化缩放因子计
    minl = min(h, w)  # 获取缩放后图像的短边长度

    while minl >= 12:  # 当短边长度大于等于12像素时，循环计算缩放比例
        scales.append(pr_scale * pow(factor, factor_count))  # 将计算出的缩放比例添加到列表中
        minl *= factor  # 缩小图像尺寸
        factor_count += 1  # 增加缩放因子计数

    return scales  # 返回所有计算出的缩放比例


# -------------------------------------#
#   对pnet处理后的结果进行处理
# -------------------------------------#
def detect_face_12net(cls_prob, roi, out_side, scale, width, height, threshold):
    """
    Pnet网络检测人脸。

    参数:
    cls_prob: 分类概率数组。
    roi: 区域提议网络的输出。
    out_side: 输出的边长。
    scale: 图像缩放比例。
    width: 图像宽度。
    height: 图像高度。
    threshold: 概率阈值。

    返回:    经过非极大值抑制后的矩形框。
    """
    # 交换数组维度以便于处理
    cls_prob = np.swapaxes(cls_prob, 0, 1)
    roi = np.swapaxes(roi, 0, 2)

    stride = 0

    if out_side != 1:  # stride略等于2
        stride = float(2 * out_side - 1) / (out_side - 1)

    (x, y) = np.where(cls_prob >= threshold)  # 找到所有大于阈值的坐标

    boundingbox = np.array([x, y]).T   # 初始化边界框数组
    bb1 = np.fix((stride * (boundingbox) + 0) * scale)   # 将坐标转换回原始图像尺寸
    bb2 = np.fix((stride * (boundingbox) + 11) * scale)
    boundingbox = np.concatenate((bb1, bb2), axis=1)  # 更新边界框数组

    dx1 = roi[0][x, y]  # 提取ROI的偏移量
    dx2 = roi[1][x, y]
    dx3 = roi[2][x, y]
    dx4 = roi[3][x, y]

    score = np.array([cls_prob[x, y]]).T  # 提取分类概率作为得分
    offset = np.array([dx1, dx2, dx3, dx4]).T   # 组合偏移量
    boundingbox = boundingbox + offset * 12.0 * scale  # 应用偏移量到边界框
    rectangles = np.concatenate((boundingbox, score), axis=1) # 组合边界框和得分
    rectangles = rect2square(rectangles)  # 将矩形调整为正方形
    pick = []

    for i in range(len(rectangles)):  # 遍历所有矩形并进行裁剪
        x1 = int(max(0, rectangles[i][0]))
        y1 = int(max(0, rectangles[i][1]))
        x2 = int(min(width, rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        sc = rectangles[i][4]
        # 确保矩形有效
        if x2 > x1 and y2 > y1:
            pick.append([x1, y1, x2, y2, sc])
    # 返回经过非极大值抑制的矩形
    return NMS(pick, 0.3)



# -----------------------------#
#   将长方形调整为正方形
# -----------------------------#
def rect2square(rectangles):
    w = rectangles[:, 2] - rectangles[:, 0]
    h = rectangles[:, 3] - rectangles[:, 1]
    l = np.maximum(w, h).T
    rectangles[:, 0] = rectangles[:, 0] + w * 0.5 - l * 0.5
    rectangles[:, 1] = rectangles[:, 1] + h * 0.5 - l * 0.5
    rectangles[:, 2:4] = rectangles[:, 0:2] + np.repeat([l], 2, axis=0).T
    return rectangles


# -------------------------------------#
#   非极大抑制
# -------------------------------------#
def NMS(rectangles, threshold):
    """
    非极大值抑制（Non-Maximum Suppression，NMS）算法。
    该算法用于过滤掉多余的边界框，只保留最有可能包含对象的边界框。

    参数:
    rectangles - 一个列表，包含多个边界框，每个边界框是一个列表，前四个元素是边界框的坐标，第五个元素是该边界框的置信度。
    threshold - 一个浮点数，表示重叠度的阈值，超过此阈值的边界框将被过滤掉。

    返回:
    result_rectangle - 经过NMS算法处理后的边界框列表。
    """

    if len(rectangles) == 0:  # 如果输入的边界框列表为空，则直接返回该列表
        return rectangles

    boxes = np.array(rectangles)  # 将输入的边界框列表转换为NumPy数组，便于后续处理
    # 分别提取边界框的x1, y1, x2, y2坐标和置信度
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, 4]

    area = np.multiply(x2 - x1 + 1, y2 - y1 + 1)  # 计算每个边界框的面积
    I = np.array(s.argsort())  # 按照置信度排序，获取排序后的索引
    pick = []  # 初始化用于保存最终选择的边界框索引的列表

    while len(I) > 0:  # 当索引列表非空时，进行循环处理
        # 计算当前置信度最高的边界框与其他边界框的重叠部分的坐标
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]])  # I[-1] have hightest prob score, I[0:-1]->others
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])

        w = np.maximum(0.0, xx2 - xx1 + 1)  # 计算重叠部分的宽度
        h = np.maximum(0.0, yy2 - yy1 + 1)  # 计算重叠部分的高度

        inter = w * h  # 计算重叠面积

        o = inter / (area[I[-1]] + area[I[0:-1]] - inter)  # 计算重叠度（Intersection over Union，IoU）
        pick.append(I[-1])  # 将置信度最高的边界框的索引添加到最终选择的列表中
        I = I[np.where(o <= threshold)[0]]  # 更新索引列表，保留重叠度不超过阈值的边界框的索引

    result_rectangle = boxes[pick].tolist()  # 根据最终选择的索引，提取边界框，并转换为列表格式返回
    return result_rectangle


# -------------------------------------#
#   对Rnet处理后的结果进行处理
# -------------------------------------#
def filter_face_24net(cls_prob, roi, rectangles, width, height, threshold):
    prob = cls_prob[:, 1]
    pick = np.where(prob >= threshold)
    rectangles = np.array(rectangles)

    x1 = rectangles[pick, 0]
    y1 = rectangles[pick, 1]
    x2 = rectangles[pick, 2]
    y2 = rectangles[pick, 3]

    sc = np.array([prob[pick]]).T

    dx1 = roi[pick, 0]
    dx2 = roi[pick, 1]
    dx3 = roi[pick, 2]
    dx4 = roi[pick, 3]

    w = x2 - x1
    h = y2 - y1

    x1 = np.array([(x1 + dx1 * w)[0]]).T
    y1 = np.array([(y1 + dx2 * h)[0]]).T
    x2 = np.array([(x2 + dx3 * w)[0]]).T
    y2 = np.array([(y2 + dx4 * h)[0]]).T

    rectangles = np.concatenate((x1, y1, x2, y2, sc), axis=1)
    rectangles = rect2square(rectangles)
    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0, rectangles[i][0]))
        y1 = int(max(0, rectangles[i][1]))
        x2 = int(min(width, rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        sc = rectangles[i][4]
        if x2 > x1 and y2 > y1:
            pick.append([x1, y1, x2, y2, sc])
    return NMS(pick, 0.3)


# ---------------------------------------------#
# ---------------------------------------------#
def filter_face_48net(cls_prob, roi, pts, rectangles, width, height, threshold):
    """
    对onet处理后的结果进行处理

    参数:
    - cls_prob: 分类概率数组。
    - roi: 区域建议网络的输出。
    - pts: 特征点坐标数组。
    - rectangles: 检测框数组。
    - width: 图像宽度。
    - height: 图像高度。
    - threshold: 概率阈值。

    返回:
    - 经过非极大值抑制后的检测框和特征点信息。
    """

    # 提取正脸概率
    prob = cls_prob[:, 1]

    # 选择概率大于阈值的检测框
    pick = np.where(prob >= threshold)
    rectangles = np.array(rectangles)

    # 提取检测框坐标
    x1 = rectangles[pick, 0]
    y1 = rectangles[pick, 1]
    x2 = rectangles[pick, 2]
    y2 = rectangles[pick, 3]

    # 提取检测框的分类概率
    sc = np.array([prob[pick]]).T

    # 提取区域建议网络的输出
    dx1 = roi[pick, 0]
    dx2 = roi[pick, 1]
    dx3 = roi[pick, 2]
    dx4 = roi[pick, 3]

    # 计算检测框的宽度和高度
    w = x2 - x1
    h = y2 - y1

    # 计算特征点坐标
    pts0 = np.array([(w * pts[pick, 0] + x1)[0]]).T
    pts1 = np.array([(h * pts[pick, 5] + y1)[0]]).T
    pts2 = np.array([(w * pts[pick, 1] + x1)[0]]).T
    pts3 = np.array([(h * pts[pick, 6] + y1)[0]]).T
    pts4 = np.array([(w * pts[pick, 2] + x1)[0]]).T
    pts5 = np.array([(h * pts[pick, 7] + y1)[0]]).T
    pts6 = np.array([(w * pts[pick, 3] + x1)[0]]).T
    pts7 = np.array([(h * pts[pick, 8] + y1)[0]]).T
    pts8 = np.array([(w * pts[pick, 4] + x1)[0]]).T
    pts9 = np.array([(h * pts[pick, 9] + y1)[0]]).T

    # 调整检测框坐标
    x1 = np.array([(x1 + dx1 * w)[0]]).T
    y1 = np.array([(y1 + dx2 * h)[0]]).T
    x2 = np.array([(x2 + dx3 * w)[0]]).T
    y2 = np.array([(y2 + dx4 * h)[0]]).T

    # 合并检测框和特征点信息
    rectangles = np.concatenate((x1, y1, x2, y2, sc, pts0, pts1, pts2, pts3, pts4, pts5, pts6, pts7, pts8, pts9),
                                axis=1)

    # 初始化检测框数组
    pick = []
    for i in range(len(rectangles)):
        # 限制检测框在图像范围内
        x1 = int(max(0, rectangles[i][0]))
        y1 = int(max(0, rectangles[i][1]))
        x2 = int(min(width, rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        # 确保检测框宽度和高度大于0
        if x2 > x1 and y2 > y1:
            pick.append([x1, y1, x2, y2, rectangles[i][4],
                         rectangles[i][5], rectangles[i][6], rectangles[i][7], rectangles[i][8], rectangles[i][9],
                         rectangles[i][10], rectangles[i][11], rectangles[i][12], rectangles[i][13], rectangles[i][14]])
    # 进行非极大值抑制
    return NMS(pick, 0.3)
