import time

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth

WINDOW_HEIGHT = 10


def get_nonzero(image):
    """取非零值并转换为坐标数组"""
    nonzero = image.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])
    return np.stack((nonzero_x, nonzero_y), axis=-1)


def mean_shift(src, line):
    """聚类并在结果数组中插入行号"""
    bandwidth = estimate_bandwidth(src, quantile=0.2, n_samples=500)

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=False)
    ms.fit(src)

    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    # n_clusters_ = len(labels_unique)

    # 插入行号
    centers = np.insert(cluster_centers, 0, values=line, axis=1)

    for pot in centers:
        pot[2] = img.shape[0] - (line + 1) * WINDOW_HEIGHT + pot[2]

    return centers


def get_centers(image):
    """按行对图片聚类并返回三维数组"""
    h = image.shape[0]

    # 高斯模糊
    blur_img = cv2.GaussianBlur(img, (5, 5), 0)

    # 阈值转换
    ret, threshold_img = cv2.threshold(blur_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 形态转换（过滤噪点）
    kernel = np.ones((5, 5), np.uint8)
    opening_img = cv2.morphologyEx(threshold_img, cv2.MORPH_OPEN, kernel)

    start_time = time.time()
    count = 0
    results = []
    while h - WINDOW_HEIGHT > 0:
        split_img = opening_img[h - WINDOW_HEIGHT: h, ]
        nonzero = get_nonzero(split_img)
        if nonzero.size > 0:
            points = mean_shift(nonzero, count)
            results.append(points)
        h -= WINDOW_HEIGHT
        count += 1
    end_time = time.time()

    print("image shape", opening_img.shape, "窗口高度", WINDOW_HEIGHT, "循环次数", count, "执行时间", end_time - start_time, "s")

    return np.array(results), opening_img


img = cv2.imread('images/3.png', cv2.IMREAD_GRAYSCALE)

# 输出图形及聚类后的点
plt.subplot(211)
plt.imshow(img, cmap='gray')
plt.subplot(212)

X, filter_img = get_centers(img)
print("------->\n", X)
for lane in X:
    for po in lane:
        cv2.circle(filter_img,
                   (int(po[1]), int(po[2])), 5, (111, 111, 111), -1)

plt.imshow(filter_img, cmap='gray')
plt.show()
