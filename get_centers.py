import time

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth


def get_nonzero(image):
    """取非零值并转换为坐标数组"""
    nonzero = image.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])
    return np.stack((nonzero_x, nonzero_y), axis=-1)


def mean_shift(img, src, line, window_height):
    """聚类并在结果数组中插入行号"""
    bandwidth = estimate_bandwidth(src, quantile=0.2, n_samples=500)

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=False)
    ms.fit(src)

    # labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    h = img.shape[0]

    # labels_unique = np.unique(labels)
    # n_clusters_ = len(labels_unique)

    # 插入行号
    centers = np.insert(cluster_centers, 0, values=int(line), axis=1)

    for pot in centers:
        pot[2] = h - (line + 1) * window_height + pot[2]

    return centers.astype(np.int32)


def get_centers(image, ms_window_height):
    """按行对图片聚类并返回三维数组"""
    h = image.shape[0]

    # # 高斯模糊
    # blur_img = cv2.GaussianBlur(image, (5, 5), 0)
    #
    # # 阈值转换
    # ret, threshold_img = cv2.threshold(blur_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #
    # # 形态转换（过滤噪点）
    # kernel = np.ones((5, 5), np.uint8)
    # opening_img = cv2.morphologyEx(threshold_img, cv2.MORPH_OPEN, kernel)

    opening_img = image

    start_time = time.time()
    count = 0
    results = []
    while h - ms_window_height > 0:
        split_img = opening_img[h - ms_window_height: h, ]
        nonzero = get_nonzero(split_img)
        if nonzero.size > 0:
            points = mean_shift(image, nonzero, count, ms_window_height)
            results.append(points.tolist())
        h -= ms_window_height
        count += 1
    end_time = time.time()

    print("image shape", opening_img.shape, "窗口高度", ms_window_height, "循环次数", count, "执行时间", end_time - start_time, "s")

    return results, opening_img


def main():
    img = cv2.imread('res/img/c6_1.jpg', cv2.IMREAD_GRAYSCALE)

    # 输出图形及聚类后的点
    plt.subplot(211)
    plt.imshow(img, cmap='gray')
    plt.subplot(212)

    points, filter_img = get_centers(img, 30)
    print("------->\n", points)
    for p in points:
        for po in p:
            cv2.circle(filter_img,
                       (int(po[1]), int(po[2])), 5, (111, 111, 111), -1)

    plt.imshow(filter_img)
    plt.show()


if __name__ == "__main__":
    main()
