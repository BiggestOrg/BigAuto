import numpy as np
import cv2
import time
from matplotlib import pyplot as plt
from itertools import cycle
from sklearn.cluster import MeanShift, estimate_bandwidth

WINDOW_HEIGHT = 10


def get_nonzero(image):
    """取非零值并转换为坐标数组"""
    nonzero = image.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])
    return np.stack((nonzero_x, nonzero_y), axis=-1)


def mean_shift(h,image, line):
    """聚类并在结果数组中插入行号"""
    bandwidth = estimate_bandwidth(image, quantile=0.4, n_samples=500)

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(image)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    # 插入行号
    print (h)
    centers = np.insert(cluster_centers, 0, values=int(line), axis=1)
    for pot in centers:
        pot[2]= h - (line+1) *WINDOW_HEIGHT + pot[2]
    print("--")
    print(centers)
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    # print("点数: ", n_clusters_, "\n", cluster_centers)

    return centers.astype(np.int32)


def get_centers(image):
    """按行对图片聚类并返回三维数组"""
    h = image.shape[0]
    image_h = image.shape[0]
    start_time = time.time()
    count = 0
    results = []
    while h - WINDOW_HEIGHT > 0:
        print(count)
        split_img = image[h - WINDOW_HEIGHT: h, ]
        nonzero = get_nonzero(split_img)
        if nonzero.size > 0:
            points = mean_shift(image_h,nonzero, count)
            results.append(points.tolist())
        h -= WINDOW_HEIGHT
        count += 1
    end_time = time.time()

    print("image shape", image.shape, "窗口高度", WINDOW_HEIGHT, "循环次数", count, "执行时间", end_time - start_time, "s")

    return results


#img = cv2.imread('3.jpeg', cv2.IMREAD_GRAYSCALE)
#img = np.zeros((20, 20), np.uint8)
#X = get_centers(img)
#print("------->\n", X)

#
# print("number of estimated clusters : %d" % n_clusters_)
#
# # #############################################################################
# # Plot result
# plt.figure(1)
# plt.clf()
#
# colors = cycle('r')
# for k, col in zip(range(n_clusters_), colors):
#     my_members = labels == k
#     cluster_center = cluster_centers[k]
#     plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
#     plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
#              markeredgecolor='k', markersize=14)
#     print(cluster_center[0], cluster_center[1], sep=",")
#
# print(img.shape)
# print(img.size)
# print(img.dtype)
#
# plt.imshow(img)
# plt.title('Estimated number of clusters: %d' % n_clusters_)
# # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# # plt.axis('off')
# plt.show()
