import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import math
import time
from get_centers import *
from sklearn.cluster import MeanShift, estimate_bandwidth



def getLines(img, center_points,number_windows,window_half_width):
    img_h, img_w = img.shape[:2]

    window_height = math.floor(img_h / number_windows)

    # 所有白色的点的位置
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    lane_lines = []

    start = 0
    # 按行循环所有行
    for row in range(number_windows):
        lane_lines_points_x = 0
        lane_lines_points_y = 0
        last_offset = 0

        points = []

        for c_points in center_points:
            if len(c_points) > 0:
                if c_points[0][0] == row:
                    points = c_points
                    break

        # 点存在并且是第一次
        if len(points) > 0:

            if len(lane_lines) == 0:
                for i_point in range(len(points)):
                    lane_lines.append([])
                    lane_lines[i_point].append([points[i_point][1], points[i_point][2]])
            else:
                # 有点的行
                for line_index in range(len(lane_lines)):
                    lane_len = len(lane_lines[line_index])
                    if lane_len > 1:
                        last_offset = lane_lines[line_index][lane_len - 1][0] - lane_lines[line_index][lane_len - 2][0]

                        window_y_low = img_h - (row + 1) * window_height
                        window_y_high = img_h - row * window_height
                        window_x_low = int(lane_lines[line_index][lane_len - 1][0] - window_half_width + last_offset)
                        window_x_high = int(lane_lines[line_index][lane_len - 1][0] + window_half_width + last_offset)

                        # cv2.rectangle(img, (window_x_low, window_y_low), (window_x_high, window_y_high),
                        #                           (0, 255, 0), 2);

                        has_point = False

                        for point in points:
                            point_x = point[1]
                            point_y = point[2]
                            if point_x > window_x_low and point_x < window_x_high and point_y > window_y_low and point_y < window_y_high:
                                lane_lines[line_index].append([point_x, point_y])
                                points.remove(point)
                                has_point = True
                                continue
                        if has_point == False:
                            lane_lines_points_x = lane_lines[line_index][lane_len - 1][0] + last_offset
                            lane_lines_points_y = img_h - row * window_height - window_height / 2
                            lane_lines[line_index].append([lane_lines_points_x, lane_lines_points_y])

                    else:
                        window_y_low = img_h - (row + 1) * window_height
                        window_y_high = img_h - row * window_height
                        window_x_low = int(lane_lines[line_index][lane_len - 1][0] - window_half_width + last_offset)
                        window_x_high = int(lane_lines[line_index][lane_len - 1][0] + window_half_width + last_offset)

                        # cv2.rectangle(img, (window_x_low, window_y_low), (window_x_high, window_y_high),
                        #               (0, 255, 0), 2);

                        for point in points:
                            point_x = point[1]
                            point_y = point[2]
                            if int(point_x) > window_x_low and int(point_x) < window_x_high and int(
                                    point_y) > window_y_low and int(point_y) < window_y_high:
                                lane_lines[line_index].append([point_x, point_y])
                                points.remove(point)
                                continue

                # 如果有单独的点
                for point in points:
                    lane_lines.append([])
                    lane_lines[len(lane_lines) - 1].append([point[1], point[2]])

        else:
            # 已经有线，没有点的行，预估点
            for line in lane_lines:

                lane_len = len(line)
                last_offset = 0

                if lane_len > 1:
                    last_offset = line[lane_len - 1][0] - line[lane_len - 2][0]
                    lane_lines_points_x = line[lane_len - 1][0] + last_offset
                    lane_lines_points_y = img_h - row * window_height - window_height / 2
                    line.append([lane_lines_points_x, lane_lines_points_y])
                else:
                    lane_lines_points_x = line[lane_len - 1][0]
                    lane_lines_points_y = img_h - row * window_height - window_height / 2

                    lane_lines[line_index].append([lane_lines_points_x, lane_lines_points_y])



    for linee in lane_lines[1]:
        cv2.circle(img,
                   (int(linee[0]),
                    int(linee[1]))
                   , 5, (111, 111, 111), -1)
        
    plt.figure("dog")
    plt.imshow(img)
    plt.show()
    return lane_lines


def segment(img):
    points = get_centers(img)
    return getLines(img, points,25,50)

#img = cv2.imread("3.jpeg", cv2.IMREAD_GRAYSCALE)
#segment(img)



