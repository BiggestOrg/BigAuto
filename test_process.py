import cv2
import math
import numpy as np
from collections import deque


lines = []
tmpline = [(0,0),(0,0)]
avgindex = deque()
mid_point_queue = deque()

# param
W = 1920
H = 1080
K_SIMILAR = 0.01
B_SIMILAR = 100
LEFT_RULER = [0, int(H / 3), int(W / 2), H]
RIGHT_RULER = [W, int(H / 3), int(W / 2), H]
MID_POINT_RERION = [int(W / 3), int(H / 4), int(2 * W / 3), int(2 * H / 3)]
MID_POINT_R = 30
MID_POINT_QUEUE_LENGTH = 5


def classify_lines(lines):
    """join similar houghlines by k,b"""
    class_lines = []

    for i in range(len(lines)):
        l = lines[i][0]
        k = (l[3] - l[1]) / (l[2] - l[0])
        b = l[1] - k * l[0]

        flag = 0
        for j in range(len(class_lines)):
            cl = class_lines[j]
            if abs(cl[0] - k) > K_SIMILAR:
                continue
            if abs(cl[1] - b) > B_SIMILAR:
                continue
            k_mean = (cl[0] * cl[2] + k) / (cl[2] + 1)
            b_mean = (cl[1] * cl[2] + b) / (cl[2] + 1)
            class_lines[j] = [k_mean, b_mean, cl[2] + 1]
            flag =1
        if flag:
            continue
        else:
            class_lines.append([k, b, 1])

    return class_lines


def calc_edge_point(k, b):
    """calc 2 edge points in image(w,h) """
    pt = []
    if (b >= 0) & (b <= H):
        pt.append((0, int(b)))

    y = k * W + b
    if (y >= 0) & (y <= H):
        pt.append((int(W), int(y)))

    x = -b / k
    if (x >=0) & (x <= W):
        pt.append((int(x), 0))

    x = (H - b) / k
    if (x >=0) & (x <= W):
        pt.append((int(x), int(H)))

    return pt


def calc_k_b(line):
    """get k , b from two point line [x0,y0,x1,y1]"""
    k = (line[3] - line[1]) / (line[2] - line[0])
    b = line[1] - k * line[0]
    return k, b


def calc_distance2(x0, y0, x1, y1):
    ''' calc two point's distance'''
    return (x0 - x1) ** 2 + (y0 - y1) ** 2


def calc_cross_point(k1, b1, k2, b2, line1 = None, line2 = None):
    """ get cross point from two line """
    if k1 == k2:
        return None, None
    x = (b1 - b2) / (k2 - k1)
    y = (b1 * k2 - b2 * k1) / (k2 - k1)

    if line1 is not None:
        if not (min(line1[0], line1[2]) < x < max(line1[0], line1[2])):
            return None, None
        if not (min(line1[1], line1[3]) < y < max(line1[1], line1[3])):
            return None

    if line2 is not None:
        if not (min(line2[0], line2[2]) < x < max(line2[0], line2[2])):
            return None, None
        if not (min(line2[1], line2[3]) < y < max(line2[1], line2[3])):
            return None, None

    return x, y


def calc_mid_point(points):
    """ calc 1 point from many points"""

    if len(mid_point_queue) < 2:
        pt = np.mean(points, axis=0)
        mid_point_queue.append(pt)
        return pt

    filter_points = []
    pt_avg = np.mean(mid_point_queue, axis=0)
    # pt_var = np.var(mid_point_queue, axis=0)
    # pt_var = [30, 30]
    # print(pt_avg, pt_var)
    for p in points:
        # TODO: sometimes there is a bug
        if ((p[0] - pt_avg[0]) ** 2 + (p[1] - pt_avg[1]) ** 2) > MID_POINT_R ** 2:
            continue
        filter_points.append(p)

    if len(filter_points) > 1:
        pt = np.mean(filter_points, axis=0)
        mid_point_queue.append(pt)

    if len(mid_point_queue) > MID_POINT_QUEUE_LENGTH:
        mid_point_queue.popleft()

    pt_avg = np.mean(mid_point_queue, axis=0)

    return pt_avg


def filter_lines(lines):
    """ del noisy lines by Ruler """

    # left  ruler y = (h / w) * x + (h / 2)
    # right ruler y = (-2 * h / w) * x + (3 * h / 2)
    ruler_left_k, ruler_left_b = calc_k_b(LEFT_RULER)
    ruler_right_k,ruler_right_b= calc_k_b(RIGHT_RULER)
    ret_lines = []
    cross_pt = []
    for ll in lines:
        l = ll[0]
        k, b = calc_k_b(l)

        flag = 0
        # cross dot on left ruler
        x0, y0 = calc_cross_point(k, b, ruler_left_k, ruler_left_b, LEFT_RULER)
        if x0 is not None:
            flag += 1
            cross_pt.append((int(x0), int(y0)))

        # cross point on right ruler
        x1, y1 = calc_cross_point(k, b, ruler_right_k, ruler_right_b, RIGHT_RULER)
        if x1 is not None:
            flag +=1
            cross_pt.append((int(x1), int(y1)))

        # print("filter lines param : ", k, b, x0, y0, x1, y1, flag)
        if flag == 1:
            ret_lines.append(ll)

    return ret_lines, cross_pt


def filter_lines2(lines):
    """ calc cross point and filter error point, then find max points circle,
     :param lines:joined lines [k, b] """
    cross_pt = []
    length = len(lines)

    # find all cross if it's in MID_POINT_REGION
    for i in range(length):
        for j in range(i+1, length):
            line1 = lines[i]
            line2 = lines[j]
            x, y = calc_cross_point(line1[0], line1[1], line2[0], line2[1])
            if x is None:
                continue

            if not (MID_POINT_RERION[0] < x < MID_POINT_RERION[2]):
                continue

            if not (MID_POINT_RERION[1] < y < MID_POINT_RERION[2]):
                continue

            cross_pt.append([x, y])

    # find max clustering point
    length_pt = len(cross_pt)
    mid_point = []
    max_num = 0
    # TODO: preformance is bad
    for i in range(length_pt):
        x0, y0 = cross_pt[i][0], cross_pt[i][1]
        min_point_x = x0
        min_point_y = y0
        num = 1
        for j in range(length_pt):
            x1, y1 = cross_pt[j][0], cross_pt[j][1]
            if calc_distance2(x0, y0, x1, y1) < MID_POINT_R ** 2:
                mid_point_x = (min_point_x * num + x1) / (num + 1)
                mid_point_y = (min_point_y * num + y1) / (num + 1)
                num += 1

            if num > max_num:
                mid_point = [[mid_point_x, mid_point_y]]
                max_num = num
            elif num == max_num:
                mid_point.append([mid_point_x, mid_point_y])
    return mid_point, cross_pt


def find_max_index(histogram, width = 100):
    """ find max index from histogram(array) """
    arr_maxindex = []
    i = width
    while i <= len(histogram) - width:
        tmp_his = histogram[i:i+width]
        if sum(tmp_his) == 0:
            i += width
            continue
        tmp_max_index = np.argwhere(tmp_his == np.max(tmp_his))
        max_index = tmp_max_index[0][0] + i
        if histogram[max_index] >= np.max(histogram[max_index-width:max_index+width]):
            arr_maxindex.append(max_index)
            i = max_index + width
        else:
            i = max_index
        i += 1

    return arr_maxindex


def slide_window(birdeye, win_width=30, win_height=30, find_max_index_width=80, find_max_index_height=300, birdeye_debug=None):
    """ slice lane by slice window
    :param birdeye: gray birdeye of lane image
    :return lanes: array of lane
    """
    ret, birdeye_bin = cv2.threshold(birdeye, 200, 255, cv2.THRESH_BINARY)
    his = np.sum(birdeye[-find_max_index_height:, ] / 255, axis=0)
    arr_maxindex = find_max_index(his, find_max_index_width)

    # slide window
    lanes = []
    for start_index in arr_maxindex:
        cur_win_mid = start_index
        offset = 0
        lane = Lane()
        for i in range(H, 0, -win_height):
            cur_win_left = cur_win_mid - win_width
            cur_win_right = cur_win_mid + win_width
            cur_win_top = i - win_height
            cur_win_bottom = i
            cur_win = birdeye_bin[cur_win_top:cur_win_bottom, cur_win_left: cur_win_right]
            cur_win_his = np.sum(cur_win // 255, axis=0)
            print("cur_win_his = ", cur_win_his)

            cur_win_max_index = win_width  # default value for max_index
            if sum(cur_win_his) > 0:
                # mean of all point x
                cur_win_max_index = int(np.mean(cur_win.nonzero()[1]))
            cur_max_index = cur_win_left + cur_win_max_index

            lane.add_point(cur_max_index, i, sum(cur_win_his))
            # k, b = lane.ployfit2(5)
            # print("k, b, max_index = ", k, b, cur_win_max_index, np.argmax(cur_win_his), cur_max_index)

            if sum(cur_win_his) == 0:
                offset = offset * .99
            else:
                offset = cur_max_index - cur_win_mid
            print(offset)

            cur_win_mid = int(cur_win_mid + offset)

            # debug
            birdeye_debug = cv2.rectangle(birdeye_debug, (cur_win_left, cur_win_top), (cur_win_right, cur_win_bottom),
                                         (255, 0, 0))
            if sum(cur_win_his) == 0:
                    birdeye_debug = cv2.circle(birdeye_debug, (cur_max_index, i), 3, (255, 255, 0))
            else:
                    birdeye_debug = cv2.circle(birdeye_debug, (cur_max_index, i), 3, (0, 255, 255))

            '''
            if k!= 0:
                y0 = i - slide_win_width
                x0 = int((y0 - b) / k)
                y1 = i + slide_win_width
                x1 = int((y1 - b) / k)
                birdeye_draw = cv2.line(birdeye_draw, (x0, y0), (x1, y1), (0, 0, 255))
            '''
            # cv2.imshow("birdeye2", birdeye_draw)
            # cv2.waitKey(0)
        lanes.append(lane)

    return lanes


def main():
    cap = cv2.VideoCapture("output/output10.avi")
    v = cv2.VideoCapture("data/test10.mp4")
    if not cap.isOpened():
        return

    while True:
        ret, frame = cap.read()
        r, f = v.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calc lane

        draw = frame.copy()
        gray_canny = cv2.Canny(gray, threshold1=50, threshold2=200, edges=1) # TODO: use LSD
        houghlines_org = cv2.HoughLinesP(gray_canny, 1, 3.1415 / 180, threshold=100, minLineLength=50, maxLineGap=1000)
        if houghlines_org is None:
            print("no houghline found")
            continue
        houghlines, debug_pt = filter_lines(houghlines_org)
        join_lines = classify_lines(houghlines)
        mid_points, debug_cross_pt = filter_lines2(join_lines)
        mid_pt = calc_mid_point(mid_points)

        #region debug draw
        # draw joined line green
        for l in join_lines:
            edge_pt = calc_edge_point(l[0], l[1])
            draw = cv2.line(draw, edge_pt[0], edge_pt[1], (0,255,0), thickness=2)

        # draw hough line red
        for l in houghlines_org:
            if len(l) > 1:
                print(l)
            l = l[0]
            draw = cv2.line(draw, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3)

        # draw crossed ruler debug point
        for p in debug_pt:
            draw = cv2.circle(draw, p, 5, (255, 0, 0), thickness=3)

        # draw mid point debug point
        for p in debug_cross_pt:
            draw = cv2.circle(draw, (int(p[0]), int(p[1])), 5, (255, 0, 0), thickness=1)

        # draw mid points
        for p in mid_points:
            draw = cv2.circle(draw, (int(p[0]), int(p[1])), 5, (255, 255, 255), thickness=1)

        # draw mid point
        draw = cv2.circle(draw, (int(mid_pt[0]), int(mid_pt[1])), 7, (0, 255, 255), thickness=4)

        # draw debug ruler
        draw = cv2.line(draw, (LEFT_RULER[0], LEFT_RULER[1]), (LEFT_RULER[2], LEFT_RULER[3]), (255, 255, 0), thickness=2)
        draw = cv2.line(draw, (RIGHT_RULER[0],RIGHT_RULER[1]), (RIGHT_RULER[2], RIGHT_RULER[3]), (255, 255, 0), thickness=2)

        # draw mid point ruler
        draw = cv2.line(draw, (MID_POINT_RERION[0], MID_POINT_RERION[1]), (MID_POINT_RERION[0], MID_POINT_RERION[3]), (255,255,0))
        draw = cv2.line(draw, (MID_POINT_RERION[0], MID_POINT_RERION[1]), (MID_POINT_RERION[2], MID_POINT_RERION[1]), (255,255,0))
        draw = cv2.line(draw, (MID_POINT_RERION[0], MID_POINT_RERION[3]), (MID_POINT_RERION[2], MID_POINT_RERION[3]), (255,255,0))
        draw = cv2.line(draw, (MID_POINT_RERION[2], MID_POINT_RERION[1]), (MID_POINT_RERION[2], MID_POINT_RERION[3]), (255,255,0))

        # draw mid line
        draw = cv2.line(draw, (int(W / 2), 0), (int(W/2), H), (255,255,255))
        draw = cv2.line(draw, (0, int(H/2)), (W, int(H/2)), (255,255,255))

        cv2.imshow("debug", draw)
        #endregion debug draw

        pt1 = np.array([[0,mid_pt[1]+10], [1920,mid_pt[1]+10], [0,1080], [1920,1080]], dtype='float32')
        pt2 = np.array([[0,0], [1920,0], [(1920) * 0.48,1080],[(1920) * 0.52,1080]], dtype='float32')
        M = cv2.getPerspectiveTransform(pt1, pt2)

        t = (mid_pt[0] - W / 2) / (mid_pt[1] - H) / 2 # TODO : why /2 ?
        Mat2_1 = np.array([[1, 0, -W / 2], [0, 1, -H], [0, 0, 1]], dtype='float32') # 平移下边界中点到原点
        Mat2_2 = np.array([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1]], dtype='float32') # 旋转t度
        Mat2_3 = np.array([[1, 0, W/2], [0, 1, H], [0, 0, 1]], dtype='float32') # 平移原点到下边界中点

        Mat22 = np.dot(Mat2_2, Mat2_1)
        Mat2 = np.dot(Mat2_3, Mat22)

        Mat3 = np.array([[3, 0, -W], [0, 1, 0], [0, 0, 1]], dtype="float32")

        # core code
        MatAll = np.dot(Mat2, M)
        MatAll = np.dot(Mat3, MatAll)

        birdeye_draw = cv2.warpPerspective(frame, MatAll, (1920, 1080))
        birdeye = cv2.warpPerspective(gray, MatAll, (1920,1080))

        lanes = slide_window(birdeye, birdeye_debug=birdeye_draw)

        cv2.imshow("birdeye", birdeye)

        cv2.imshow("birdeye_debug", birdeye_draw)

        cv2.imshow("frame", f)

        k = cv2.waitKey(0)
        if k == ord('q'):
            break
    pass


class Lane:
    def __init__(self):
        self.points = []
        pass

    def add_point(self, x, y, num=1):
        self.points.append([x, y, num])


if __name__ == '__main__':
    main()
