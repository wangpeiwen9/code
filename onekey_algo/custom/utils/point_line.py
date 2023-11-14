# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2022/12/26
# Forum: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2022 All Rights Reserved.
import copy

import numpy as np


class Point:
    x = 0
    y = 0

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __str__(self):
        return f"x:{self.x}, y:{self.y}"


class Line:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def __str__(self):
        return f"p1:{self.p1}, p2:{self.p2}"


def get_line_para(line):
    line.a = line.p1.y - line.p2.y
    line.b = line.p2.x - line.p1.x
    line.c = line.p1.x * line.p2.y - line.p2.x * line.p1.y


def get_cross_point(l1, l2):
    get_line_para(l1)
    get_line_para(l2)
    d = l1.a * l2.c - l2.a * l1.b
    p = Point()
    p.x = (l1.b * l2.c - l2.b * l1.c) * 1.0 / d
    p.y = (l1.c * l2.a - l2.c * l1.a) * 1.0 / d
    return p


def get_cross_angle(l1, l2):
    arr_0 = np.array([(l1.p2.x - l1.p1.x), (l1.p2.y - l1.p1.y)])
    arr_1 = np.array([(l2.p2.x - l2.p1.x), (l2.p2.y - l2.p1.y)])
    cos_value = (float(arr_0.dot(arr_1)) / (np.sqrt(arr_0.dot(arr_0)) * np.sqrt(arr_1.dot(arr_1))))
    return np.arccos(cos_value) * (180 / np.pi)


def get_distance_point2line(point: Point, line: Line):
    """
    Args:
        point: [x0, y0]
        line: [x1, y1, x2, y2]
    """
    line_point1, line_point2 = np.array([line.p1.x, line.p1.y]), np.array([line.p2.x, line.p2.y])
    vec1 = line_point1 - [point.x, point.y]
    vec2 = line_point2 - [point.x, point.y]
    distance = np.abs(np.cross(vec1, vec2)) / np.linalg.norm(line_point1 - line_point2)
    return distance


def get_distance_point(p1: Point, p2: Point):
    """
    获取两点之间的距离
    Args:
        p1:
        p2:

    Returns:

    """
    return ((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2) ** 0.5


def get_parallel_line(point: Point, line: Line):
    gradient = (line.p1.y - line.p2.y) / (line.p1.x - line.p2.x + 1e-6)
    return Line(copy.deepcopy(point), Point(point.x + 1, point.y + gradient))


def get_foot(point: Point, line: Line):
    start_x, start_y = line.p1.x, line.p1.y
    end_x, end_y = line.p2.x, line.p2.y
    pa_x, pa_y = point.x, point.y

    p_foot = [0, 0]
    k = (end_y - start_y) * 1.0 / (end_x - start_x)
    a = k
    b = -1.0
    c = start_y - k * start_x
    p_foot_x = int((b * b * pa_x - a * b * pa_y - a * c) / (a * a + b * b))
    p_foot_y = int((a * a * pa_y - a * b * pa_x - b * c) / (a * a + b * b))

    return Point(p_foot_x, p_foot_y)


if __name__ == '__main__':
    p1 = Point(1, 1)
    p2 = Point(3, 3)
    line1 = Line(p1, p2)

    p3 = Point(0, 4)
    p4 = Point(4, 0)
    line2 = Line(p3, p4)
    print(get_parallel_line(p3, line1))
    print(get_cross_angle(line1, get_parallel_line(p3, line1)))
    print(get_distance_point2line(p1, line2))
    print(get_foot(p1, line2))
    # print(Pc.x, Pc.y)
