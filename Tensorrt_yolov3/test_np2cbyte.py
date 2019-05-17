#!/usr/bin/python3
# *_* coding: utf-8 *_*
# @Author: samon
# @Email: samonsix@163.com
# @IDE: PyCharm
# @File: test_np2cbyte.py
# @Date: 19-5-16 下午4:38
# @Descr:
from ctypes import *
import ctypes
import numpy
import cv2


dll = cdll.LoadLibrary('install/libyolov3.so')

n = 100
one_R = [1 for r in range(n)]
R = [one_R for rr in range(n)]
one_G = [128 for g in range(n)]
G = [one_G for gg in range(n)]
one_B = [256 for b in range(n)]
B = [one_B for bb in range(n)]

RGB = numpy.zeros((n, n, 3), dtype=c_uint8)
RGB[:, :, 0] = B      # B
RGB[:, :, 1] = G      # G
RGB[:, :, 2] = R      # R
img = RGB

# rows, cols, channel = img.shape

# dataptr = img.ctypes.data_as(ctypes.c_char_p)
# dll.test(dataptr, n, n, 3)


# 测试方法2, 直接从图片获取numpy
img = cv2.imread('08.jpg')

# img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
rows, cols, chanel = img.shape

dataptr = img.ctypes.data_as(ctypes.c_char_p)


class StructPointer(ctypes.Structure):
    _fields_ = [("num", ctypes.c_int),
                ("location", ctypes.c_int * 400)]

dll.yolov3.restype = ctypes.POINTER(StructPointer)

# 数据,row, col, bit(位数，1表示灰度图，3表示rgb图)
import time
print("start")
dll.test(dataptr, rows, cols, chanel)
time.sleep(1)
dll.yolov3(dataptr, rows, cols, chanel)
st = time.time()
dll.yolov3(dataptr, rows, cols, chanel)
dll.yolov3(dataptr, rows, cols, chanel)
dll.yolov3(dataptr, rows, cols, chanel)
dll.yolov3(dataptr, rows, cols, chanel)
dll.yolov3(dataptr, rows, cols, chanel)
dll.yolov3(dataptr, rows, cols, chanel)
dll.yolov3(dataptr, rows, cols, chanel)
dll.yolov3(dataptr, rows, cols, chanel)
dll.yolov3(dataptr, rows, cols, chanel)
p = dll.yolov3(dataptr, rows, cols, chanel)
all_point = []
index = 0
for i in range(p.contents.num):
    point = [p.contents.location[index], p.contents.location[index+1],
             p.contents.location[index + 2], p.contents.location[index+3]]
    index += 4
    all_point.append(point)
print(all_point)
print(time.time()-st)








