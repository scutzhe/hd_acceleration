#!/usr/bin/python3
# *_* coding: utf-8 *_*
# @Author: samn-dylen
# @Email: samonsix@162.com_dylenzheng@gmail.com
# @IDE: PyCharm
# @File: tensorrt.py
# @Date: 19-5-16 下午4:38
# @Descr:
from ctypes import *
import ctypes
import numpy
import cv2

dll = cdll.LoadLibrary('./libyolov3.so')

def tensorrt(image):
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
	
    rows, cols, channel = image.shape
    dataptr = image.ctypes.data_as(ctypes.c_char_p)
    class StructPointer(ctypes.Structure):
    	_fields_ = [("num", ctypes.c_int),("location", ctypes.c_int * 400)]

    dll.yolov3.restype = ctypes.POINTER(StructPointer)
    p = dll.yolov3(dataptr, rows, cols, channel)

    #print(p.contents.num)
    index = 0
    all_point = []
    for i in range(p.contents.num):
        point = [p.contents.location[index], p.contents.location[index+1],p.contents.location[index + 2], p.contents.location[index+3]]
        index += 4
        all_point.append(point)
	
    return all_point

img = cv2.imread('08.jpg')
all_point = tensorrt(img)
for i in all_point:
	print(i)
