#!/usr/bin/python3
# *_* coding: utf-8 *_*
# @Author: dylen
# @Email: dylenzheng@gmail.com
# @IDE: PyCharm
# @File: tensorrt.py
# @Date: 19-5-16 下午3:38
# @Descr:

#coding=utf-8
from ctypes import cdll
import ctypes

def tensorrt(img_path):
	dll = cdll.LoadLibrary('./libyolov3.so')
	class StructPointer(ctypes.Structure):
	    _fields_ = [("num", ctypes.c_int),
	                ("location", ctypes.c_int * 400)]

	dll.yolov3.restype = ctypes.POINTER(StructPointer)

	# b表示byte类型，引起注意
	img_path = img_path.encode('utf-8')
	p = dll.yolov3(img_path)
	#print(p.contents.num)
	index = 0
	all_point = []
	for i in range(p.contents.num):
	    point = [p.contents.location[index], p.contents.location[index+1],
	             p.contents.location[index + 2], p.contents.location[index+3]]
	    index += 4
	    all_point.append(point)

	return all_point

img_path = '08.jpg'
all_point = tensorrt(img_path)
for i in all_point:
	print(i)