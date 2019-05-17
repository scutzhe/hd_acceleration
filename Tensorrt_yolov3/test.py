#!/usr/bin/python3
# *_* coding: utf-8 *_*
# @Author: samon
# @Email: samonsix@163.com
# @IDE: PyCharm
# @File: test.py
# @Date: 19-5-16 下午3:38
# @Descr:

#coding=utf-8
from ctypes import cdll
import ctypes




# class message_format(Structure):
#     _fields_=[("topic", c_byte*MQSTRLENGTH),
#               ("tag", c_byte*MQSTRLENGTH),
#               ("message_id", c_byte*MQSTRLENGTH),
#               ("content", c_byte*MQCONTENTLENGTH)]
#
#
# def parseMessage(input):
#     mq_topic = create_string_buffer(MQSTRLENGTH)
#     mq_tag = create_string_buffer(MQSTRLENGTH)
#     mq_id = create_string_buffer(MQSTRLENGTH)
#     mq_content = create_string_buffer(MQCONTENTLENGTH)
#
#     memmove(mq_topic, byref(input.topic), MQSTRLENGTH)
#     memmove(mq_tag, byref(input.tag), MQSTRLENGTH)
#     memmove(mq_id, byref(input.message_id), MQSTRLENGTH)
#     memmove(mq_content, byref(input.content), MQCONTENTLENGTH)
#
#     return mq_topic.value, mq_tag.value, mq_id.value, mq_content.value


dll = cdll.LoadLibrary('install/libyolov3.so')


class StructPointer(ctypes.Structure):
    _fields_ = [("num", ctypes.c_int),
                ("location", ctypes.c_int * 400)]


dll.yolov3.restype = ctypes.POINTER(StructPointer)
p = dll.yolov3(b"08.jpg")

print(p.contents.num)
index = 0
all_point = []
for i in range(p.contents.num):
    point = [p.contents.location[index], p.contents.location[index+1],
             p.contents.location[index + 2], p.contents.location[index+3]]
    index += 4
    all_point.append(point)







