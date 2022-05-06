# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 11:10:44 2022

@author: rogierw
"""

import numpy as np
import matplotlib as plt

def indexContainingSubstring(the_list, substring):
    for i, s in enumerate(the_list):
        if substring in s:
              return i
    return -1

def closestNode(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    return np.argmin(abs(deltas))

def getScreenWidth():
    from win32api import GetSystemMetrics
    screen_width = GetSystemMetrics(0)
    return screen_width

def getScreenHeight():
    from win32api import GetSystemMetrics
    screen_height = GetSystemMetrics(1)
    return screen_height