"""
Created on Wed Feb 04 01:35:36 2026
@author: santaro



"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from pathlib import Path

class Circle:
    def __init__(self, name="circle", center=np.zeros(2), r=1, m=1):
        self.name = name
        self.center = center
        self.r = r
        self.m = m

class Line:
    def __init__(self, name="line", a=1, b=1, c=0):
        self.name = name
        self.a = a
        self.b = b
        self.c = c


class GeomContact:
    def __init__(self, object0, object1, name=""):
        self.name = name
        self.object0 = object0
        self.object1 = object1
        self.d = None

    def distance_c2l(self):
        if isinstance(self.object0, Circle):
            c = self.object0
        elif isinstance(self.object1, Circle):
            c = self.object1
        if isinstance(self.object0, Line):
            l = self.object0
        elif isinstance(self.object1, Line):
            l = self.object1
        if not c or not l:
            raise ValueError(f"distance_c2l requires one Circle and One Line, object1: {type(self.object0)}, object2: {type(self.object1)}")

        self.d = np.abs(l.a * c.center[0] + l.b * c.center[1] + l.c) / np.sqrt(l.a**2 + l.b**2) - c.r
        return self.d


class ContactForce:
    def __init__(self):
        pass


if __name__ == "__main__":
    print("---- run ----")

    l1 = Line()
    c1 = Circle()
    # c1 = Circle()

    geom = GeomContact(l1, c1)
    geom.distance_c2l()
    print(geom.d)


