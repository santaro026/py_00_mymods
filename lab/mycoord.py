# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 09:53:21 2024
@author: santaro

"""

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from matplotlib import colormaps

"""
the shape of points is aligned to:
    (number of points, number of dimension)
    (number of frames, number of points, number of dimension)

"""

class CoordTransformer2D:
    def __init__(self, name='', local_origin=np.zeros(2), theta=0):
        self.name = name
        self.local_origin = np.asarray(local_origin)
        self.theta = theta
        self.R = CoordTransformer2D.make_rotation_matrix(self.theta)
        self.R_inv = CoordTransformer2D.make_rotation_matrix(-self.theta)

    @staticmethod
    def make_rotation_matrix(theta):
        theta = np.atleast_1d(theta)
        R = np.stack([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]
                      ], axis=0).transpose(2, 0, 1)
        return R

    @staticmethod
    def rotate(p, theta, center=np.zeros(2)):
        p = np.atleast_2d(p)
        R = CoordTransformer2D.make_rotation_matrix(theta)
        p_translated = p - center
        p_rotated = (R @ p_translated[:, :, np.newaxis]).squeeze()
        p_result = p_rotated + center
        return p_result

    def transform_coord(self, p, towhich='tolocal'):
        p = np.atleast_2d(p)
        if towhich == 'tolocal':
            p_translated = p - self.local_origin
            p_transformed = (self.R_inv @ p_translated[:, :, np.newaxis]).squeeze()
        elif towhich == 'toglobal':
            p_rotated = (self.R @ p[:, :, np.newaxis]).squeeze()
            p_transformed = p_rotated + self.local_origin
        else:
            raise ValueError(f'Unknown transform direction : {towhich}')
        return p_transformed

    def polar_coord(self, p, towhich='tocartesian'):
        p = np.atleast_2d(p)
        if towhich == 'topolar':
            r = np.linalg.norm(p, axis=1)
            theta = np.arctan2(p[:, 1], p[:, 0])
            p_transformed = np.stack([r, theta], axis=1)
        elif towhich == 'tocartesian':
            x = p[:, 0] * np.cos(p[:, 1])
            y = p[:, 0] * np.sin(p[:, 1])
            p_transformed = np.stack([x, y], axis=1)
        else:
            raise ValueError(f'Unknown transform direction : {towhich}')
        return p_transformed

class CoordTransformer3D:
    def __init__(self, name='', local_origin=np.zeros(3), euler_angles=np.zeros(3), rot_order='zyx'):
        self.name = name
        self.local_origin = local_origin
        self.rot_order = rot_order
        self.euler_angles = np.atleast_2d(euler_angles) # ordered according to rotating sequence
        self.Rx, self.Ry, self.Rz = CoordTransformer3D.make_rotation_matrix(thetax=self.euler_angles[:, 0], thetay=self.euler_angles[:, 1], thetaz=self.euler_angles[:, 2])
        self.Rx_inv, self.Ry_inv, self.Rz_inv = CoordTransformer3D.make_rotation_matrix(thetax=-self.euler_angles[:, 0], thetay=-self.euler_angles[:, 1], thetaz=-self.euler_angles[:, 2])
        if rot_order == 'zyx':
            self.rotM = self.Rz @ self.Ry @ self.Rx
            self.rotM_inv = self.Rz_inv @ self.Ry_inv @ self.Rx_inv
        elif rot_order == 'xyz':
            self.rotM = self.Rx @ self.Ry @ self.Rz
            self.rotM_inv = self.Rx_inv @ self.Ry_inv @ self.Rz_inv

    @staticmethod
    def make_rotation_matrix(thetax, thetay, thetaz):
        thetax, thetay, thetaz = np.atleast_1d(thetax), np.atleast_1d(thetay), np.atleast_1d(thetaz)
        num_frames = len(thetax)
        zero = np.zeros(num_frames)
        one = np.ones(num_frames)
        Rx = np.stack([[one, zero, zero],
                    [zero, np.cos(thetax), -np.sin(thetax)],
                    [zero, np.sin(thetax), np.cos(thetax)]], axis=0).transpose(2, 0, 1)
        Ry = np.stack([[np.cos(thetay), zero, np.sin(thetay)],
                    [zero, one, zero],
                    [-np.sin(thetay), zero, np.cos(thetay)]], axis=0).transpose(2, 0, 1)
        Rz = np.stack([[np.cos(thetaz), -np.sin(thetaz), zero],
                    [np.sin(thetaz), np.cos(thetaz), zero],
                    [zero, zero, one]], axis=0).transpose(2, 0, 1)
        return Rx, Ry, Rz

    @staticmethod
    def rotate_euler(p, euler_angles, rot_order, center=np.zeros(3)):
        p = np.atleast_2d(p)
        num_frames = len(p)
        euler_angles = np.atleast_2d(euler_angles)
        thetax = np.full(num_frames, euler_angles[:, 0]) if euler_angles.shape[0] <= 1 else euler_angles[:, 0]
        thetay = np.full(num_frames, euler_angles[:, 1]) if euler_angles.shape[0] <= 1 else euler_angles[:, 1]
        thetaz = np.full(num_frames, euler_angles[:, 2]) if euler_angles.shape[0] <= 1 else euler_angles[:, 2]
        Rx, Ry, Rz = CoordTransformer3D.make_rotation_matrix(thetax=thetax, thetay=thetay, thetaz=thetaz)
        if not (num_frames == 1 or len(euler_angles) == 1 or len(euler_angles) == num_frames):
            raise ValueError(f'**** dimensions of p and euler_angles doesnt match\np: {p.shape}\nrotation Rx: {Rx.shape}')
        if rot_order == 'zyx':
            p_translated = p - center
            p_rotated = (Rz @ Ry @ Rx @ p_translated[:, :, np.newaxis]).squeeze()
            p_transformed = p_rotated + center
        elif rot_order == 'xyz':
            p_translated = p - center
            p_rotated = (Rx @ Ry @ Rz @ p_translated[:, :, np.newaxis]).squeeze()
            p_transformed = p_rotated + center
        else:
            raise ValueError(f'**** rot_order parameter is invalid: {rot_order}.')
        return p_transformed

    def transform_coord(self, p, towhich='tolocal'):
        p = np.atleast_2d(p)
        if towhich == 'tolocal':
            p_translated = p - self.local_origin
            p_transformed = (self.rotM_inv @ p_translated[:, :, np.newaxis]).squeeze()
        elif towhich == 'toglobal':
            p_rotated = (self.rotM @ p[:, :, np.newaxis]).squeeze()
            p_transformed = p_rotated + self.local_origin
        else:
            raise ValueError(f'****error: towhich parameter is invalid: {towhich}.')
        return p_transformed

def visualize_points(ps, colors=['r', 'b', 'y', 'g', 'c']*20, markersize=4, center=np.zeros(2), xyrange=2, centermarks=None, closeauto=4):
    fig, ax = plt.subplots(figsize=(12, 12))
    xylim = center + xyrange * np.array([-1, 1])
    ax.set(xlim=xylim, ylim=xylim)
    ax.set_aspect(1)
    ax.grid()
    ax.axhline(y=0, xmin=0.2, xmax=0.8, c='k', ls='--', lw=1)
    ax.axvline(x=0, ymin=0.2, ymax=0.8, c='k', ls='--', lw=1)
    if centermarks is not None:
        for _x, _y in centermarks:
            ax.plot([_x-xyrange*0.1, _x+xyrange*0.1], [_y, _y], c='k', ls='--', lw=1)
            ax.plot([_x, _x], [_y-xyrange*0.1, _y+xyrange*0.1], c='k', ls='--', lw=1)
    for _c, _p in enumerate(ps):
        # _p = _p[np.newaxis, :] if np.ndim(_p) == 1 else _p
        _p = np.atleast_2d(_p)
        ax.scatter(_p[:, 0], _p[:, 1], c=colors[_c], s=markersize)
    plt.show(block=False)
    if closeauto:
        plt.pause(closeauto)
        plt.close()



class BearingGeometoryGalculator:
    def __init__(self, p_Aring, p_Bring, p_balls, p_cage):
        self.p_Aring = p_Aring
        self.p_Bring = p_Bring
        self.p_balls = p_balls
        self.p_cage = p_cage

    def calc_ball_distribution(p_C):
        azms = np.arctan2(-p_C[:, 1, :], p_C[:, 2, :]) # arctan2(y-coord, x-coord), anlysis plane is YZ
        _azms = np.vstack([azms[1:], azms[0]+2*np.pi])
        dazms = _azms - azms
        dazms = np.where(dazms<0, dazms+2*np.pi, dazms)
        dazms = np.where(dazms>2*np.pi, dazms-2*np.pi, dazms)
        dazms_deg = np.degrees(dazms)
        return dazms_deg

def calc_ball_distribution(p_C):
    azms = np.arctan2(-p_C[:, 1, :], p_C[:, 2, :]) # arctan2(y-coord, x-coord), anlysis plane is YZ
    _azms = np.vstack([azms[1:], azms[0]+2*np.pi])
    dazms = _azms - azms
    dazms = np.where(dazms<0, dazms+2*np.pi, dazms)
    dazms = np.where(dazms>2*np.pi, dazms-2*np.pi, dazms)
    dazms_deg = np.degrees(dazms)
    return dazms_deg





if __name__ == '__main__':
    print('----- test -----\n')

    #### 2d
    # num = 100
    # theta = np.linspace(0, 2*np.pi, num)
    # transformer2d = CoordTransformer2D(name='sample', local_origin=np.zeros(2), theta=np.radians(30))
    # p = np.vstack([np.arange(num), np.zeros(num)]).T
    # p = transformer2d.polar_coord(p, towhich='topolar')
    # p2 = transformer2d.transform_coord(p, towhich='toglobal')
    # print(f"p.shape: {p.shape}")
    # visualize_points([p, p2], xyrange=100)

    #### 3d
    num = 100
    euler_angles = np.vstack([np.zeros(num), np.zeros(num), np.linspace(0, 1*np.pi, num)]).T
    # transformer3d = CoordTransformer3D(name='sample_3d', local_origin=np.zeros(3), euler_angles=euler_angles)
    p = np.array([20, 0, 0])
    # euler_angles = np.array([[0, 0, 1], [0, 0, 2]])
    p2 = CoordTransformer3D.rotate_euler(p, euler_angles=euler_angles, rot_order='zyx')
    # p2 = transformer3d.transform_coord(p2)
    visualize_points([p, p2], xyrange=100)



