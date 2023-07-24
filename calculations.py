#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
@FILE      : calculations.py
@TIME      : 2023/07/24 15:18:50
@AUTHOR    : wangyu / NMC
@VERSION   : 1.0
@DESC      : 提供各种物理量的计算
'''


### to import parent dir files ###
# import os, sys
### this is for jupyter notebook ###
#current_folder = globals()['_dh'][0]
#parentdir = os.path.dirname(current_folder)
### this is for normal python file ###
#parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#sys.path.insert(0,parentdir)

from datetime import datetime, timedelta

import cv2 
import numpy as np 
import utils.DataIO as DIO


def gama(idate, fn_ter, fn_gama, fn_tplvl, lvls, 
                   nor, sou, wst, est, dlon, dlat): 
    """
    根据网格上空的 气压层 温度数据，计算温度的垂直递减率 γ
    注意这里的 γ = val*1e3
    因此实际使用计算是，需要 *1e-3
    """
    # 首先需要实际地形，然后基于地形计算对应高度的温度垂直递减率
    ter_hgt_mat = cv2.imread(fn_ter, flags=(cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH))
    p_hgt = 1013.25*np.power(1-(ter_hgt_mat/44300), 5.256)

    nrows = int((nor-sou)/dlat + 1.5)
    ncols = int((est-wst)/dlon + 1.5)
    gama = np.zeros((24, nrows, ncols))

    # load t lvls 
    t_lvls = np.zeros((len(lvls), 24, nrows, ncols))
    for i, ilvl in enumerate(lvls): 
        ifn_tplvl = fn_tplvl.format(t=idate, p=ilvl)
        _, imat, lons, lats = DIO.readNC(ifn_tplvl, 't', 'longitude', 'latitude', nor, sou, wst, est, dlon, dlat)
        t_lvls[i, :, :, :] = imat

    # calculat gama
    for i in range(len(lvls)-1):
        lvl_i = lvls[i]
        lvl_ip1 = lvls[i+1]
        h_i = 44300 * (1 - np.power(lvl_i/1013.25, 1/5.256))
        h_ip1 = 44300 * (1 - np.power(lvl_ip1/1013.25, 1/5.256))
        if i == 0: 
            # 注意这里进行比较的是气压！
            calc_idx = (p_hgt >= lvl_ip1)
        elif (i > 0) and (i < len(lvls)-1):
            calc_idx = (lvl_i >= p_hgt) & (p_hgt > lvl_ip1)
        else: 
            calc_idx = (lvl_i > p_hgt)

        # 这里直接使用 t_lvls[i, :, calc_idx] 会造成行列数相反，很奇怪
        igama = (t_lvls[i][:, calc_idx] - t_lvls[i+1][:, calc_idx]) / (h_i-h_ip1)
        gama[:, calc_idx] = igama

    # output gama 
    gama*= 1e3
    ifn_gama = fn_gama.format(t=idate)
    dt1 = int((idate - datetime(1900, 1, 1, 0)).seconds/3600+0.5)
    DIO.writeNC(ifn_gama, gama, lons, lats, np.arange(dt1, dt1+24), 'gama', dim3_name='hour')



if __name__ == "__main__": 
    fn_tplvl = 'y:/ERA5CN_t.plvl_{p:04d}/{t:%Y/%Y%m%d.nc}'
    fn_ter = 'z:/Terrain_0.0_60.0/DEM025_EAISA_SW.tif'
    fn_gama = 'y:/ERA5CN_GAMA/{t:%Y/%Y%m%d.nc}'

    lvls = [1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500]
    nor, sou, wst, est = 60.0, 0.0, 70.0, 140.0
    dlon, dlat = 0.25, 0.25

    idate = datetime(2023, 5, 20)
    date_end = datetime(2023, 7, 15)
    while idate <= date_end: 
        gama(idate, fn_ter, fn_gama, fn_tplvl, lvls, nor, sou, wst, est, dlon, dlat)
        idate += timedelta(days=1)
