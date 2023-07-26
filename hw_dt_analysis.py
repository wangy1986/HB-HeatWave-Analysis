#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
@FILE      : hw_dt_analysis.py
@TIME      : 2023/07/24 10:42:59
@AUTHOR    : wangyu / NMC
@VERSION   : 1.0
@DESC      : 从基础热力学方程的角度出发，分析各项对升温的贡献
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

import calculations as calc
import numpy as np 
import utils.DataIO as DIO


def dT_analysis(idate, fn_t2m, fn_u10, fn_v10, fn_w, fn_gama, fn_ssrd, fn_sntr, fn_sshf, fn_slhf, fn_mbld): 
    """
    ∂T/∂t = -V·▽T - w(γd-γ) + (1/cp)(dQ/dt)
    计算上式 右侧3项
    """
    ifn_t2m = fn_t2m.format(t=idate)
    ifn_u10 = fn_u10.format(t=idate)
    ifn_v10 = fn_v10.format(t=idate)
    ifn_w = fn_u10.format(t=idate)
    ifn_gama = fn_gama.format(t=idate)
    #ifn_ssr




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
        calc.gama(idate, fn_ter, fn_gama, fn_tplvl, lvls, nor, sou, wst, est, dlon, dlat)
        idate += timedelta(days=1)
    #calculate_gama(idate, fn_ter, fn_gama, fn_tplvl, lvls, nor, sou, wst, est, dlon, dlat)
