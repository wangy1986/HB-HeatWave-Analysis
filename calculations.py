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
import matplotlib as mpl
import numpy as np 
import sys, os 
import utils.DataIO as DIO
import utils.draw_pictures as DP

global_era5_dict = {}


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


def Q(idate, fn_Q, fn_ssrd, fn_sntr, fn_strd, fn_sshf, fn_slhf, fn_mbld, 
      nor, sou, wst, est, dlon, dlat): 
    """
    计算非绝热加热项Q

    ssrd: surface solar radiation downward, unit = J m**-2
    sntr: surface_net_upward_longwave_flux, unit = J m**-2
    strd: Surface thermal radiation downwards, unit = J m**-2
    sshf: surface_upward_sensible_heat_flux, unit = J m**-2
    slhf: surface_upward_latent_heat_flux, unit = J m**-2
    mbld: Mean boundary layer dissipation, unit = W m**-2

    To convert from watts (W) to joules (J), 
    you need to know the time period of interest. 
    The formula for calculating energy in joules (J) is: 
        E(J) = P(W) x t(s) 
    where E is the energy in Joules, 
    P is the power in Watts, 
    and t is the time in seconds

    """
    ifn_Q = fn_Q.format(t=idate)
    nrows = int((nor-sou)/dlat+1.5)
    ncols = int((est-wst)/dlon+1.5)
    Qmat = np.zeros((24, nrows, ncols))

    idate = idate.replace(hour=0)
    while idate <= idate.replace(hour=23): 
        dacu_ssrd = get_acu_val_1H(idate, fn_ssrd, 'ssrd', nor, sou, wst, est, dlon, dlat)
        dacu_strd = get_acu_val_1H(idate, fn_strd, 'strd', nor, sou, wst, est, dlon, dlat)
        dacu_sntr = get_acu_val_1H(idate, fn_sntr, 'str', nor, sou, wst, est, dlon, dlat)
        dacu_sshf = get_acu_val_1H(idate, fn_sshf, 'sshf', nor, sou, wst, est, dlon, dlat)
        dacu_slhf = get_acu_val_1H(idate, fn_slhf, 'slhf', nor, sou, wst, est, dlon, dlat)
        # mbld 不是累计值, for mbld, need to conver the unit from (W m**-2) to (J m**-2)
        mbld = get_val_1H(idate, fn_mbld, 'mbld', nor, sou, wst, est, dlon, dlat)*3600

        Qmat[idate.hour, :, :] = dacu_ssrd + (dacu_strd - dacu_sntr) + \
                                 mbld + dacu_sshf + dacu_slhf
        idate += timedelta(hours=1)

    lons = np.arange(wst, est+0.5*dlon, dlon)
    lats = np.arange(sou, nor+0.5*dlat, dlat)
    dt1 = int((idate - datetime(1900, 1, 1, 0)).seconds/3600+0.5)
    DIO.writeNC(ifn_Q, Qmat, lons, lats, np.arange(dt1, dt1+24), 'Q', dim3_name='hour')


def get_val_1H(t_utc, fn_val, val_name, nor, sou, wst, est, dlon, dlat): 
    """
    获取某一物理量的值
    """
    ifn_val = fn_val.format(t=t_utc)
    if ifn_val not in global_era5_dict: 
        if (not os.access(ifn_val, os.R_OK)): 
            return None 
        else: 
            _, dmat, _, _ = DIO.readNC(ifn_val, val_name, "longitude", 'latitude', nor, sou, wst, est, dlon, dlat)
            global_era5_dict[ifn_val] = dmat
    
    return np.squeeze(dmat[t_utc.hour, :, :])


def get_acu_val_1H(t_utc, fn_acu, acu_name, nor, sou, wst, est, dlon, dlat): 
    """
    获取累计物理量 再 t_utc-1H ~ t_utc 之间的差值

    经测试, 表明对ERA5, accumulate 是从每日的 00UTC开始，即: 
        01UTC的数据，直接使用 01UTC的数据
        其余时刻的数据，需要使用 当前时刻 - 上一时刻 的矩阵获取
        ps: 对于 00utc时刻的数据，需要用 当日00UTC的数据 - 上一日23UTC数据获取

    """
    ifn_acu_t1 = fn_acu.format(t=t_utc-timedelta(hours=1))
    ifn_acu_t2 = fn_acu.format(t=t_utc)

    if ifn_acu_t1 not in global_era5_dict: 
        if (not os.access(ifn_acu_t1, os.R_OK)): 
            return None 
        else: 
            _, acu1, _, _ = DIO.readNC(ifn_acu_t1, acu_name, "longitude", 'latitude', nor, sou, wst, est, dlon, dlat)
            global_era5_dict[ifn_acu_t1] = acu1
    if ifn_acu_t2 not in global_era5_dict: 
        if (not os.access(ifn_acu_t2, os.R_OK)): 
            return None 
        else: 
            _, acu2, _, _ = DIO.readNC(ifn_acu_t2, acu_name, "longitude", 'latitude', nor, sou, wst, est, dlon, dlat)
            global_era5_dict[ifn_acu_t2] = acu2
    
    acu1 = global_era5_dict[ifn_acu_t1]
    acu2 = global_era5_dict[ifn_acu_t2]

    if t_utc.hour == 1:
        dacu = np.squeeze(acu2)
    else:
        dacu = np.squeeze(acu2 - acu1)
    return dacu


def show_accumulate_variable(idate1_utc, idate2_utc, fn_acu, acu_name, fn_png):
    """
    画图显示 acumulate 物理量
    测试这些物理量说明中的 
    "it is accumulated from the beginning of the forecast time to the end of the forecast step"
    到底是什么意思

    经测试, 表明对ERA5, accumulate 是从每日的 00UTC开始，即: 
        01UTC的数据，直接使用 01UTC的数据
        其余时刻的数据，需要使用 当前时刻 - 上一时刻 的矩阵获取
        ps: 对于 00utc时刻的数据，需要用 当日00UTC的数据 - 上一日23UTC数据获取
    """
    while idate1_utc <= idate2_utc: 
        t1 = idate1_utc
        t2 = idate1_utc+timedelta(hours=1)

        ifn_acu_t1 = fn_acu.format(t=t1)
        ifn_acu_t2 = fn_acu.format(t=t2)
        if (not os.access(ifn_acu_t1, os.R_OK)) or (not os.access(ifn_acu_t2, os.R_OK)):
            idate1_utc += timedelta(hours=1)
            continue 

        if ifn_acu_t1 == ifn_acu_t2: 
            _, acu, lons, lats = DIO.readNC(ifn_acu_t1, acu_name, 'longitude', 'latitude')
            acu_t1 = acu[t1.hour, :, :]
            acu_t2 = acu[t2.hour, :, :]
        else: 
            _, acu1, lons, lats = DIO.readNC(ifn_acu_t1, acu_name, 'longitude', 'latitude')
            _, acu2, lons, lats = DIO.readNC(ifn_acu_t2, acu_name, 'longitude', 'latitude')
            acu_t1 = acu1[t1.hour, :, :]
            acu_t2 = acu2[t2.hour, :, :]

        ifn_png = fn_png.format(t=t2)
        if t2.hour == 1:
            dacu = np.squeeze(acu_t2)
        else:
            dacu = np.squeeze(acu_t2 - acu_t1)
        min_val = np.floor(np.min(dacu))
        max_val = np.ceil(np.max(dacu))
        if abs(max_val - min_val) < 1e-5: 
            min_val, max_val = -1.0, 1.0, 
        clvls = np.arange(min_val, max_val+0.001, (max_val-min_val)/20)
        titles = {
            'left':     'Δ%s' % acu_name, 
            'right':    '{t2:%Y%m%d/%H}H - {t1:%Y%m%d/%H}H (LST)'.format(t2=t2+timedelta(hours=8), t1=t1+timedelta(hours=8))
        }
        DP.show_2D_mat(dacu, lons, lats, ifn_png, False, clvls, 
                       cmap=mpl.colormaps['tab20c_r'], titles=titles)
        #exit()
        idate1_utc += timedelta(hours=1)


if __name__ == "__main__": 
    fn_slhf = 'y:/ERA5CN_SLHF.land/{t:%Y/%Y%m%d.nc}'
    fn_sntr = 'y:/ERA5CN_SNTR.land/{t:%Y/%Y%m%d.nc}'
    fn_sshf = 'y:/ERA5CN_SSHF.land/{t:%Y/%Y%m%d.nc}'
    fn_ssrd = 'y:/ERA5CN_SSRD.land/{t:%Y/%Y%m%d.nc}'
    fn_strd = 'y:/ERA5CN_STRD.land/{t:%Y/%Y%m%d.nc}'
    fn_mbld = 'y:/ERA5CN_MBLD/{t:%Y/%Y%m%d.nc}'

    fn_tplvl = 'y:/ERA5CN_t.plvl_{p:04d}/{t:%Y/%Y%m%d.nc}'
    fn_ter = 'z:/Terrain_0.0_60.0/DEM025_EAISA_SW.tif'


    fn_gama = 'y:/ERA5CN_GAMA/{t:%Y/%Y%m%d.nc}'
    fn_sshf_png = 'y:/_Analysis/Png_SSHF/{t:%Y%m%d%H}.png'
    fn_ssrd_png = 'y:/_Analysis/Png_SSRD/{t:%Y%m%d%H}.png'
    fn_slhf_png = 'y:/_Analysis/Png_SLHF/{t:%Y%m%d%H}.png'
    fn_sntr_png = 'y:/_Analysis/Png_SNTR/{t:%Y%m%d%H}.png'
    fn_strd_png = 'y:/_Analysis/Png_STRD/{t:%Y%m%d%H}.png'
    fn_mbld_png = 'y:/_Analysis/Png_MBLD/{t:%Y%m%d%H}.png'

    lvls = [1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500]
    nor, sou, wst, est = 60.0, 0.0, 70.0, 140.0
    dlon, dlat = 0.25, 0.25

    idate = datetime(2023, 5, 20)
    date_end = datetime(2023, 7, 15)
    
    #while idate <= date_end: 
    #    gama(idate, fn_ter, fn_gama, fn_tplvl, lvls, nor, sou, wst, est, dlon, dlat)
    #    idate += timedelta(days=1)

    #show_accumulate_variable(idate, date_end, fn_ssrd, 'ssrd', fn_ssrd_png)
    show_accumulate_variable(idate, date_end, fn_slhf, 'slhf', fn_slhf_png)
    show_accumulate_variable(idate, date_end, fn_sntr, 'str', fn_sntr_png)
    show_accumulate_variable(idate, date_end, fn_sshf, 'sshf', fn_sshf_png)
    show_accumulate_variable(idate, date_end, fn_strd, 'strd', fn_strd_png)