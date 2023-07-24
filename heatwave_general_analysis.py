#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
@FILE      : Month6_general_analysis.py
@TIME      : 2023/07/20 22:34:20
@AUTHOR    : wangyu / NMC
@VERSION   : 1.0
@DESC      : 本文件负责进行 2023年 6 月 华北地区高温事件的基础统计分析
             华北地区 6 月份高温事件集中在 6月14-17日, 6月21-30日

'''


### to import parent dir files ###
# import os, sys
### this is for jupyter notebook ###
#current_folder = globals()['_dh'][0]
#parentdir = os.path.dirname(current_folder
### this is for normal python file ###
#parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#sys.path.insert(0,parentdir)

from datetime import datetime, timedelta 
import numpy as np 
import pandas as pd 
import matplotlib as mpl 
import matplotlib.pyplot as plt
import cv2 
import geopandas as gpd

from utils import common_func as cf 
from utils import DataIO as DIO
from utils import draw_pictures as DP


def draw_daily_max_temperature_stat(fn_t2m_stat,  fn_daily_tmax, fn_png, idate_lst, nor, sou, wst, est, dlon, dlat):
    """
    """
    
    


def draw_daily_max_temperature(fn_t2m_grd, fn_tmx24_stat, fn_daily_tmax, fn_png, idate_lst, nor, sou, wst, est, dlon, dlat):
    """
    首先绘制华北地区 日2m最高温度 的空间分布
    这里日最高温度，比较 当日 12时 ~ 20时 温度
    华北地区:  nor, sou, wst, est = 45, 30, 110, 125

    温度数据使用 ERA-5 tmax 数据
    # Maximum 2m temperature since previous post-processing
    #   defined as: 
    #   This parameter is the highest temperature of air at 2m above the
    #   surface of land, sea or inland water since the parameter was last
    #   archived in a particular forecast. 2m temperature is calculated by
    #   interpolating between the lowest model level and the Earth's
    #   surface, taking account of the atmospheric conditions. This
    #   parameter has units of kelvin (K). Temperature measured in kelvin
    #   can be converted to degrees Celsius (°C) by subtracting 273.15.
    """
    ### create daily tmx24 file 
    # 通过求取 当日 12H~20H(LST) 的小时最高温度的最大值，获取网格点的日最高温度
    ifn_tmx_1h = fn_t2m_grd.format(t=idate_lst)
    bvld, tmx_1h, lons, lats = DIO.readNC(ifn_tmx_1h, 'mx2t', 'longitue', 'latitude', 
                                          nor=nor, sou=sou, wst=wst, est=est, dlon=dlon, dlat=dlat)
    if not bvld: 
        return None 
    
    nrows = int((nor-sou)/dlat + 1.5)
    ncols = int((est-wst)/dlon + 1.5)
    tmx_daily = np.max(tmx_1h[12-8:21-8, :, :], axis=0)-273.15
    cnts_lonlat = daily_heatwave_detect(tmx_daily)

    ifn_tmx_daily = fn_daily_tmax.format(t=idate_lst)
    DIO.writeNC(ifn_tmx_daily, tmx_daily, lons, lats, 'mx2t', 'longitude', 'latitude')


    ### load station data 
    ifn_tmx24_stat = fn_tmx24_stat.format(t=idate_lst)
    nrows = int((nor-sou)/dlat + 1.5)
    ncols = int((est-wst)/dlon + 1.5)
    dst_lonlat = {
        "nor": nor,
        "sou": sou,
        "wst": wst,
        "est": est,
        "dlon": dlon,
        "dlat": dlat,
        "nrows": nrows,
        "ncols": ncols,
        "ngrids": nrows*ncols 
    }
    ifn_png = fn_png.format(t=idate_lst.replace(hour=20))
    _, stat_data = DIO.load_m3(ifn_tmx24_stat, dst_latlon_dict=dst_lonlat, encoding='gbk')
    stat_data = stat_data[stat_data["val"]>=35]

    #####################
    ### draw pictures ###

    tmax = np.max(tmx_daily)
    tmin = np.min(tmx_daily)
    clvls = np.arange(np.floor(tmin), np.ceil(tmax+1), 1)
    titles_left = {
        'left':     'Date: {t:%Y/%m/%d}'.format(t=idate_lst), 
        'right':    'ERA5 Daily TMX24: %.2f ℃' % (tmax)
    }
    titles_right = {
        'right':    'Station Daily TMX24: %.2f ℃' % (np.max(stat_data["val"].values))
    }
    #DP.show_2D_mat(tmx_daily, lons, lats, ifn_png, False, clvls=clvls, cmap=mpl.colormaps['seismic'], titles=titles, 
    #               contours_lonlat=cnts_lonlat)
    DP.show_2D_mat_2pic_with_obs(dmat1=tmx_daily, lons1=lons, lats1=lats, clvls1=clvls, cmap1=mpl.colormaps['seismic'], title1=titles_left, contours_lonlat1=cnts_lonlat, 
                                 lons2=lons, lats2=lats, obs2=stat_data, obs2_cmap=mpl.colormaps['Reds'], title2=titles_right, contours_lonlat2=cnts_lonlat, 
                                 archive_fn=ifn_png, )


def daily_heatwave_detect(dmat, thresh=35.0): 
    """
    基于当日的 日最高温度数据，识别 heat wave
    返回值为 [[lon, lat], [lon, lat], ... ]组成的 list
    """

    # 使用阈值法制作 >35° 掩膜
    _, masked_mat = cv2.threshold(dmat, thresh, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    masked_mat = cv2.morphologyEx(masked_mat.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(masked_mat, mode=cv2.RETR_EXTERNAL, 
                                           method=cv2.CHAIN_APPROX_SIMPLE)
    
    cnts_lonlat = cf.trans_cvContours_to_lonlat(contours, wst, sou, dlon, dlat)

    return cnts_lonlat 

    
def heatwave_co_pair(): 
    """
    不同日数据识别出的 > 35℃ heatwave 之间进行配对
    构建一次 heat wave 事件
    """

if __name__ == "__main__": 
    fn_tmx01_grd = 'y:/ERA5_T2MX/{t:%Y/%Y%m%d}.nc'
    fn_daily_tmax = 'y:/TMP/Huabei_Daily_TMax/{t:%Y%m%d}.nc'
    fn_tmx24_stat = 'z:/YLRC_STATION/TEMP/rtmx24/{t:%Y/%Y%m%d%H}.000'
    fn_png = 'f:/华北地区热浪分析/T2Max/{t:%Y%m%d}.png'
    #nor, sou, wst, est = 47, 28, 108, 127
    nor, sou, wst, est = 60, 10, 70, 140
    dlon, dlat = 0.25, 0.25

    idate = datetime(2023, 6, 1)
    date_end = datetime(2023, 6, 1)

    # step 1. 
    while idate <= date_end:
        draw_daily_max_temperature(fn_tmx01_grd, fn_tmx24_stat, fn_daily_tmax, fn_png, idate, nor, sou, wst, est, dlon, dlat)
        idate += timedelta(days=1)




