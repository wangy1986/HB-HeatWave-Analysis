#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
@FILE      : populations.py
@TIME      : 2023/07/21 01:38:55
@AUTHOR    : wangyu / NMC
@VERSION   : 1.0
@DESC      : 本程序用于读取 人口数据
'''


### to import parent dir files ###
# import os, sys
### this is for jupyter notebook ###
#current_folder = globals()['_dh'][0]
#parentdir = os.path.dirname(current_folder)
### this is for normal python file ###
#parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#sys.path.insert(0,parentdir)

import numpy as np
import pyproj
import rasterio
import warnings
warnings.filterwarnings('ignore')

from copy import deepcopy
from rasterio.crs import CRS 
from datetime import datetime, timedelta

pops_dict = {}
# src._transform like: 
# [-180.0, 0.0083333333333333, 0.0, 89.99999999999929, 0.0, -0.0083333333333333]
transform_dict = {}

def load_populations(fn_pops, year): 
    """
    读取给定年份的人口数据
    """ 
    ifn_pops = fn_pops.format(t=datetime(year, 1, 1))
    with rasterio.open(ifn_pops) as src: 
        # Get the CRS of the raster
        #crs = CRS.from_wkt(src.crs.wkt)

        pops_dict[year] = src.read(1)
        # src._transform like: 
        # [-180.0, 0.0083333333333333, 0.0, 89.99999999999929, 0.0, -0.0083333333333333]
        transform_dict[year] = src._transform


def get_pops_lonlat(year, lon, lat): 
    if year not in pops_dict: 
        return None 
    
    # Convert longitude and latitude to pixel coordinates
    if type(lat) is not np.ndarray:
        x = int((lat - transform_dict[year][3])/transform_dict[year][-1]+0.5)
    else: 
        x = ((lat - transform_dict[year][3])/transform_dict[year][-1]+0.5).astype(int)
    
    if type(lon) is not np.ndarray:
        y = int((lon - transform_dict[year][0])/transform_dict[year][1]+0.5)
    else: 
        y = ((lon - transform_dict[year][0])/transform_dict[year][1]+0.5).astype(int)
    
    population = pops_dict[year][x, y]
    return population


if __name__ == "__main__": 
    # test read population file 
    fn_pops = 'y:/Populations.LandScan/landscan-global-{t:%Y}-assets/landscan-global-{t:%Y}.tif'

    load_populations(fn_pops, 2022)

    # beijing's lon and lat
    #lon, lat = 116.4, 39.9
    lons = np.array([117, 116])
    lats = np.array([40, 41])
    #print(get_pops_lonlat(2022, lons, lats))
    for ilon, ilat in zip(lons, lats): 
        print(get_pops_lonlat(2022, ilon, ilat))
