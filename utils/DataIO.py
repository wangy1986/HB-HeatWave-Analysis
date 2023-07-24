# 负责进行数据的 read / write 操作

import numpy as np
from netCDF4 import Dataset
import pandas as pd 
import os, time
import cv2 as cv

import sys, os
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir)

from utils import common_func as cf

def readNC(fn, val_name, lon_name='lon', lat_name='lat', 
           nor=60.0, sou=0.0, wst=70.0, est=140.0, dlon=0.05, dlat=0.05, 
           fill_val=np.nan, 
           method=cv.INTER_NEAREST, 
           max_threshold=9999.0):
    """
    负责读取 NC 文件，并将数据切割到指定的目标区域 [nor, sou, wst, est]
    
    返回数组的默认方向是 从 south -> north, west -> east
    即以 西南角 为坐标原点

    --------------
    Parameters
    --------------
    fn:                 nc文件的文件名
    val_name:           读取那个变量, 例如: 't2m'

    lon_name:           nc文件中经度数据的名字，例如: 'lon'
                        默认值是 'lon'
    lat_name:           nc文件中纬度数据的名字，例如: 'lon'
                        默认值是 'lat'

    nor:                获取数据的北边界, 默认值是 60.0
    sou:                获取数据的北边界, 默认值是 0.0
    wst:                获取数据的西边界, 默认值是 70.0
    est:                获取数据的东边界, 默认值是 140.0
    
    dlon:               经度步长, 默认值是 0.05
    dlat:               纬度步长, 默认值是 0.05

    fill_val:           当数据不存在时，的填充值
                        例如，当需要的数据边界范围超过实际nc数据的范围
                            则在超出部分填充 np.nan
                        默认值是 np.nan

    method:             当需要插值时，使用的插值方法
                        默认使用 最邻近方法 cv.INTER_NEAREST
                        当前插值使用 openCV 内置的插值函数 resize
                        因此接收 cv.resize() 的参数，如下：
                            cv.INTER_NEAREST:       最近邻
                            cv.INTER_LINEAR:        双线性
                            cv.INTER_AREA:          基于区域关系的重采样
                            cv.INTER_CUBIC:         双三次插值
                            cv.INTER_LANCZOS4:      基于 8*8 邻域的 Lanczos 插值算法

    max_threshold:      进行阈值控制              
    --------------
    Returns
    --------------
    b_vld, dmat, lons, lats:   
                        b_vld:  数据是否为有效数据
                        dmat:   数据矩阵 = [nense, nrows, ncols]
                        lons:   经度数组 = [ncols]
                        lats:   纬度数组 = [nrows]

                        默认方向是 从 south -> north, west -> east
                        即以 西南角为坐标原点
    """

    lons = np.arange(wst, est+0.5*dlon, dlon)
    lats = np.arange(sou, nor+0.5*dlat, dlat)
    dmat = np.zeros((1, len(lats), len(lons))) + fill_val

    # 如果数据文件不存在
    # 则直接返回 fill_val 填充的数组
    if not os.access(fn, os.R_OK): 
        print('NO file: %s' % fn)
        return False, dmat, lons, lats

    # 文件存在，则尝试读取 原始数据
    h1 = None
    try: 
        h1 = Dataset(fn)

        if lon_name not in h1.variables.keys(): 
            for ikey in h1.variables.keys(): 
                if 'lon' in ikey: 
                    lon_name = ikey 
                    break 
        if lat_name not in h1.variables.keys(): 
            for ikey in h1.variables.keys(): 
                if 'lat' in ikey: 
                    lat_name = ikey 
                    break

        if val_name not in h1.variables.keys(): 
            #print('the key: [ %s ] dose not exist in h1.variables' % val_name)
            for ikey in h1.variables.keys(): 
                if (ikey != lon_name) and (ikey != lat_name) and \
                    (ikey != 'ensemble') and (ikey != 'ens') and \
                    (ikey != 'time') and (ikey != 'fhour')and (ikey != 'level'): 
                    val_name = ikey 
                    #print('replace it as : [ %s ]' % val_name)
                    break

        lons_nc = h1.variables[lon_name][:]
        lats_nc = h1.variables[lat_name][:]
        # reshape the raw data to [n_ense, nrows, ncols]
        mat_nc = np.squeeze(h1.variables[val_name][:]).reshape(-1, len(lats_nc), len(lons_nc))
        
        # nc数据的经纬度范围
        nor_nc = np.max(lats_nc)
        sou_nc = np.min(lats_nc)
        wst_nc = np.min(lons_nc)
        est_nc = np.max(lons_nc)
        dlon_nc = lons_nc[1] - lons_nc[0]
        dlat_nc = lats_nc[1] - lats_nc[0]

        n_ense = mat_nc.shape[0]
    except Exception as e: 
        print('Error in reading file: %s' % fn)
        print('%s' % e.__str__)
        if h1 is not None: 
            print(h1.variables.keys())
        return False, dmat, lons, lats

    if ((type(lats_nc) is np.ma.core.MaskedArray) and (type(lats_nc.mask) is not np.bool_)) or \
        ((type(lons_nc) is np.ma.core.MaskedArray) and (type(lons_nc.mask) is not np.bool_)): 
            print('the lat/lon array is masked, return the raw data')
            return False, dmat, None, None

    # 真实数据块的经纬度范围，即 nc 文件提供的数据，和所需的经纬度范围数据的交集
    nor_data = min(nor_nc, nor)
    sou_data = max(sou_nc, sou)
    wst_data = max(wst_nc, wst)
    est_data = min(est_nc, est)

    # 这里需要考虑原始 nc 数据有可能是 以西北角为原点的情况，即此时的 dlat_nc < 0
    if dlat_nc < 0: 
        mat_nc = mat_nc[:, ::-1, :]
        dlat_nc = np.abs(dlat_nc)
        lats_nc = lats_nc[::-1]
    if dlon_nc < 0: 
        mat_nc = mat_nc[:, :, ::-1]
        dlon_nc = np.abs(dlon_nc)
        lons_nc = lons_nc[::-1]
        
    # 真实数据在 nc 数据中的 索引值
    idx_nc_nor = int((nor_data - lats_nc[0]) / dlat_nc + 0.5)
    idx_nc_sou = int((sou_data - lats_nc[0]) / dlat_nc + 0.5)
    idx_nc_wst = int((wst_data - lons_nc[0]) / dlon_nc + 0.5)
    idx_nc_est = int((est_data - lons_nc[0]) / dlon_nc + 0.5)
    
    # 从nc数据中切割出该块数据
    dmat_nc = mat_nc[:, idx_nc_sou:idx_nc_nor+1, idx_nc_wst:idx_nc_est+1]

    # 真实数据在 output 数据中的索引值
    idx_out_nor = int((nor_data - sou) / dlat + 0.5)
    idx_out_sou = int((sou_data - sou) / dlat + 0.5)
    idx_out_wst = int((wst_data - wst) / dlon + 0.5)
    idx_out_est = int((est_data - wst) / dlon + 0.5)

    nrows_out = idx_out_nor - idx_out_sou + 1
    ncols_out = idx_out_est - idx_out_wst + 1
    # 使用 opencv 中内置插值方法，将 nc 数据插值到目标分辨率
    # 注意，opencv resize接受的dimension参数为 [width, height] 
    dmat_out = np.zeros((n_ense, nrows_out, ncols_out))
    for i in range(n_ense): 
        dmat_out[i, :, :] = cv.resize(dmat_nc[i, :, :], (ncols_out, nrows_out), interpolation=method)
    
    # 将插值后的数据填充到 dmat
    dmat = np.zeros((n_ense, len(lats), len(lons))) + fill_val
    dmat[:, idx_out_sou:idx_out_nor+1, idx_out_wst:idx_out_est+1] = dmat_out
    dmat[dmat > max_threshold] = 0.0 
    '''
    # test
    from matplotlib import pyplot as plt
    plt.imshow(dmat_nc)
    plt.show()
    '''
    return True, dmat, lons, lats
# end of readNC


def writeNC(to_fn,
            dmat, lons, lats, dim3=None, 
            val_name='val', lon_name='longitude', lat_name='latitude', dim3_name='dim3'): 
    """
    将 dmat 的数据输出为 nc 格式文件，数据的变量名由 val_name 指定
    lons, lats 表示数据的经纬度
    dmat.shape = [dim3, lat, lon]

    -------------
    Parameters
    -------------
    to_fn:              文件存储路径

    dmat:               数据矩阵， np.ndarray
    lons:               经度矩阵，1维 np.ndarray
    lats:               纬度矩阵，1维 np.ndarray
    dim3:               第3个纬度，1维 np.ndarray

    val_name:           nc文件中，数据的名称，字符串
    lon_name:           nc文件中，经度数组的名称，字符串
    lat_name:           nc文件中，纬度数组的名称，字符串
    dim3_name:          nc文件中，dim3 维度的名称，字符串

    -------------
    Returns
    -------------
    True/False:         文件写入成功: True
                        文件写入失败: False

    """
    # 检测 fn 是否可行，即查看存储文件fn时，所需的路径是否已经创建
    # 如果路径不存在，则创建该路径
    cf.check_fn_available(to_fn)

    count = 0
    while(os.path.exists(to_fn) and (not os.access(to_fn, os.W_OK))): 
        # 如果文件存在，但同时不可写，此时应该为文件被其他程序占用，此时选择等待
        time.sleep(1)
        count += 1
        if count > 10: 
            # 如果10s后仍然不能写，则跳过写过程
            return False

    da = Dataset(to_fn, 'w', format='NETCDF4')

    # create dimensions
    da.createDimension(lon_name, len(lons))
    da.createDimension(lat_name, len(lats))
    # create variables
    out_lon = da.createVariable(lon_name, 'f4', (lon_name, ))
    out_lat = da.createVariable(lat_name, 'f4', (lat_name, ))

    if len(dmat.shape) == 3: 
        da.createDimension(dim3_name, len(dim3))
        out_dim3 = da.createVariable(dim3_name, 'f4', (dim3_name, ))

    if len(dmat.shape) == 2:
        out_val = da.createVariable(val_name, 'int', (lat_name, lon_name), zlib=True)
    elif len(dmat.shape) == 3: 
        out_val = da.createVariable(val_name, 'int', (dim3_name, lat_name, lon_name), zlib=True)
    # 设置 out_val 属性
    out_val.scale_factor = 0.01
    out_val.missing_value = 65535.0
    out_val.set_auto_maskandscale(True)

    # write data to NC Dataset da  
    out_lat[:] = lats
    out_lon[:] = lons
    out_val[:] = dmat

    da.close()
    return True

# end of writeNC


def generate_model_fcst_nc_name(dpath, model_name, init_date_utc, fstH): 
    """
    拼凑预报文件文件名 

    ---------------
    Parameters
    ---------------
    dpath:                  数据路径
    model_name:             模式名称（string）

    init_date_utc:          预报的起报时间, datetime
    fstH:                   预报时效, int

    ---------------
    Returns
    ---------------
    fn                      文件的绝对路径
    """

    init_date_str = '%4d%02d%02d%02d' % (init_date_utc.year, init_date_utc.month, init_date_utc.day, init_date_utc.hour)
    if (model_name.lower() == 'glbcn') or (model_name.lower() == 'glb'): 
        # 全球模式数据 or 全球模式中国区版本
        fn = '%s/%s/%s.%03d.nc' % (dpath, init_date_str, init_date_str, fstH)
    elif model_name.lower() == 'scmoc': 
        # SCMOC 预报
        fn = '%s/%s/%s.%03d.nc' % (dpath, init_date_str, init_date_str, fstH)
    else: 
        fn = None

    return fn
# end of generate_model_fcst_nc_name


def rewrite_ncep(fn):
    """
    提供的 nc 版本 ncep 温度数据，纬度数组的方向是反的，因此这里需要更正
    """ 
    h1 = Dataset(fn, 'a')
    for ikey in h1.variables.keys(): 
        if ikey == 't2m': 
            h1.variables[ikey][:] = h1.variables[ikey][:][::-1, :]

    h1.close()
# end of rewrite_ncep


def load_m3(abs_filepath: str, dst_latlon_dict: dict, 
    min_val=None, max_val=None, stations=None, encoding='utf8', 
    missing_val=None): 
    """
    读取 micaps-3 格式的站点预报数据
    本质上，与 read_station_obs_abs_file 是一样的，因此直接调用该函数
    然后组装成 station_data ( basic.data.station_data, 见 basic/__init__.py )

    赵瑞霞全球预报【站点】 的编码是 utf8!
    ------------------
    Parameters
    ------------------
    abs_filepath:       m3文件名的绝对路径 

    dst_latlon_dict:    经纬度网格信息，当设置为 None时，不进行经纬度控制
                        dst_latlon_dict = {
                            "nor": Constants.NOR_FST,
                            "sou": Constants.SOU_FST,
                            "wst": Constants.WST_FST,
                            "est": Constants.EST_FST,
                            "dlon": Constants.D_LON_FST,
                            "dlat": Constants.D_LAT_FST,
                            "nrows": Constants.N_ROWS_FST,
                            "ncols": Constants.N_COLS_FST,
                            "ngrids": Constants.N_GRID_FST
                        }

    min_val:            是否对数据进行阈值限制，当设置为None时，不进行阈值限制
    max_val:

    stations:           站点列表，如果为None，则获取m3 文件中所有站点预报，如果不为None，获取 stations 中的那些站点
                        本质上是一个 pd.dataframe

    missing_val:        默认的缺测值
                        当为 None 时，无缺测值检验
                        当为 float 值时， 将 float_val +/- 0.1 以内的值，设置为 pd.na
    ------------------
    Returns
    ------------------
    True/False, dmat
        True:       正常读取数据
        False:      无法获取数据（可能是异常，也可能是数据不存在）
        dmat:       pd.dataframe like:
                            stID       lon      lat     hgt    val
                    0       50514  117.3214  49.5758   661.8   4.3
                    1       50516  118.5739  49.4772   576.0   5.0
                    2       50524  119.4569  49.3414   597.4   2.2
                    3       50525  119.7500  49.1500   620.8   4.5

    """
    if not os.access(abs_filepath, os.R_OK):
        return False, None

    if dst_latlon_dict is not None: 
        nor = dst_latlon_dict['nor']
        sou = dst_latlon_dict['sou']
        wst = dst_latlon_dict['wst']
        est = dst_latlon_dict['est']
    else: 
        nor, sou, wst, est = None, None, None, None

    # read micaps-3 data from file
    try:
        dmat = pd.read_csv(abs_filepath, sep="\s+", encoding=encoding, skiprows=[0, 1], header=None, names=["stID", "lon", "lat", "hgt", "val"])

        # 还需要额外判断，站点id 是 字符串 / 数字
        # 如果是 字符串，需要转换为 数字
        # like  '054511' -> 54511
        #       'Gxxxxx' -> ???
        if type(dmat.iloc[0, 0]) is str: 
            dmat['stID'] = dmat.apply(_convert_str_to_stID, axis=1)

        if stations is not None:
            dmat = pd.merge(stations.loc[:, ("stID", "lon", "lat", "hgt")], 
                            dmat.loc[:, ("stID", "val")], 
                            how="inner", on=["stID"], suffixes=("", ""))[["stID", "lon", "lat", "hgt", "val"]]
        # dmat = dmat.loc[:, ["stID", "lon", "lat", "hgt", "val"]]

        if (nor is not None) and (sou is not None) and (wst is not None) and (est is not None):
            dmat = dmat[(dmat['lat'] <= nor) & (dmat['lat'] >= sou) & (dmat['lon'] >= wst) & (dmat['lon'] <= est)].reset_index(drop=True)

        if (min_val is not None) and (max_val is not None):
            dmat = dmat[(dmat['val'] >= min_val) & (dmat['val'] <= max_val)].reset_index(drop=True)
        
        if missing_val is not None: 
            dmat["val"][(missing_val-0.1 <= dmat["val"]) & (dmat["val"] <= missing_val+0.1)] = pd.NA
        # dmat[dmat>9998.98 & dmat < 9999.02] = pd.NA

    except Exception as ex:
        print(ex.__str__())
        dmat = None

    if dmat is None:
        return False, None
    else:
        return True, dmat
# end of load_m3


def _convert_str_to_stID(irec): 
    """
    将 字符串形式 的站点id，转化为 数字格式
    irec = pd.Series, 是一条站点记录（包括站点号、经度、纬度、高度和取值）

    :param irec:        一条站点记录（包括站点号、经度、纬度、高度和取值）
    :return:            int 格式的站点号

    """
    if irec['stID'].isdigit(): 
        return int(irec['stID'])
    elif irec['stID'][0] == 'G':
        # 赵瑞霞全球预报中出现的站点，站点号第一个字母为G
        # 被称为“G”类站点。据说是从数据库中下载的海洋区域为主的观测站
        # 下载时就没有站号，只有经纬度和数值
        # 这里出于整个 pandas dataframe 运行速度考虑，将其转化为 int 型 
        return 7100000 + int(irec['stID'][1:])
    elif irec['stID'][0] == 'C': 
        # 赵瑞霞全球预报（中国区）出现的站点，站点第一个字母为C
        # C类站，具体出现原因不明
        return 3500000 + int(irec['stID'][1:])


if __name__ == "__main__":
    pass
