#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
   @Project   ：NBM-Clone 
   
   @File      ：common_func.py
   
   @Author    ：yhaoxian
   
   @Date      ：2021/12/15 17:13 
   
   @Describe  : 
   
"""
import os
import sys
import pickle
import numpy as np


from datetime import datetime, timedelta
from math import cos, sin, asin, sqrt
#from numba import jit



def is_leap_year(dateTime: datetime):
    """
    判断当前时间是否为闰年
    用当前时间年份的3月1日减去一天，判断日期是29还是28来判断是否为闰年
    Args:
        dateTime:      日期    2021-12-15 18:00:00

    Returns:
        True   闰年
        False  平年

    """
    date_yesterday = datetime(dateTime.year, 3, 1) - timedelta(days=1)
    if 29 == date_yesterday.day:
        return True
    else:
        return False


def get_days_each_month(dateTime: datetime):
    """
    根据年份的获取每月天数
    Args:
        dateTime:      日期    2021-11-20 00:00:00

    Returns:
        if  is leap year, return [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        if not leap year, return [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    """
    if is_leap_year(dateTime):
        return [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        return [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def get_days_a_month(dateTime: datetime):
    """
    根据日期获取当月天数
    Args:
        dateTime:      日期    2021-11-20 00:00:00

    Returns:
        30

    """
    date_y, date_m = dateTime.year, dateTime.month
    if date_m < 12:
        return (datetime(date_y, date_m + 1, 1) - datetime(date_y, date_m, 1)).days
    else:
        return (datetime(date_y + 1, 1, 1) - datetime(date_y, 12, 1)).days


def convert_datetime_to_yyyyddd(dateTime: datetime):
    """
    将指定日期转换成年，当年的第多少天
    Args:
        dateTime:      日期    2021-11-20 00:00:00

    Returns:
        (2021, 324)

    """
    days = (dateTime - datetime(dateTime.year, 1, 1)).days + 1
    return dateTime.year, days


def convert_yyyyddd_to_yyyymmdd(year, days):
    """
    将指定年份和第几天转换成年、月、日
    Args:
        year:      年份      2021
        days:      第几天    324

    Returns:
        (2021, 11, 20)

    """
    assert days > 0
    date_start = datetime(year, 1, 1) + timedelta(days=days - 1)
    return date_start


def convert_yyyyddd_to_ddddd_since_1900(year, days):
    """
    计算指定年份，指定天数距1990年的总天数
    Args:
        year:      年份      2021
        days:      第几天    324

    Returns:
        44519

    """
    return (datetime(year, 1, 1) - datetime(1900, 1, 1)).days + days


def convert_ddddd_since_1900_to_yyyyddd(ddddd: int):
    """
    计算从1900年1月1日+ddddd天后的日期
    Args:
        ddddd:        第多少天    44519

    Returns:
        (2021, 324)

    """
    dateTime_1 = datetime(1900, 1, 1) + timedelta(days=ddddd)
    dateTime_2 = datetime(dateTime_1.year, 1, 1)
    days = (dateTime_1 - dateTime_2).days
    return dateTime_1.year, days


def convert_ddddd_since_1900_to_yyyymmdd(ddddd):
    """
    计算从1900年到ddddd天的日期
    Args:
        ddddd:        第多少天    44519

    Returns:
        (2021, 11, 20)

    """
    dateTime_1 = datetime(1900, 1, 1) + timedelta(days=ddddd - 1)
    return dateTime_1


def convert_yyyymmdd_to_ddddd_since_1900(dateTime: datetime):
    """

    Args:
        dateTime:      日期    2021-11-20 00:00:00

    Returns:
        44519

    """
    dateTime_1 = datetime(1900, 1, 1)
    return (dateTime - dateTime_1).days + 1


def output_log_record(fn_log, infor_str, b_screen_print=True):
    """
    向 log 文件输出日志
    Args:
        fn_log:                   日志文件文件名, 如果为 None，则不输出信息
        infor_str:                日志信息，本函数会自动在信息头部添加时间信息
        b_screen_print:           是否在屏幕上打印该日志信息

    Returns:
        None

    """
    tt = datetime.now()
    output_str = '%4d%02d%02d %02d:%02d:%02d [%s]' % (
        tt.year, tt.month, tt.day, tt.hour, tt.minute, tt.second, infor_str)

    if b_screen_print:
        print(output_str)

    if fn_log is None:
        return

    with open(fn_log, 'a') as f:
        if output_str[-1] == '\n':
            f.write(output_str)
        else:
            f.write('%s\n' % output_str)


def convert_fst_time_to_obs_time(date_time: datetime, fst_f, b_fst_utc, b_obs_utc):
    """

    将预报时间(包含起报时间)转换成实况时间

    Args:
        date_time:          预报时间      ex: 2021-10-10 08:00:00
        fst_f:              预报时效      ex: 312
        b_fst_utc:
        b_obs_utc:

    Returns:

    """
    fst_f += 0 if b_fst_utc else -8
    fst_f += 0 if b_obs_utc else 8
    date_obs = date_time + timedelta(hours=fst_f)
    return date_obs


def convert_fst_time_to_obs_time_tlist(t_list, start_h, fst_h, b_fst_utc, b_obs_utc):
    """
    将预报时间转换为实况时间
    Args:
        t_list:          [trange1, trange2, ...]
                        itrange = [start_y, start_m, start_d, end_y, end_m, end_d]
        start_h:        起报时间
        fst_h:          预报时效
        b_fst_utc:      上面给出的时报时间是否为UTC
        b_obs_utc:      待转换的实况时间是否为UTC

    Returns:
        tlist_obs, obs_h
            tlist_obs = [trange1, trange2, ...], 与 tlist 对应的实况时间段
                itrange = [start_y, start_m, start_d, end_y, end_m, end_d]

        obs_h:      实况观测时间

    """
    tlist_obs = []
    for itr in t_list:
        date1_obs = convert_fst_time_to_obs_time(datetime(itr[0], itr[1], itr[2], start_h), fst_h, b_fst_utc, b_obs_utc)
        date2_obs = convert_fst_time_to_obs_time(datetime(itr[3], itr[4], itr[5], start_h), fst_h, b_fst_utc, b_obs_utc)
        tlist_obs.append([date1_obs.year, date1_obs.month, date1_obs.day, 
                          date2_obs.year, date2_obs.month, date2_obs.day])
    return tlist_obs, date1_obs.hour


#@jit(nopython=True, cache=True)
def Haversine_KM(lon_1, lat_1, lon_2, lat_2):
    """
    计算地图上两点经纬度间的距离
    https://blog.csdn.net/baidu_32923815/article/details/79719813
    Args:
        lon_1:
        lat_1:
        lon_2:
        lat_2:

    Returns:

    """
    # 将十进制度数转化为弧度
    lon_1 = lon_1 * 0.01745329
    lat_1 = lat_1 * 0.01745329
    lon_2 = lon_2 * 0.01745329
    lat_2 = lat_2 * 0.01745329
    # Haversine公式
    a = sin((lat_2 - lat_1) / 2) ** 2 + cos(lat_1) * cos(lat_2) * sin((lon_2 - lon_1) / 2) ** 2
    # 地球平均半径: 6370.856 km
    return 2 * asin(sqrt(a)) * 6370.856


def trans_t_list_to_days_list(t_list):
    """
    将 tlist = [[start_y, start_m, start_d, end_y, end_m, end_d], ...0 ]
    转换为 day_list = [start_ddddd, ... ], [end_dddddd, ... ]
    Args:
        t_list:         [[start_y, start_m, start_d, end_y, end_m, end_d], ...0 ]

    Returns:
        n_total_days, [start_ddddd1, start_ddddd2, ...], [end_ddddd1, end_ddddd2, ...]

    """
    n_total_days = 0
    d1_list = []
    d2_list = []
    for i in range(len(t_list)):
        d1 = convert_yyyymmdd_to_ddddd_since_1900(datetime(t_list[i][0], t_list[i][1], t_list[i][2]))
        d2 = convert_yyyymmdd_to_ddddd_since_1900(datetime(t_list[i][3], t_list[i][4], t_list[i][5]))
        d1_list.append(d1)
        d2_list.append(d2)
        n_total_days += (d2 - d1 + 1)
    return n_total_days, d1_list, d2_list


def get_dict_parameters(para_dict, key_list):
    """
    用于提取 para_dict 之中的参数
    Args:
        para_dict:        参数字典
        key_list:         待检验的关键词, = [key_name1, key_name2, ..., key_nameN]

    Returns:
        value_list： 对应 key 的键值，但如果对应key不存在，则其值以 None 表示
                    = [value1, value2, ..., valueN]

    """
    value_list = []
    for ikey in key_list:
        if ikey in para_dict:
            value_list.append(para_dict[ikey])
        else:
            print('para_dict中缺少必要参数 %s' % ikey)
            value_list.append(None)

    return value_list


def check_fn_available(fn):
    """
    检测 fn 是否可行，即查看存储文件fn时，所需的文件夹是否已经齐备
    Args:
        fn:         文件绝对路径

    Returns:

    """
    path = fn[0:max(fn.rfind('/'), fn.rfind('\\'))]
    try: 
        check_path_exist_else_mkdir(path, '', None)
    except: 
        pass 


def check_path_exist_else_mkdir(path, path_name_str, log_fn):
    """
    判断路径是否存在，如果路径不存在，则创建该路径
    Args:
        path:                  目标路径 (str/list/dict)
        path_name_str:         路径的名称
        log_fn:                日志文件名

    Returns:
        None

    """
    if type(path) is dict:
        for k, v in path.items():
            if type(v) is not str:
                check_path_exist_else_mkdir(v, path_name_str, log_fn)
            else:
                if os.path.exists(v) is False:
                    log_str = '创建路径 %s-%s: %s' % (path_name_str, k, v)
                    output_log_record(log_fn, log_str)
                    os.makedirs(v)
    elif type(path) is list:
        for i_path in path:
            if os.path.exists(i_path) is False:
                log_str = '创建路径 %s: %s' % (path_name_str, path)
                output_log_record(log_fn, log_str)
                os.makedirs(i_path)
    else:
        if os.path.exists(path) is False:
            log_str = '创建路径 %s: %s' % (path_name_str, path)
            output_log_record(log_fn, log_str)
            os.makedirs(path)


def check_path_exist_else_exit(path, path_name_str, log_fn):
    """
    判断路径是否存在，如果路径不存在，则程序退出！
    Args:
        path:                  目标路径 (str/list/dict)
        path_name_str:         路径的名称
        log_fn:                日志文件名

    Returns:
        None

    """
    if type(path) is dict:
        for k, v in path.items():
            if os.path.exists(v) is False:
                err_str = 'path %s[%s]: [%s] 不存在！程序退出' % (path_name_str, k, v)
                output_log_record(log_fn, err_str)
                sys.exit(1)
    elif type(path) is list:
        for i_path in path:
            if os.path.exists(i_path) is False:
                err_str = 'path %s: %s 不存在！程序退出' % (path_name_str, i_path)
                output_log_record(log_fn, err_str)
                sys.exit(1)
    else:
        if os.path.exists(path) is False:
            err_str = 'path %s: [%s] 不存在！程序退出' % (path_name_str, path)
            output_log_record(log_fn, err_str)
            sys.exit(1)


def save_obj(obj, name, to_path):
    """
    存储 object, 至位置 to_path, save as file name: 'name.pkl'
    as wb format
    Args:
        obj:                 待存储的对象
        name:                obj的名字，同时也是存储的文件名(不带 .pkl 后缀)
        to_path:             存储路径

    Returns:
        None

    """
    fn = '%s/%s.pkl' % (to_path, name)
    check_fn_available(fn)
    with open(fn, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# end of save_obj


def load_obj(name, from_path):
    """
    从路径 from_path 读取 object
    Args:
        name:             obj的名字，同时也是存储的文件名(不带 .pkl 后缀)
        from_path:        obj file 存储路径

    Returns:
        None/obj

        None: 如果读取失败，例如文件损坏或不存在, return None
        obj:  读取成功

    """
    try:
        with open('%s/%s.pkl' % (from_path, name), 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print('读取 obj: %s 失败，返回None, %s' % (name, e))
        return None


def convert_npy_2_npz(path, b_del_npy_files, b_sub_dir):
    """
    本函数负责将文件夹内所有 npy 文件转换为 npz 文件
    npy为非压缩数组
    npz为压缩数组
    Args:
        path:                  带转换的目录
        b_del_npy_files:      是否删除旧的 npy 文件
        b_sub_dir:            是否对文件夹的子文件夹进行同样操作

    Returns:
        None

    """
    for root, dirs, files in os.walk(path):

        for f1 in files:
            # 是 .npy 文件
            if f1[-4:] == '.npy':
                f11 = '%s/%s' % (dir, f1)
                f2 = '%s/%s.npz' % (dir, f1[:-4])

                print('convert [%s] -> [%s]' % (f11, f2))
                np.savez_compressed(f2, np.load(f11))

                if b_del_npy_files:
                    os.remove(f11)
        # end of loop files

        # 递归调用本函数处理子文件夹
        if b_sub_dir:
            for i_path in dirs:
                convert_npy_2_npz(i_path, b_del_npy_files, b_sub_dir)


def get_now_time_str():
    """
    获取字符串格式的当前时间
    """
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def datatype_2_str(i_struct, n_blanks: int = 0):
    """
    将字典格式转换为字符串
    每一个val, key 以 换行相隔
    Args:
        i_struct:       目标数据类型
        n_blanks:       行前空格数

    Returns:

    """
    cur_sp = ' ' * n_blanks
    cur_sub_sp = ' ' * (n_blanks + 4)

    if type(i_struct) is dict:
        i_str_list = []
        mid_str = ''
        for i_key, i_val in i_struct.items():
            i_str_list.append(cur_sub_sp + str(i_key) + ': ' + datatype_2_str(i_val, n_blanks + 4))
            mid_str = '\n'.join(i_str_list)
        return cur_sp + '{\n' + mid_str + cur_sp + '}\n'

    elif type(i_struct) is list:
        tmp_str = ''
        for i_val in i_struct:
            tmp_str += (datatype_2_str(i_val, n_blanks + 4) + ' ,')
        return '[\n' + cur_sub_sp + tmp_str + '\n' + cur_sub_sp + ']'
    elif type(i_struct) is tuple:
        tmp_str = ''
        for i_val in i_struct:
            tmp_str += (datatype_2_str(i_val, n_blanks + 4) + ' ,')
        return '(' + tmp_str + ')'
    elif type(i_struct) is set:
        tmp_str = ''
        for i_val in i_struct:
            tmp_str += (datatype_2_str(i_val, n_blanks + 4) + ' ,')
        return '{' + tmp_str + '}'
    else:
        str_mid = str(i_struct)
        return str_mid


def trans_cvContours_to_lonlat(contours, wst, sou, dlon, dlat):
    """
    将 cv.findContours 提取获得的轮廓转换为 经纬度 格式
    Parameters:
    -----------
    contours:           轮廓数据 单位是像素
                        = [countour_1, countour_2, ...]
                        type(contour_i ) = np.ndarray
                        countour_i.shape = [npoints, 1, 2]
                        contour_i = [ [[x1, y1]], 
                                      [[x2, y2]], 
                                      ...
                                      [[xn, yn]]  ]
                        
    wst:                图像x轴 0坐标 对应的 经度值
    sou:                图像y轴 0坐标 对应的 纬度值
    dlon:               经度步长
    dlat:               纬度步长
    Returns:
    --------
    contours_lonlat     经纬度转换后的轮廓数据 
                        = [contour_1, contour_2, ...]
                        contour_i shape = [npoints, 2]
                        contour_i = [ [lon1, lat1], 
                                      [lon2, lat2], 
                                      ...
                                      [lonn, latn]  ]
    """
    contours_lonlat = []
    for icontour in contours: 
        ic_lonlat = np.zeros((icontour.shape[0], 2))
        for ii in range(icontour.shape[0]):
            ic_lonlat[ii, 0] = wst + icontour[ii, 0, 0] * dlon
            ic_lonlat[ii, 1] = sou + icontour[ii, 0, 1] * dlat

        contours_lonlat.append(ic_lonlat)

    return contours_lonlat


if __name__ == '__main__':
    date_1 = datetime.strptime("20211120", "%Y%m%d")
    date_2 = datetime.strptime("20210101", "%Y%m%d")
    date_3 = datetime(2021, 11, 20)
    pass
