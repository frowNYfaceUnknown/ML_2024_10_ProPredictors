import os
import math
import numpy
import numpy as np
import matplotlib.cm as cm
from statistics import fmean
from typing import Any, Union
import matplotlib.pyplot as plt

def change_delimiter(_dataset: str, _dir: str = None, _from: str = ";", _to: str = ",", _out_name: str = "edited.csv") -> bool:
    """
        Change delimiter of a CSV file, <_dataset>, from <_from> to <_to> located at <_dir>. Returns True if successful, otherwise False.

        NOTE: If <_dir> is not provided, the func expects full path to the dataset in <_dataset> and the output <_out_name> will be saved in the same directory as utils.py
    """
    if _dir != None:
        READ_FILE_PATH = os.path.join(_dir, _dataset)
        WRITE_FILE_PATH = os.path.join(_dir, _out_name)
    else:
        READ_FILE_PATH = _dataset
        WRITE_FILE_PATH = _out_name

    try:
        rfd = open(READ_FILE_PATH, "a+")
        rfd.seek(0)

        wfd = open(WRITE_FILE_PATH, "w")

        while True:

            line = rfd.readline()
            if not line:
                break

            line_vec = line.split(_from)
            new_line = _to.join(line_vec)
            
            wfd.write(new_line)

        rfd.close()
        wfd.close()

    except Exception as e:
        return False

    return True

def load_partial_dataset(                       ## edit later so that instead of _filter_column_name and _remove_list, we simply pass a
        _dataset: str,                          ## _filter :dict: and that filters out the col name along with the list of values to be
        _sep: str,                              ## filtered from that col. Essentially allowing to filter multiple cols while loading!!
        _filter_column_name: str,
        _remove_list: list[str],
        _verbose: bool = False
    ) -> list[list[str]]:

    try:
        fd = open(_dataset, "r")
        lines = fd.readlines()
        return_list = []

        i = _find_index_from_name([ele.strip() for ele in lines[0].split(_sep)], _filter_column_name)
        if i == -1:
            raise KeyError(f"Column Name {_filter_column_name} not found in dataset: {_dataset}")

        if _verbose:
            print("Length before removing:", len(lines))

        _remove_list = [ele.casefold() for ele in _remove_list]
        for line in lines:
            
            lst = line.split(_sep)
            lst = [ele.strip().casefold() for ele in lst]

            if lst[i] not in _remove_list:
                return_list.append(lst)

        if _verbose:
            print("Length after removing:", len(return_list))
        
        return return_list
    
    except Exception as err:
        raise Exception(f"Error occured: {err}")

def _find_index_from_name(target: list[str], filter: str) -> int:

    for idx, name in enumerate(target):

        if name.casefold() == filter.casefold():
            return idx
    
    return -1

def plotXY(
        _dataset: list[list[str]], x_coords: str, y_coords: str,
        x_label: str = None, y_label: str = None,
        use_azm: bool = False, azm_dct = None
    ) -> None:

    if x_label == None:
        x_label = x_coords
    
    if y_label == None:
        y_label = y_coords

    idx_x = _find_index_from_name(_dataset[0], x_coords)
    if idx_x == -1:
        raise KeyError(f"Column Name {x_coords} not found in dataset: {_dataset}")

    idx_y = _find_index_from_name(_dataset[0], y_coords)
    if idx_y == -1:
        raise KeyError(f"Column Name {y_coords} not found in dataset: {_dataset}")
    
    list_x = [float(data[idx_x]) for data in _dataset[1:]]
    list_y = [float(data[idx_y]) for data in _dataset[1:]]

    colors = iter(cm.rainbow(np.linspace(0, 1, len(list_x))))
    for x, y in [(list_x[i], list_y[i]) for i in range(len(list_x))]:
        plt.plot(x, y, 'o', color=next(colors))

    print( min(list_x), max(list_x), min(list_y), max(list_y) )
    # plt.axis(( min(list_x), max(list_x), min(list_y), max(list_y) ))
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if use_azm:
        
        if azm_dct == None:
            azm_dct = measure_azimuth_vehicle_wise(_dataset)
        
        colors = iter(cm.rainbow(np.linspace(0, 1, len(list_x))))
        for i in range(len(list_x)):
            print(i, int(_dataset[1:][i][0]))
            slope = 1/(math.tan(azm_dct[int(_dataset[1:][i][0])]))
            offset = list_y[i] - (slope * list_x[i])
            xrange = list(numpy.linspace(min(list_x), max(list_x), 10))
            yrange = [(slope * xrange[j]) + offset for j in range(len(xrange))]
            print(xrange, yrange)
            plt.plot(xrange, yrange, color=next(colors))
            plt.show()

    plt.show()

def measure_azimuth_record_wise(                ## [NOTE] code very specific to the dataset [!!!]
        _record: list[str],
        _dataset: list[list[str]]
    ) -> list[int]:

    lat_idx = _find_index_from_name(_dataset[0], "lat")
    lon_idx = _find_index_from_name(_dataset[0], "lon")
        
    recordList = []

    lats = _extract_from(_record, lat_idx)
    lons = _extract_from(_record, lon_idx)

    for idx in range(len(lons) - 1):            ## [TODO] or lats, should not matter, verify later [!] -- verified [√]

        del_lons = lons[idx + 1] - lons[idx]

        azx = math.atan2(
            math.sin(del_lons) * math.cos(lats[idx + 1]),
            ( math.cos(lats[idx]) * math.sin(lats[idx + 1]) ) - ( math.sin(lats[idx]) * math.cos(lats[idx + 1]) * math.cos(del_lons) )
        )

        recordList.append(azx)
    
    return recordList

def _extract_from(record: list[str], idx: int, step: int = 6) -> list[float]:         ## default step value is six, because the values repeat at every sixth index in the dataset

    returnList = []
    
    while idx < len(record):
        try:
            returnList.append(float(record[idx]))
        except ValueError as err:
            continue
        finally:
            idx += step
    
    return returnList

def measure_azimuth_vehicle_wise(               ## [NOTE] code very specific to the dataset [!!!]
        _dataset: list[list[str]],
        start: int = 0,                         ## multiplier to the step value of 6. Essentially means the time time stamp of the starting index
        end: int = 1,                           ## multiplier to the step value of 6. Essentially means the time difference between the two records
        use_avg: bool = False
    ) -> dict[int, float]:
    
    lat_idx = _find_index_from_name(_dataset[0], "lat")
    lon_idx = _find_index_from_name(_dataset[0], "lon")

    recordDict = {}

    for data in _dataset[1:]:

        if not use_avg:
            del_lons = float(data[lon_idx + (6 * start) + (6 * end)]) - float(data[lon_idx + (6 * start)])
            lats_end = float(data[lat_idx + (6 * start) + (6 * end)])
            lats_start = float(data[lat_idx + (6 * start)])

            azx = math.atan2(
                math.sin(del_lons) * math.cos(lats_end),
                ( math.cos(lats_start) * math.sin(lats_end) ) - ( math.sin(lats_start) * math.cos(lats_end) * math.cos(del_lons) )
            )
        else:
            azx = fmean(measure_azimuth_record_wise(data, _dataset))

        recordDict[int(data[0])] = azx
    
    return recordDict

def plotSequentially(
        _dataset: list[list[str]], x_coords: str, y_coords: str,
        x_label: str = None, y_label: str = None
    ) -> None:

    if x_label == None:
        x_label = x_coords
    
    if y_label == None:
        y_label = y_coords

    idx_x = _find_index_from_name(_dataset[0], x_coords)
    if idx_x == -1:
        raise KeyError(f"Column Name {x_coords} not found in dataset: {_dataset}")

    idx_y = _find_index_from_name(_dataset[0], y_coords)
    if idx_y == -1:
        raise KeyError(f"Column Name {y_coords} not found in dataset: {_dataset}")
    
    list_x = [float(data[idx_x]) for data in _dataset[1:]]
    list_y = [float(data[idx_y]) for data in _dataset[1:]]

    colors = iter(cm.rainbow(np.linspace(0, 1, len(list_x))))
    for i, (x, y) in enumerate([(list_x[i], list_y[i]) for i in range(len(list_x))]):
        
        plt.plot(x, y, 'o', color=next(colors))

        lat_idx = _find_index_from_name(_dataset[0], "lat")
        lon_idx = _find_index_from_name(_dataset[0], "lon")
        
        latList = _extract_from(_dataset[1:][i], lat_idx)
        lonList = _extract_from(_dataset[1:][i], lon_idx)

        plt.plot(latList, lonList)
    
    plt.show()