import os
import math
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

def load_partial_dataset(           ## edit later so that instead of _filter_column_name and _remove_list, we simply pass a
        _dataset: str,              ## _filter :dict: and that filters out the col name along with the list of values to be
        _sep: str,                  ## filtered from that col. Essentially allowing to filter multiple cols while loading!!
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

def plotXY(_dataset: list[list[str]], x_coords: str, y_coords: str, x_label: str = None, y_label: str = None) -> None:

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

    plt.plot(list_x, list_y, 'ro')
    print( min(list_x), max(list_x), min(list_y), max(list_y) )
    plt.axis(( min(list_x), max(list_x), min(list_y), max(list_y) ))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def measure_azimuth(_dataset: list[list[str]]) -> list[list[int]]:        ## [NOTE] code very specific to the dataset [!!!]

    returnList = []

    for record in _dataset:
        
        recordList = []

        lats = _extract_from(record, "lat")
        lons = _extract_from(record, "lon")

        for idx in range(len(lons) - 1):            ## [TODO] or lats, should not matter, verify later [!]

            del_lons = lons[idx + 1] - lons[idx]

            azx = math.atan2(
                math.sin(del_lons) * math.cos(lats[idx + 1]),
                ( math.cos(lats[idx]) * math.sin(lats[idx + 1]) ) - ( math.sin(lats[idx]) * math.cos(lats[idx + 1]) * math.cos(del_lons) )
            )

            recordList.append(azx)

        returnList.append(recordList)

def _extract_from(record: list[str], col: str, step: int = 6) -> list[int]:         ## default step value is six, because the values repeat at every sixth index in the dataset

    raise NotImplementedError ## for now