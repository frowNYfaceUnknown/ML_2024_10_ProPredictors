from typing import Any, Union, Type

from os import listdir
from os.path import join, isfile

import math
import numpy as np
from statistics import fmean
from sklearn.cluster import KMeans

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

LOC_WIN = 'E:\\ML_Project\\ML_2024_10_ProPredictors\\Dataset\\pNEUMA_V-Loc10-0900-0930'
LOC_DRV = '/content/drive/MyDrive/ML Datasets/pNEUMA - Vision/pNEUMA_V-Loc10-0900-0930'

FRAME_LOC_DRV = join(LOC_DRV, 'Frames')
ANNOT_LOC_DRV = join(LOC_DRV, 'Annotations')

FRAME_LOC_WIN = join(LOC_WIN, 'Frames')
ANNOT_LOC_WIN = join(LOC_WIN, 'Annotations')

def change_delimiter(_dataset: str, _dir: str = None, _from: str = ";", _to: str = ",", _out_name: str = "edited.csv") -> bool:
    """
        Change delimiter of a CSV file, <_dataset>, from <_from> to <_to> located at <_dir>. Returns True if successful, otherwise False.

        NOTE: If <_dir> is not provided, the func expects full path to the dataset in <_dataset> and the output <_out_name> will be saved in the same directory as utils.py
    """
    if _dir != None:
        READ_FILE_PATH = join(_dir, _dataset)
        WRITE_FILE_PATH = join(_dir, _out_name)
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
        _filter_column_name: str = None,
        _remove_list: list[str] = [],
        _verbose: bool = False
    ) -> list[list[str]]:

    try:
        fd = open(_dataset, "r")
        lines = fd.readlines()
        return_list = []

        if _filter_column_name != None:
            i = _find_index_from_name([ele.strip() for ele in lines[0].split(_sep)], _filter_column_name)
            if i == -1:
                raise KeyError(f"Column Name {_filter_column_name} not found in dataset: {_dataset}")

        if _verbose:
            print("Length before removing:", len(lines))

        _remove_list = [ele.casefold() for ele in _remove_list]
        for line in lines:

            lst = line.split(_sep)
            lst = [ele.strip().casefold() for ele in lst]

            if (_filter_column_name == None) or (lst[i] not in _remove_list):
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

## pNeuma Dataset Related Code

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
            xrange = list(np.linspace(min(list_x), max(list_x), 10))
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

## pNeuma Vision Dataset Related Code

def extract_hist_from_bounding_box(
        image_to_load: str,
        loc_x_bounds: Type[range],
        loc_y_bounds: Type[range],
        start_time: int = 0,
        time_stamps: int = 10,
        filters: list[str] = ["Motorcycle"],    ## , "Taxi", "Bus"
        save_hist: bool = False,                ## yet to implement
    ) -> list[int]:
    
    image = Image.open(join(FRAME_LOC_WIN, image_to_load))
    marks = sorted([file for file in listdir(ANNOT_LOC_WIN) if isfile(join(ANNOT_LOC_WIN, file))])

    draw  = ImageDraw.Draw(image)
    draw.rectangle((loc_x_bounds.start, loc_y_bounds.start, loc_x_bounds.stop, loc_y_bounds.stop), outline="black", width=10)

    data_xs, data_ys = [], []
    for i in range(start_time, start_time + time_stamps):
        dataset = load_partial_dataset(join(ANNOT_LOC_WIN, marks[i]), ',', "Type", filters)
        for data in dataset[1:]:
            if int(data[3]) in loc_x_bounds and int(data[4]) in loc_y_bounds:
                data_xs.append(int(data[3]))
                data_ys.append(int(data[4]))
                _draw_circle(image, (int(data[3]), int(data[4])), 10)

    ys = sorted(data_ys)
    # min_y, max_y = min(data_ys), max(data_ys)
    # norm_ys = [(y - min_y)/(max_y - min_y)*10 for y in data_ys]
    # print(_jenks_breaks(ys, 5))
    plt.hist(ys, bins=10)
    plt.xlabel("Pixel y-coordinate (top to bottom in image)")
    plt.ylabel("No. of vehicles")

    image.show()
    plt.show()

    return ys

def _draw_circle(image, center, radius):

    draw = ImageDraw.Draw(image)

    # Calculate the bounding box of the circle
    x0 = center[0] - radius
    y0 = center[1] - radius
    x1 = center[0] + radius
    y1 = center[1] + radius

    # Draw the circle
    draw.ellipse([x0, y0, x1, y1], fill="black")

def plot_classes_hist(
        data: list[int], classes: list[float]
    ) -> None:

    out: list[list[int]] = []
    for i in range(len(classes) - 1):
        out.append([])

    for i in range(len(classes) - 1):
        for ele in data:
            if (ele > classes[i]) and (ele < classes[i+1]):
                out[i].append(ele)
    print(out)
    plt.hist(out)
    plt.show()

def jenks_breaks(data, num_classes) -> list[float]:

    # Flatten the histogram bins to form the data
    # data = np.concatenate(data)

    # Ensure data is sorted
    data = sorted(data)

    # Initialization
    n = len(data)
    mat1 = np.zeros((n + 1, num_classes + 1))
    mat2 = np.zeros((n + 1, num_classes + 1))
    for i in range(1, n + 1):
        mat1[i][1] = 1
        mat2[i][1] = 0
        for j in range(2, num_classes + 1):
            mat2[i][j] = -1
            mat1[i][j] = float('inf')

    # Calculation
    v = 0.0
    for l in range(2, n + 1):
        s1 = 0.0
        s2 = 0.0
        w = 0.0
        for m in range(1, l + 1):
            i3 = l - m + 1
            val = float(data[i3 - 1])
            s2 += val * val
            s1 += val
            w += 1
            v = s2 - (s1 * s1) / w
            i4 = i3 - 1
            if i4 != 0:
                for j in range(2, num_classes + 1):
                    if mat1[l][j] >= (v + mat1[i4][j - 1]):
                        mat2[l][j] = i3 - 1
                        mat1[l][j] = v + mat1[i4][j - 1]
        mat1[l][1] = v

    # Backtracking
    k = n
    kclass: list[float] = [0] * (num_classes + 1)
    kclass[num_classes] = float(data[n - 1])
    for j in range(num_classes, 0, -1):
        count_num = int(mat2[k][j])
        kclass[j - 1] = float(data[count_num])
        k = int(count_num) + 1
    return kclass

def elbow_method(features):

    features = np.array(features).reshape(-1, 1)

    # Apply K-Means with multiple cluster numbers to demonstrate the elbow method
    inertia_values = []
    cluster_range = range(1, 11)

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(features)
        inertia_values.append(kmeans.inertia_)

    # Plotting the Elbow Method graph
    plt.figure(figsize=(8, 6))
    plt.plot(cluster_range, inertia_values, '-o', color='blue')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.xticks(cluster_range)
    plt.grid(True)
    plt.show()