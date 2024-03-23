from pprint import pprint
from utils import load_partial_dataset, plotXY, plotSequentially, measure_azimuth_record_wise, measure_azimuth_vehicle_wise

LOC_01 = "pNEUMA-Loc01-0900-0930.csv"
LOC_10 = "pNEUMA-Loc10-0900-0930.csv"

dataset = load_partial_dataset(                                         ## loading dataset with potential noise nodes removed
    f"E:\ML_Project\ML_2024_10_ProPredictors\Dataset\{LOC_10}",
    ";",
    "type",
    ["Motorcycle", "Taxi", "Bus"],
    _verbose=True)

# azm_lst_record = measure_azimuth_record_wise(dataset[1], dataset)
azm_dct_vehicle = measure_azimuth_vehicle_wise(dataset, use_avg=True)
plotSequentially(dataset, "lat", "lon", "Latitude", "Longitude", )
plotXY(dataset, "lat", "lon", "Latitude", "Longitude")

# print(azm_lst_record)
# pprint(azm_dct_vehicle)

# plotXY(dataset, "lat", "lon", "Latitude", "Longitude", use_azm=True, azm_dct=azm_dct_vehicle)

"""

data:
data
[1.72, 1.72, 1.72, 1.72, 1.72, 1.72, 1.72, 1.72, 1.72, 2.09, 2.09, 2.09, 2.09, 2.09, 2.09, 2.09, 2.09, 2.09, 2.09, 2.09, 2.09, 2.09, 2.09, 2.09, 2.09, 2.09, 2.09, 2.09, 2.09, 2.09, 2.09, 2.09, 2.38, 2.38, 2.38, 2.38, 2.38, 2.38, 2.38, 2.38, 2.38, 2.38, 2.38, 2.38, 2.38, 2.38, 2.38, 2.38, 2.38, 2.38, 2.38, 2.38, 2.38, 2.38, 2.38, 2.38, 2.38, 2.38, 2.38, 2.46, 2.46, 2.46, 2.46, 2.46, 2.46, 2.46, 2.46, 2.46, 2.46, 2.46, 2.46, 2.46, 2.46, 2.46, 2.46, 2.46, 2.46, 2.78, 2.78, 2.78, 2.78, 2.78, 2.78, 2.78, 2.78, 2.78, 2.78, 2.78, 2.78, 2.78, 2.78, 2.78, 2.78, 2.78, 2.78, 3.03, 3.03, 3.03, 3.03, 3.21, 3.21, 3.21, 3.21, 3.21, 3.54, 3.92, 3.92, 3.92, 3.92, 3.92, 3.92, 4.12, 4.12, 4.12, 4.12, 4.12, 4.48, 4.48, 4.48, 4.48, 4.48, 4.48, 4.48, 4.72, 4.72, 4.72, 4.72, 4.72, 5.01, 5.01, 5.01, 5.01, 5.01, 5.01, 5.01, 5.01, 5.01, 5.01, 5.01, 5.01, 5.01, 5.01, 5.01, 5.01, 5.01, 5.24, 5.24, 5.24, 5.24, 5.24, 5.24, 5.24, 5.24, 5.24, 5.24, 5.24, 5.24, 5.24, 5.24, 5.24, 5.24, 5.24, 5.24, 5.24, 5.24, 5.24, 5.24, 5.24, 5.24, 5.24, 5.24, 5.24, 5.24, 5.24, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.62, 5.62, 5.62, 5.62, 5.62, 5.62, 5.62, 5.62, 5.62, 5.62, 5.62, 5.62, 5.62, 5.62, 5.62, 5.62, 5.62, 5.62, 5.62, 5.62, 5.62, 5.62, 5.62, 5.62, 5.62, 5.87, 5.87, 5.87, 5.87, 5.87, 5.87, 5.87, 5.87, 5.87, 5.87, 5.87, 5.87, 5.87, 5.87, 5.87, 5.87, 5.87, 5.87, 5.87, 5.87, 6.13, 6.13, 6.13, 6.13, 6.13, 6.13, 6.13, 6.13, 6.86, 7.01, 7.01, 7.13, 7.13, 7.48, 7.48, 8.14, 8.14, 8.14, 8.14, 8.14, 8.14, 8.14, 8.44, 8.44, 8.44, 8.44, 8.44, 8.44, 8.44, 8.44, 8.44, 8.44, 8.44, 8.44, 8.44, 8.56, 8.56, 8.56, 8.56, 8.56, 8.56, 8.56, 8.56, 8.56, 8.56, 8.56, 8.56, 8.56, 8.56, 8.56, 8.56, 8.56, 8.56, 8.56, 8.56, 8.56, 8.56, 8.56, 8.56, 8.56, 8.56, 8.56, 8.56, 8.56, 8.56, 8.56, 8.56, 8.56, 8.56, 8.56, 8.56, 8.56, 8.56, 8.56, 8.56, 8.56, 8.56, 8.56, 8.56, 8.76, 8.76, 8.76, 8.76, 8.76, 8.76, 8.76, 8.76, 8.76, 8.76, 8.76, 8.76, 8.76, 8.76, 8.76, 8.76, 8.76, 8.76, 8.76, 8.76, 8.76, 8.76, 8.76, 8.76, 8.76, 8.76, 8.76, 8.76, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.38, 9.38, 9.38]

Jenks algo:
def jenks_breaks(data, num_classes):

    # Flatten the histogram bins to form the data
    data = np.concatenate(data)

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
    kclass = [0] * (num_classes + 1)
    kclass[num_classes] = float(data[n - 1])
    for j in range(num_classes, 1, -1):
        count_num = int(mat2[k][j])
        kclass[j - 1] = float(data[count_num])
        k = int(count_num) + 1
    return kclass[1:]

Expected Output:
[15.2, 40.0, 70.1, 98.3]

Generated Output:
[0, 42.0, 80.0, 95.0]

"""