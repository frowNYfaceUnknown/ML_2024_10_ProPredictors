from utils import extract_hist_from_bounding_box, jenks_breaks, plot_classes_hist, elbow_method

x_bounds_subloc_1 = range(3080, 3940 + 1)
y_bounds_subloc_1 = range(1060, 1220 + 1)

x_bounds_subloc_2 = range(1710, 2060 + 1)
y_bounds_subloc_2 = range(1080, 1175 + 1)

x_bounds_subloc_3 = range(2280, 2870 + 1)
y_bounds_subloc_3 = range(1425, 1530 + 1)

## to reproduce subloc 1 images
data = extract_hist_from_bounding_box(
    "00001.jpg",
    loc_x_bounds=x_bounds_subloc_1,
    loc_y_bounds=y_bounds_subloc_1,
    start_time=0,
    time_stamps=10
)
max_num_classes = 10  # Maximum number of classes to consider
elbow_method(data)
plot_classes_hist(data, jenks_breaks(data, 5))

data = extract_hist_from_bounding_box(
    "00001.jpg",
    loc_x_bounds=x_bounds_subloc_1,
    loc_y_bounds=y_bounds_subloc_1,
    start_time=0,
    time_stamps=100
)
elbow_method(data)
plot_classes_hist(data, jenks_breaks(data, 5))

data = extract_hist_from_bounding_box(
    "00001.jpg",
    loc_x_bounds=x_bounds_subloc_1,
    loc_y_bounds=y_bounds_subloc_1,
    start_time=0,
    time_stamps=10,
    filters=["Motorcycle", "Taxi"]
)
elbow_method(data)
plot_classes_hist(data, jenks_breaks(data, 4))

data = extract_hist_from_bounding_box(
    "00001.jpg",
    loc_x_bounds=x_bounds_subloc_1,
    loc_y_bounds=y_bounds_subloc_1,
    start_time=0,
    time_stamps=100,
    filters=["Motorcycle", "Taxi"]
)
elbow_method(data)
plot_classes_hist(data, jenks_breaks(data, 4))

## to reproduce subloc 2 images
data = extract_hist_from_bounding_box(
    "00001.jpg",
    loc_x_bounds=x_bounds_subloc_2,
    loc_y_bounds=y_bounds_subloc_2,
    start_time=106,
    time_stamps=10
)
elbow_method(data)
plot_classes_hist(data, jenks_breaks(data, 4))

data = extract_hist_from_bounding_box(
    "00001.jpg",
    loc_x_bounds=x_bounds_subloc_2,
    loc_y_bounds=y_bounds_subloc_2,
    start_time=106,
    time_stamps=100
)
elbow_method(data)
plot_classes_hist(data, jenks_breaks(data, 5))          ## incorrect k-value predicted by elbow method

data = extract_hist_from_bounding_box(
    "00001.jpg",
    loc_x_bounds=x_bounds_subloc_2,
    loc_y_bounds=y_bounds_subloc_2,
    start_time=106,
    time_stamps=10,
    filters=["Motorcycle", "Taxi"]
)
elbow_method(data)
plot_classes_hist(data, jenks_breaks(data, 3))

data = extract_hist_from_bounding_box(
    "00001.jpg",
    loc_x_bounds=x_bounds_subloc_2,
    loc_y_bounds=y_bounds_subloc_2,
    start_time=106,
    time_stamps=100,
    filters=["Motorcycle", "Taxi"]
)
elbow_method(data)
plot_classes_hist(data, jenks_breaks(data, 6))          ## incorrect k-value predicted by elbow method

## to reproduce subloc 3 images
data = extract_hist_from_bounding_box(
    "00001.jpg",
    loc_x_bounds=x_bounds_subloc_3,
    loc_y_bounds=y_bounds_subloc_3,
    start_time=86,
    time_stamps=10
)
elbow_method(data)
plot_classes_hist(data, jenks_breaks(data, 4))

data = extract_hist_from_bounding_box(
    "00001.jpg",
    loc_x_bounds=x_bounds_subloc_3,
    loc_y_bounds=y_bounds_subloc_3,
    start_time=86,
    time_stamps=100
)
elbow_method(data)
plot_classes_hist(data, jenks_breaks(data, 4))

data = extract_hist_from_bounding_box(
    "00001.jpg",
    loc_x_bounds=x_bounds_subloc_3,
    loc_y_bounds=y_bounds_subloc_3,
    start_time=86,
    time_stamps=10,
    filters=["Motorcycle", "Taxi"]
)
elbow_method(data)
plot_classes_hist(data, jenks_breaks(data, 4))          ## sort-of incorrect ? Should be 3 because Taxis were removed

data = extract_hist_from_bounding_box(
    "00001.jpg",
    loc_x_bounds=x_bounds_subloc_3,
    loc_y_bounds=y_bounds_subloc_3,
    start_time=86,
    time_stamps=100,
    filters=["Motorcycle", "Taxi"]
)
elbow_method(data)
plot_classes_hist(data, jenks_breaks(data, 4))          ## sort-of incorrect ? Should be 3 because Taxis were removed