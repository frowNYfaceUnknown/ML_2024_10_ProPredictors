from utils import load_partial_dataset, extract_hist_from_bounding_box

x_bounds_subloc_1 = range(3080, 3940 + 1)
y_bounds_subloc_1 = range(1060, 1220 + 1)

x_bounds_subloc_2 = range(1710, 2060 + 1)
y_bounds_subloc_2 = range(1080, 1175 + 1)

x_bounds_subloc_3 = range(2280, 2870 + 1)
y_bounds_subloc_3 = range(1425, 1530 + 1)

## to reproduce subloc 1 images
extract_hist_from_bounding_box(
    "00001.jpg",
    loc_x_bounds=x_bounds_subloc_1,
    loc_y_bounds=y_bounds_subloc_1,
    start_time=0,
    time_stamps=10  # 100
)

## to reproduce subloc 2 images
extract_hist_from_bounding_box(
    "00001.jpg",
    loc_x_bounds=x_bounds_subloc_2,
    loc_y_bounds=y_bounds_subloc_2,
    start_time=106,
    time_stamps=10  # 100
)

## to reproduce subloc 3 images
extract_hist_from_bounding_box(
    "00001.jpg",
    loc_x_bounds=x_bounds_subloc_3,
    loc_y_bounds=y_bounds_subloc_3,
    start_time=86,
    time_stamps=10  # 100
)