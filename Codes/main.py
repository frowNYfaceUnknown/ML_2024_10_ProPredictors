from utils import load_partial_dataset, plotXY, measure_azimuth

dataset = load_partial_dataset(                                         ## loading dataset with potential noise nodes removed
    "E:\ML_Project\ML_2024_10_ProPredictors\Dataset\pNEUMA-Loc1.csv",
    ";",
    "type",
    ["Motorcycle", "Taxi", "Bus"],
    _verbose=True)

plotXY(dataset, "lat", "lon")
# azm_lst = measure_azimuth(dataset)