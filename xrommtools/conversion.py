from __future__ import annotations

import pandas as pd


def dlc_to_xma(cam1data, cam2data, trialname, savepath):
    h5_save_path = savepath + "/" + trialname + "-Predicted2DPoints.h5"
    csv_save_path = savepath + "/" + trialname + "-Predicted2DPoints.csv"

    if isinstance(cam1data, str):  # is string
        if ".csv" in cam1data:
            cam1data = pd.read_csv(cam1data, sep=",", header=None)
            cam2data = pd.read_csv(cam2data, sep=",", header=None)
            pointnames = list(cam1data.loc[1, 1:].unique())

            # reformat CSV / get rid of headers
            cam1data = cam1data.loc[3:, 1:]
            cam1data.columns = range(cam1data.shape[1])
            cam1data.index = range(cam1data.shape[0])
            cam2data = cam2data.loc[3:, 1:]
            cam2data.columns = range(cam2data.shape[1])
            cam2data.index = range(cam2data.shape[0])

        elif ".h5" in cam1data:  # is .h5 file
            cam1data = pd.read_hdf(cam1data)
            cam2data = pd.read_hdf(cam2data)
            pointnames = list(cam1data.columns.get_level_values("bodyparts").unique())

        else:
            raise ValueError("2D point input is not in correct format")
    else:
        pointnames = list(cam1data.columns.get_level_values("bodyparts").unique())

    # make new column names
    nvar = len(pointnames)
    pointnames = [item for item in pointnames for repetitions in range(4)]
    post = ["_cam1_X", "_cam1_Y", "_cam2_X", "_cam2_Y"] * nvar
    cols = [m + str(n) for m, n in zip(pointnames, post)]

    # remove likelihood columns
    cam1data = cam1data.drop(cam1data.columns[2::3], axis=1)
    cam2data = cam2data.drop(cam2data.columns[2::3], axis=1)

    # replace col names with new indices
    c1cols = list(range(0, cam1data.shape[1] * 2, 4)) + list(
        range(1, cam1data.shape[1] * 2, 4)
    )
    c2cols = list(range(2, cam1data.shape[1] * 2, 4)) + list(
        range(3, cam1data.shape[1] * 2, 4)
    )
    c1cols.sort()
    c2cols.sort()
    cam1data.columns = c1cols
    cam2data.columns = c2cols

    df = pd.concat([cam1data, cam2data], axis=1).sort_index(axis=1)
    df.columns = cols
    df.to_hdf(h5_save_path, key="df_with_missing", mode="w")
    df.to_csv(csv_save_path, na_rep="NaN", index=False)
