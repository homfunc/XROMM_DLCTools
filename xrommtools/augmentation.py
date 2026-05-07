from __future__ import annotations

import os

import cv2
import numpy as np
import pandas as pd

from .common import default_camera_substrings


def add_frames(
    path_config_file,
    data_path,
    iteration,
    frames,
    nnetworks=1,
    path_config_file_cam2="enterpathofcam2config",
):
    # input: config file paths, path of data to add to trainingdataset, frames-csv file where first col is trialnames and following cols are frame numbers
    # will look for 2D points file based on name (if there are multiple csv files)

    configs = [path_config_file[:-12], path_config_file_cam2[:-12]]
    cameras = [1, 2]
    subs = default_camera_substrings()
    pts = [
        "2Dpts",
        "2dpts",
        "2DPts",
        "2dPts",
        "pts2D",
        "Pts2D",
        "pts2d",
        "points2D",
        "Points2d",
        "points2d",
        "2Dpoints",
        "2dpoints",
        "2DPoints",
    ]
    corr = ["correct", "Correct", "corrected", "Corrected"]
    # read frames from csv
    if ".csv" in frames:
        f = pd.read_csv(frames, header=None)
        trialnames = list(f.iloc[:, 0])  # first row of frames file must be trialnames
        picked_frames = []
        # this is disgusting code
        for row in range(f.shape[0]):
            picked_frames.append(list(f.loc[row, 1:]))
        for count, row in enumerate(picked_frames):
            picked_frames[count] = [x for x in row if str(x) != "nan"]  # remove nans
        for count, row in enumerate(picked_frames):
            picked_frames[count] = [int(x) for x in row]  # convert to int
    else:
        raise ValueError("frames must be a .csv file with trialnames and frame numbers")

    if nnetworks == 2:
        for camera in cameras:
            contents = os.listdir(configs[camera - 1] + "/" + "labeled-data")
            if len(contents) == 1:
                dataset_name = contents[0]
                labeleddata_path = configs[camera - 1] + "/" + "labeled-data/" + dataset_name
            else:
                raise ValueError("There must be only one data set in the labeled-data folder")

            contents = os.listdir(labeleddata_path)
            h5file = [x for x in contents if ".h5" in x]
            csvfile = [x for x in contents if ".csv" in x]
            data = pd.read_hdf(labeleddata_path + "/" + h5file[0])  # read old point labels

            ## Extract selected frames from videos

            for trialnum, trial in enumerate(trialnames):
                # get video file
                file = []
                relnames = []
                contents = os.listdir(data_path + "/" + trial)
                for name in contents:
                    if any(x in name for x in subs[camera - 1]):
                        file = name
                if not file:
                    raise ValueError("Cannot locate %s video file or image folder" % trial)

                # if video file is actually folder of frames
                if os.path.isdir(data_path + "/" + trial + "/" + file):
                    imgpath = data_path + "/" + trial + "/" + file
                    imgs = os.listdir(imgpath)
                    relpath = "labeled-data/" + dataset_name + "/"
                    frames = picked_frames[trialnum]
                    frames.sort()

                    for count, img in enumerate(imgs):
                        if count + 1 in frames:  # ASSUMES FRAMES PROVIDED ARE 1 index
                            image = cv2.imread(imgpath + "/" + img)
                            relname = relpath + trial + "_%s.png" % str(count + 1).zfill(4)
                            relnames = relnames + [relname]
                            cv2.imwrite(
                                labeleddata_path + "/" + trial + "_%s.png" % str(count + 1).zfill(4),
                                image,
                            )  # save frame
                else:
                    # file is actually a file
                    # extract frames from video and convert to png
                    video = data_path + "/" + trial + "/" + file
                    relpath = "labeled-data/" + dataset_name + "/"
                    frames = picked_frames[trialnum]
                    frames.sort()
                    cap = cv2.VideoCapture(video)
                    success, image = cap.read()
                    count = 0
                    while success:
                        if count + 1 in frames:
                            relname = relpath + trial + "_%s.png" % str(count + 1).zfill(4)
                            relnames = relnames + [relname]
                            cv2.imwrite(
                                labeleddata_path + "/" + trial + "_%s.png" % str(count + 1).zfill(4),
                                image,
                            )  # save frame
                        success, image = cap.read()
                        count += 1
                    cap.release()

                # get 2D points file / data
                # extract 2D points data
                contents = os.listdir(data_path + "/" + trial + "/" + "it" + str(iteration))
                pointsfile = [x for x in contents if ".csv" in x]

                if not pointsfile:
                    raise ValueError("Cannot locate %s 2D points file" % trial)

                # if multiple csv files, look for "2Dpoints" in the name
                if len(pointsfile) > 1:
                    t = []
                    for q in pointsfile:
                        if any(x in q for x in pts):
                            t = t + [q]
                    # if there are multiple 2D points files, look for "corrected" in the name
                    if len(t) > 1:
                        for r in pointsfile:
                            if any(x in r for x in corr):
                                file = r
                    else:
                        file = t[0]
                else:
                    file = pointsfile[0]
                print(
                    "Reading and adding the following frames from "
                    + data_path
                    + "/"
                    + trial
                    + "/"
                    + "it"
                    + str(iteration)
                    + "/"
                    + file
                )
                df = pd.read_csv(
                    data_path + "/" + trial + "/" + "it" + str(iteration) + "/" + file,
                    sep=",",
                    header=None,
                )
                df = df.loc[1:,].reset_index(drop=True)
                print(frames)
                frames = [x - 1 for x in frames]  # account for zero index in python
                xpos = df.iloc[frames, 0 + (camera - 1) * 2 :: 4]
                ypos = df.iloc[frames, 1 + (camera - 1) * 2 :: 4]
                temp_data = pd.concat([xpos, ypos], axis=1).sort_index(axis=1)
                if temp_data.shape[1] > data.shape[1]:
                    raise ValueError(
                        "There are %d extra points in the corrected points file"
                        % ((temp_data.shape[1] - data.shape[1]) / 2)
                    )
                if temp_data.shape[1] < data.shape[1]:
                    raise ValueError(
                        "There are %d missing points in the corrected points file"
                        % ((data.shape[1] - temp_data.shape[1]) / 2)
                    )
                temp_data.index = relnames
                temp_data.columns = data.columns
                data = pd.concat([data, temp_data])
            data.replace(" NaN", np.nan, inplace=True)
            data.replace(" NaN ", np.nan, inplace=True)
            data.replace("NaN ", np.nan, inplace=True)
            data = data.astype("float")
            data = data.round(2)
            data = data.apply(pd.to_numeric)
            data.to_hdf(labeleddata_path + "/" + h5file[0], key="df_with_missing", mode="w")
            data.to_csv(labeleddata_path + "/" + csvfile[0], na_rep="NaN")

    else:  # default, one network for both videos
        config = path_config_file[:-12]
        contents = os.listdir(config + "/" + "labeled-data")
        if len(contents) == 1:
            dataset_name = contents[0]
            labeleddata_path = config + "/" + "labeled-data/" + dataset_name
        else:
            raise ValueError("There must be only one data set in the labeled-data folder")

        contents = os.listdir(labeleddata_path)
        h5file = [x for x in contents if ".h5" in x]
        csvfile = [x for x in contents if ".csv" in x]
        data = pd.read_hdf(labeleddata_path + "/" + h5file[0])  # read old point labels

        for camera in cameras:
            ## Extract selected frames from videos

            for trialnum, trial in enumerate(trialnames):
                # get video file
                relnames = []
                file = []

                contents = os.listdir(data_path + "/" + trial)
                for name in contents:
                    if any(x in name for x in subs[camera - 1]):
                        file = name
                if not file:
                    raise ValueError("Cannot locate %s video file or image folder" % trial)

                # if video file is actually folder of frames
                if os.path.isdir(data_path + "/" + trial + "/" + file):
                    imgpath = data_path + "/" + trial + "/" + file
                    imgs = os.listdir(imgpath)
                    relpath = "labeled-data/" + dataset_name + "/"
                    frames = picked_frames[trialnum]
                    frames.sort()

                    for count, img in enumerate(imgs):
                        if count + 1 in frames:  # ASSUMES FRAMES PROVIDED ARE 1 index
                            image = cv2.imread(imgpath + "/" + img)
                            relname = (
                                relpath + trial + "_cam" + str(camera) + "_%s.png" % str(count + 1).zfill(4)
                            )
                            relnames = relnames + [relname]
                            cv2.imwrite(
                                labeleddata_path + "/" + trial + "_cam" + str(camera) + "_%s.png" % str(count + 1).zfill(4),
                                image,
                            )  # save frame
                else:
                    # file is actually a file
                    # extract frames from video and convert to png
                    video = data_path + "/" + trial + "/" + file
                    relpath = "labeled-data/" + dataset_name + "/"
                    frames = picked_frames[trialnum]
                    frames.sort()
                    cap = cv2.VideoCapture(video)
                    success, image = cap.read()
                    count = 0
                    while success:
                        if count + 1 in frames:
                            relname = (
                                relpath + trial + "_cam" + str(camera) + "_%s.png" % str(count + 1).zfill(4)
                            )
                            relnames = relnames + [relname]
                            cv2.imwrite(
                                labeleddata_path + "/" + trial + "_cam" + str(camera) + "_%s.png" % str(count + 1).zfill(4),
                                image,
                            )  # save frame
                        success, image = cap.read()
                        count += 1
                    cap.release()

                # get 2D points file / data
                # extract 2D points data
                contents = os.listdir(data_path + "/" + trial + "/" + "it" + str(iteration))
                pointsfile = [x for x in contents if ".csv" in x]

                if not pointsfile:
                    raise ValueError("Cannot locate %s 2D points file" % trial)

                # if multiple csv files, look for "2Dpoints" in the name
                if len(pointsfile) > 1:
                    t = []
                    for q in pointsfile:
                        if any(x in q for x in pts):
                            t = t + [q]
                    # if there are multiple 2D points files, look for "corrected" in the name
                    if len(t) > 1:
                        for r in pointsfile:
                            if any(x in r for x in corr):
                                pointsfile = r
                else:
                    pointsfile = pointsfile[0]
                if isinstance(pointsfile, str) != True:
                    raise ValueError(
                        "Please check the points files in trial "
                        + trial
                        + " iteration "
                        + str(iteration)
                        + " folder"
                    )
                df = pd.read_csv(
                    data_path + "/" + trial + "/" + "it" + str(iteration) + "/" + pointsfile,
                    sep=",",
                    header=None,
                )
                df = df.loc[1:,].reset_index(drop=True)
                xpos = df.iloc[frames, 0 + (camera - 1) * 2 :: 4]
                ypos = df.iloc[frames, 1 + (camera - 1) * 2 :: 4]
                temp_data = pd.concat([xpos, ypos], axis=1).sort_index(axis=1)
                temp_data.index = relnames
                if temp_data.shape[1] > data.shape[1]:
                    raise ValueError(
                        "There are %d extra points in the corrected points file"
                        % ((temp_data.shape[1] - data.shape[1]) / 2)
                    )
                if temp_data.shape[1] < data.shape[1]:
                    raise ValueError(
                        "There are %d missing points in the corrected points file"
                        % ((data.shape[1] - temp_data.shape[1]) / 2)
                    )
                temp_data.columns = data.columns
                data = pd.concat([data, temp_data])
        data.replace(" NaN", np.nan, inplace=True)
        data.replace(" NaN ", np.nan, inplace=True)
        data.replace("NaN ", np.nan, inplace=True)
        data = data.astype("float")
        data = data.round(2)
        data = data.apply(pd.to_numeric)
        data.to_hdf(labeleddata_path + "/" + h5file[0], key="df_with_missing", mode="w")
        data.to_csv(labeleddata_path + "/" + csvfile[0], na_rep="NaN")

    print("Frames from %d trials successfully added to training dataset" % len(trialnames))
