from __future__ import annotations

import os
import random

import cv2
import numpy as np
import pandas as pd

from .common import default_camera_substrings


def xma_to_dlc(
    path_config_file,
    data_path,
    dataset_name,
    scorer,
    nframes,
    nnetworks=1,
    path_config_file_cam2=[],
):
    config = path_config_file[:-12]
    cameras = [1, 2]
    picked_frames = []
    dfs = []
    idx = []
    pnames = []
    subs = default_camera_substrings()
    trialnames = [
        folder
        for folder in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, folder)) and not folder.startswith(".")
    ]

    ### PART 1: Pick frames for dataset

    for trial in trialnames:
        # Read 2D points file
        contents = os.listdir(data_path + "/" + trial)
        filename = [x for x in contents if ".csv" in x]  # csv filename
        df1 = pd.read_csv(data_path + "/" + trial + "/" + filename[0], sep=",", header=None)

        # read pointnames from header row
        pointnames = df1.loc[0, ::4].astype(str).str[:-7].tolist()
        pnames.append(pointnames)

        df1 = df1.loc[1:,].reset_index(drop=True)  # remove header row

        # temp_idx = rows where fewer than half of columns are NaN
        ncol = df1.shape[1]
        temp_idx = list(df1.index.values[(~pd.isnull(df1)).sum(axis=1) >= ncol / 2])

        # randomize frames w/i each trial and append to index list
        random.shuffle(temp_idx)
        idx.append(temp_idx)
        dfs.append(df1)

    # a couple errors
    if sum(len(x) for x in idx) < nframes:
        raise ValueError("nframes is bigger than number of detected frames")

    # if pointnames aren't the same across trials
    if any(pnames[0] != x for x in pnames):
        raise ValueError("Make sure point names are consistent across trials")

    # pick frames to extract (NOTE this is random currently)
    # current code iteratively picks one frame at a time from each shuffled trial until # of picked_frames hits nframes
    # There is a much neater way to do this
    count = 0
    while sum(len(x) for x in picked_frames) < nframes:
        for trialnum in range(len(idx)):
            if sum(len(x) for x in picked_frames) < nframes:
                if count == 0:
                    picked_frames.insert(trialnum, [idx[trialnum][count]])
                elif count < len(idx[trialnum]):
                    picked_frames[trialnum] = picked_frames[trialnum] + [idx[trialnum][count]]
        count += 1

    ### Part 2: Extract images and 2D point data

    if nnetworks == 2:
        configs = [path_config_file[:-12], path_config_file_cam2[:-12]]

        for camera in cameras:
            print("Extracting camera %d trial images and 2D points..." % camera)
            relnames = []
            data = pd.DataFrame()
            # new training dataset folder
            newpath = configs[camera - 1] + "/labeled-data/" + dataset_name + "_cam" + str(camera)
            h5_save_path = newpath + "/CollectedData_" + scorer + ".h5"
            csv_save_path = newpath + "/CollectedData_" + scorer + ".csv"

            if os.path.exists(newpath):
                contents = os.listdir(newpath)
                if contents:
                    raise ValueError(
                        "There are already data in the camera %d training dataset folder" % camera
                    )
            else:
                os.makedirs(newpath)  # make new folder

            for trialnum, trial in enumerate(trialnames):
                # get video file
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
                    relpath = "labeled-data/" + dataset_name + "_cam" + str(camera) + "/"
                    frames = picked_frames[trialnum]
                    frames.sort()

                    for count, img in enumerate(imgs):
                        if count in frames:
                            image = cv2.imread(imgpath + "/" + img)
                            relname = relpath + trial + "_%s.png" % str(count + 1).zfill(4)
                            relnames = relnames + [relname]
                            cv2.imwrite(
                                newpath + "/" + trial + "_%s.png" % str(count + 1).zfill(4), image
                            )  # save frame

                else:
                    # file is actually a file
                    # extract frames from video and convert to png
                    video = data_path + "/" + trial + "/" + file
                    relpath = "labeled-data/" + dataset_name + "_cam" + str(camera) + "/"
                    frames = picked_frames[trialnum]
                    frames.sort()
                    cap = cv2.VideoCapture(video)
                    success, image = cap.read()
                    count = 0
                    while success:
                        if count in frames:
                            relname = relpath + trial + "_%s.png" % str(count + 1).zfill(4)
                            relnames = relnames + [relname]
                            cv2.imwrite(
                                newpath + "/" + trial + "_%s.png" % str(count + 1).zfill(4), image
                            )  # save frame
                        success, image = cap.read()
                        count += 1
                    cap.release()

                # extract 2D points data
                df1 = dfs[trialnum]
                xpos = df1.iloc[frames, 0 + (camera - 1) * 2 :: 4]
                ypos = df1.iloc[frames, 1 + (camera - 1) * 2 :: 4]
                temp_data = pd.concat([xpos, ypos], axis=1).sort_index(axis=1)
                data = pd.concat([data, temp_data])

            ### Part 3: Complete final structure of datafiles
            dataFrame = pd.DataFrame()
            temp = np.empty((data.shape[0], 2))
            temp[:] = np.nan
            for i, bodypart in enumerate(pointnames):
                index = pd.MultiIndex.from_product(
                    [[scorer], [bodypart], ["x", "y"]],
                    names=["scorer", "bodyparts", "coords"],
                )
                frame = pd.DataFrame(temp, columns=index, index=relnames)
                frame.iloc[:, 0:2] = data.iloc[:, 2 * i : 2 * i + 2].values.astype(float)
                dataFrame = pd.concat([dataFrame, frame], axis=1)
            dataFrame.replace("", np.nan, inplace=True)
            dataFrame.replace(" NaN", np.nan, inplace=True)
            dataFrame.replace(" NaN ", np.nan, inplace=True)
            dataFrame.replace("NaN ", np.nan, inplace=True)
            dataFrame.apply(pd.to_numeric)
            dataFrame.to_hdf(h5_save_path, key="df_with_missing", mode="w")
            dataFrame.to_csv(csv_save_path, na_rep="NaN")
            print("...done.")

    else:
        relnames = []
        data = pd.DataFrame()
        # new training dataset folder
        newpath = config + "/labeled-data/" + dataset_name
        h5_save_path = newpath + "/CollectedData_" + scorer + ".h5"
        csv_save_path = newpath + "/CollectedData_" + scorer + ".csv"

        if os.path.exists(newpath):
            contents = os.listdir(newpath)
            if contents:
                raise ValueError(
                    "There are already data in the camera %d training dataset folder" % camera
                )
        else:
            os.makedirs(newpath)  # make new folder

        for camera in cameras:
            print("Extracting camera %d trial images and 2D points..." % camera)

            for trialnum, trial in enumerate(trialnames):
                # get video file
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
                        if count in frames:
                            image = cv2.imread(imgpath + "/" + img)
                            relname = (
                                relpath + trial + "_cam" + str(camera) + "_%s.png" % str(count + 1).zfill(4)
                            )
                            relnames = relnames + [relname]
                            cv2.imwrite(
                                newpath + "/" + trial + "_cam" + str(camera) + "_%s.png" % str(count + 1).zfill(4),
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
                        if count in frames:
                            relname = (
                                relpath + trial + "_cam" + str(camera) + "_%s.png" % str(count + 1).zfill(4)
                            )
                            relnames = relnames + [relname]
                            cv2.imwrite(
                                newpath + "/" + trial + "_cam" + str(camera) + "_%s.png" % str(count + 1).zfill(4),
                                image,
                            )  # save frame
                        success, image = cap.read()
                        count += 1
                    cap.release()

                # extract 2D points data
                df1 = dfs[trialnum]
                xpos = df1.iloc[frames, 0 + (camera - 1) * 2 :: 4]
                ypos = df1.iloc[frames, 1 + (camera - 1) * 2 :: 4]
                temp_data = pd.concat([xpos, ypos], axis=1).sort_index(axis=1)
                temp_data.columns = range(temp_data.shape[1])
                data = pd.concat([data, temp_data])

        ### Part 3: Complete final structure of datafiles
        dataFrame = pd.DataFrame()
        temp = np.empty((data.shape[0], 2))
        temp[:] = np.nan
        for i, bodypart in enumerate(pointnames):
            index = pd.MultiIndex.from_product(
                [[scorer], [bodypart], ["x", "y"]],
                names=["scorer", "bodyparts", "coords"],
            )
            frame = pd.DataFrame(temp, columns=index, index=relnames)
            frame.iloc[:, 0:2] = data.iloc[:, 2 * i : 2 * i + 2].values.astype(float)
            dataFrame = pd.concat([dataFrame, frame], axis=1)
        dataFrame.replace("", np.nan, inplace=True)
        dataFrame.replace(" NaN", np.nan, inplace=True)
        dataFrame.replace(" NaN ", np.nan, inplace=True)
        dataFrame.replace("NaN ", np.nan, inplace=True)
        dataFrame.apply(pd.to_numeric)
        dataFrame.to_hdf(h5_save_path, key="df_with_missing", mode="w")
        dataFrame.to_csv(csv_save_path, na_rep="NaN")
        print("...done.")

    print("Training data extracted to projectpath/labeled-data. Now use deeplabcut.create_training_dataset")
