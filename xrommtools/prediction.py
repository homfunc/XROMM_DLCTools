from __future__ import annotations

import os

import pandas as pd

from .common import default_camera_substrings
from .conversion import dlc_to_xma


def analyze_xromm_videos(
    path_config_file,
    path_data_to_analyze,
    iteration,
    nnetworks=1,
    path_config_file_cam2=[],
):
    from deeplabcut.pose_estimation_tensorflow.predict_videos import analyze_videos

    # assumes you have cam1 and cam2 videos as .avi in their own seperate trial folders
    # assumes all folders w/i new_data_path are trial folders
    # convert jpg stacks?

    # analyze videos
    cameras = [1, 2]
    config = path_config_file
    configs = [path_config_file, path_config_file_cam2]
    subs = default_camera_substrings()
    trialnames = os.listdir(path_data_to_analyze)

    for trialnum, trial in enumerate(trialnames):
        trialpath = path_data_to_analyze + "/" + trial
        contents = os.listdir(trialpath)
        savepath = trialpath + "/" + "it%d" % iteration
        if os.path.exists(savepath):
            temp = os.listdir(savepath)
            if temp:
                raise ValueError(
                    "There are already predicted points in iteration %d subfolders" % iteration
                )
        else:
            os.makedirs(savepath)  # make new folder
        # get video file
        for camera in cameras:
            file = []
            for name in contents:
                if any(x in name for x in subs[camera - 1]):
                    file = name
            if not file:
                raise ValueError("Cannot locate %s video file or image folder" % trial)

            video = trialpath + "/" + file
            # analyze video
            if nnetworks == 1:
                analyze_videos(config, [video], destfolder=savepath, save_as_csv=True)
            else:
                analyze_videos(configs[camera - 1], [video], destfolder=savepath, save_as_csv=True)

        # get filenames and read analyzed data
        contents = os.listdir(savepath)
        datafiles = [s for s in contents if ".h5" in s]
        if not datafiles:
            raise ValueError("Cannot find predicted points. Some wrong with DeepLabCut?")
        cam1data = pd.read_hdf(savepath + "/" + datafiles[0])
        cam2data = pd.read_hdf(savepath + "/" + datafiles[1])
        dlc_to_xma(cam1data, cam2data, trial, savepath)
