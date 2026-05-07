"""
XROMM Tools for DeepLabCut
Developed by J.D. Laurence-Chasen

Legacy compatibility layer:
This module preserves the original function signatures while delegating
implementation to the modular `xrommtools` package.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from xrommtools import add_frames as _add_frames
from xrommtools import analyze_xromm_videos as _analyze_xromm_videos
from xrommtools import dlc_to_xma as _dlc_to_xma
from xrommtools import xma_to_dlc as _xma_to_dlc


def xma_to_dlc(
    path_config_file,
    data_path,
    dataset_name,
    scorer,
    nframes,
    nnetworks=1,
    path_config_file_cam2=[],
):
    return _xma_to_dlc(
        path_config_file,
        data_path,
        dataset_name,
        scorer,
        nframes,
        nnetworks=nnetworks,
        path_config_file_cam2=path_config_file_cam2,
    )


def dlc_to_xma(cam1data, cam2data, trialname, savepath):
    return _dlc_to_xma(cam1data, cam2data, trialname, savepath)


def analyze_xromm_videos(
    path_config_file,
    path_data_to_analyze,
    iteration,
    nnetworks=1,
    path_config_file_cam2=[],
):
    return _analyze_xromm_videos(
        path_config_file,
        path_data_to_analyze,
        iteration,
        nnetworks=nnetworks,
        path_config_file_cam2=path_config_file_cam2,
    )


def add_frames(
    path_config_file,
    data_path,
    iteration,
    frames,
    nnetworks=1,
    path_config_file_cam2="enterpathofcam2config",
):
    return _add_frames(
        path_config_file,
        data_path,
        iteration,
        frames,
        nnetworks=nnetworks,
        path_config_file_cam2=path_config_file_cam2,
    )
