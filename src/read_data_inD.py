"""Utilities for reading inD dataset CSV files."""

import os

import numpy as np
from loguru import logger
from .common import *
import pandas as pd


class DataReaderInD(object):
    def __init__(self, prefix_number, data_path):
        self.prefix_number = prefix_number
        self.data_path = data_path
        self.csv_tracks_path, self.csv_tracksMeta_path, self.csv_recordingMeta_path, self.background_path = self.generate_path()

        self.tracks, self.id_list = self.read_tracks_csv()
        self.tracksMeta = self.read_static_info()
        self.recordingMeta = self.read_recording_info()
        logger.info(f"read data done, total tracks: {len(self.tracks)}")

    def generate_path(self):
        # 构建子目录路径
        data_dir = os.path.join(self.data_path, "data")

        # 构建两个目标文件的路径
        tracks_path = os.path.join(data_dir, f"{self.prefix_number}_tracks.csv")
        tracks_meta_path = os.path.join(data_dir, f"{self.prefix_number}_tracksMeta.csv")
        recording_meta_path = os.path.join(data_dir, f"{self.prefix_number}_recordingMeta.csv")
        background_path = os.path.join(data_dir, f"{self.prefix_number}_background.png")

        return str(tracks_path), str(tracks_meta_path), str(recording_meta_path), str(background_path)

    def read_tracks_csv(self):
        # Read the csv file, convert it into a useful data structure
        df = pd.read_csv(self.csv_tracks_path)
        grouped = df.groupby([TRACK_ID], sort=False)

        tracks = []
        id_list = set()
        for group_id, rows in grouped:

            # 提取所有速度和加速度相关字段的值
            v_a_values = np.concatenate([
                rows[X_VELOCITY].values,
                rows[Y_VELOCITY].values,
                rows[X_ACCELERATION].values,
                rows[Y_ACCELERATION].values,
                rows[LON_VELOCITY].values,
                rows[LAT_VELOCITY].values,
                rows[LON_ACCELERATION].values,
                rows[LAT_ACCELERATION].values
            ])

            # 若全部为 0，跳过该轨迹
            if np.all(v_a_values == 0):
                logger.info(f"All Zeros, skipping track {group_id}")
                continue
            bounding_boxes = np.transpose(np.array([rows[X].values,
                                                    -rows[Y].values,
                                                    rows[LENGTH].values,
                                                    rows[WIDTH].values]))
            track_data = {TRACK_ID: np.int64(group_id),
                          # for compatibility, int would be more space efficient
                          FRAME: rows[FRAME].values,
                          BBOX: bounding_boxes,
                          HEADING: rows[HEADING].values,
                          X_VELOCITY: rows[X_VELOCITY].values,
                          Y_VELOCITY: -rows[Y_VELOCITY].values,
                          X_ACCELERATION: rows[X_ACCELERATION].values,
                          Y_ACCELERATION: -rows[Y_ACCELERATION].values,
                          LON_VELOCITY: rows[LON_VELOCITY].values,
                          LAT_VELOCITY: rows[LAT_VELOCITY].values,
                          LON_ACCELERATION: rows[LON_ACCELERATION].values,
                          LAT_ACCELERATION: rows[LAT_ACCELERATION].values,
                          }
            tracks.append(track_data)
            id_ = np.int64(group_id)[0]
            id_list.add(id_)
        return tracks, id_list

    def read_static_info(self):
        # Read the csv file, convert it into a useful data structure
        df = pd.read_csv(self.csv_tracksMeta_path)

        # Declare and initialize the static_dictionary
        static_dictionary = {}

        # Iterate over all rows of the csv because we need to create the bounding boxes for each row
        for i_row in range(df.shape[0]):
            track_id = int(df[TRACK_ID][i_row])
            static_dictionary[track_id] = {TRACK_ID: track_id,
                                           INITIAL_FRAME: int(df[INITIAL_FRAME][i_row]),
                                           FINAL_FRAME: int(df[FINAL_FRAME][i_row]),
                                           NUM_FRAMES: int(df[NUM_FRAMES][i_row]),
                                           CLASS: str(df[CLASS][i_row])
                                           }
        return static_dictionary

    def read_recording_info(self):
        df = pd.read_csv(self.csv_recordingMeta_path)

        # Declare and initialize the extracted_meta_dictionary
        extracted_meta_dictionary = {
                                     FRAME_RATE: int(df[FRAME_RATE][0]),
                                     SPEED_LIMIT: float(df[SPEED_LIMIT][0]),
                                     ORTHO_PX_TO_METER: float(df[ORTHO_PX_TO_METER][0]),
                                     AREA_ID: int(df[AREA_ID][0]),
                                     }
        return extracted_meta_dictionary

