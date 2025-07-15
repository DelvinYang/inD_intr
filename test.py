from typing import List

import numpy as np
import torch
from loguru import logger
from matplotlib import pyplot as plt
from tqdm import tqdm

from common import State
from plot_and_test import draw_scene
from read_data_inD import DataReaderInD
from scenarioind import ScenarioInD, Vehicle


prefix_number, data_path = '00', '/Users/delvin/Desktop/programs/跨文化返修/inD'

data_reader = DataReaderInD(prefix_number, data_path)
scenario = ScenarioInD(data_reader)

ego_id = 6
frame_num = 150

ego_veh: Vehicle = scenario.find_vehicle_by_id(ego_id)

start_frame = ego_veh.initial_frame
end_frame = ego_veh.final_frame

ego_state: State = scenario.find_vehicle_state(frame_num, ego_id)
svs_state: List[State] = scenario.find_svs_state(frame_num, ego_id)

ev_pos = ego_state.position  # np.array([x, y])
ev_heading_rad = np.deg2rad(ego_state.heading)  # 注意不是 heading_vis

# 单位朝向向量
ev_heading_vec = np.array([np.cos(ev_heading_rad), np.sin(ev_heading_rad)])

for sv in svs_state:
    sv_pos = sv.position
    vec_ev_to_sv = sv_pos - ev_pos

    # 单位化方向向量
    vec_ev_to_sv_unit = vec_ev_to_sv / (np.linalg.norm(vec_ev_to_sv) + 1e-6)

    # 计算夹角（弧度）
    dot = np.clip(np.dot(ev_heading_vec, vec_ev_to_sv_unit), -1.0, 1.0)
    angle = np.arccos(dot)  # 范围：[0, pi]

    # 判断方向（用叉积判断左右）
    cross = np.cross(ev_heading_vec, vec_ev_to_sv_unit)

    angle_deg = np.rad2deg(angle)

logger.debug(f"ego_state: {ego_state}")