from typing import List

import numpy as np
import torch
from loguru import logger
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.common import State
from scripts.plot_and_test import draw_scene
from src.read_data_inD import DataReaderInD
from src.scenarioind import ScenarioInD, Vehicle
from frenet_system_creator import FrenetSystem


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
ev_frenet_system = FrenetSystem(scenario.set_reference_path(ego_id))
ds_pos_ev, _ = ev_frenet_system.cartesian2ds_frame(ev_pos)

# ev 在 ref_path 上面, 前方 refpath 方向, 右正左负
for sv in svs_state:
    sv_pos = sv.position
    ds_pos_sv, _ = ev_frenet_system.cartesian2ds_frame(sv_pos)


logger.debug(f"ego state: {ego_state}")