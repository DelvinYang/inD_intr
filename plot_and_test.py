import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scenarioind import ScenarioInD
from read_data_inD import DataReaderInD
from common import ORTHO_PX_TO_METER, State


def draw_scene(scenario: ScenarioInD, frame_id: int):
    pass


# ========== 使用示例 ==========
if __name__ == "__main__":
    prefix_number = "00"
    data_path = "/Users/delvin/Desktop/programs/跨文化返修/inD"

    data_reader = DataReaderInD(prefix_number, data_path)
    scenario = ScenarioInD(data_reader)
    draw_scene(scenario, frame_id=150)