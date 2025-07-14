import os

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation

from common import *
from read_data_inD import DataReaderInD
from scenarioind import ScenarioInD


def draw_scene(scenario: ScenarioInD, frame_id: int):
    """Visualize all objects of ``frame_id`` on the orthophoto.

        Parameters
        ----------
        scenario : ScenarioInD
            Scenario instance containing loaded tracks.
        frame_id : int
            Frame to plot.
        """

    data_reader = scenario.data_reader

    if not os.path.exists(data_reader.background_path):
        raise FileNotFoundError(f"Background image not found: {data_reader.background_path}")

    # Load and convert the background image to RGB for matplotlib
    img = cv2.imread(data_reader.background_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read background image {data_reader.background_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)

    px_to_meter = data_reader.recordingMeta[ORTHO_PX_TO_METER]
    area_id = str(data_reader.recordingMeta.get('locationId', '1'))
    area_cfg = RELEVANT_AREAS.get(area_id)

    def scale(val: float) -> float:
        """Convert value from dataset units to pixels on the downscaled image."""
        return val / px_to_meter / SCALE_DOWN_FACTOR

    for veh_id in scenario.vehicles:
        state = scenario.find_vehicle_state(frame_id, veh_id)
        if state is None:
            continue

        x = scale(state.x[0])
        y = scale(state.y[0])
        w = scale(state.width)
        h = scale(state.height)

        # Convert heading from degrees to radians and adapt to the image
        heading_vis = -state.heading
        if heading_vis < 0:
            heading_vis += 360
        heading_rad = np.deg2rad(heading_vis)

        rotated_bbox_vertices = get_rotated_bbox(x, y, w, h, heading_rad)[0]

        polygon = patches.Polygon(
            rotated_bbox_vertices,
            closed=True,
            linewidth=1,
            edgecolor='r',
            facecolor='none'
        )
        ax.add_patch(polygon)

    if area_cfg:
        x_lim = [val / SCALE_DOWN_FACTOR for val in area_cfg['x_lim']]
        y_lim = [val / SCALE_DOWN_FACTOR for val in area_cfg['y_lim']]
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

    ax.set_title(f"Frame {frame_id}")
    plt.tight_layout()
    plt.show()


def play_scene(scenario: ScenarioInD, interval: int = 200):
    """Continuously play all frames of the scenario on the orthophoto."""
    data_reader = scenario.data_reader

    if not os.path.exists(data_reader.background_path):
        raise FileNotFoundError(f"Background image not found: {data_reader.background_path}")

    img = cv2.imread(data_reader.background_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read background image {data_reader.background_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    px_to_meter = data_reader.recordingMeta[ORTHO_PX_TO_METER]
    area_id = str(data_reader.recordingMeta.get('locationId', '1'))
    area_cfg = RELEVANT_AREAS.get(area_id)

    def scale(val: float) -> float:
        return val / px_to_meter / SCALE_DOWN_FACTOR

    min_frame = min(v.initial_frame for v in scenario.vehicles.values())
    max_frame = max(v.final_frame for v in scenario.vehicles.values())

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)

    def update(frame_id):
        ax.clear()
        ax.imshow(img)

        for veh_id in scenario.vehicles:
            state = scenario.find_vehicle_state(frame_id, veh_id)
            if state is None:
                continue

            x = scale(state.x[0])
            y = scale(state.y[0])
            w = scale(state.width)
            h = scale(state.height)

            heading_vis = -state.heading
            if heading_vis < 0:
                heading_vis += 360
            heading_rad = np.deg2rad(heading_vis)

            vertices = get_rotated_bbox(x, y, w, h, heading_rad)[0]
            polygon = patches.Polygon(vertices, closed=True, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(polygon)

        if area_cfg:
            x_lim = [val / SCALE_DOWN_FACTOR for val in area_cfg['x_lim']]
            y_lim = [val / SCALE_DOWN_FACTOR for val in area_cfg['y_lim']]
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)

        ax.set_title(f"Frame {frame_id}")

    animation.FuncAnimation(fig, update, frames=range(min_frame, max_frame + 1), interval=interval, repeat=True)
    plt.tight_layout()
    plt.show()


# ========== 使用示例 ==========
if __name__ == "__main__":
    prefix_number = "00"
    data_path = "/Users/delvin/Desktop/programs/跨文化返修/inD"

    data_reader = DataReaderInD(prefix_number, data_path)
    scenario = ScenarioInD(data_reader)
    play_scene(scenario)