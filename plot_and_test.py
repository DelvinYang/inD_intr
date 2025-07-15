import os
import time

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from common import *
from read_data_inD import DataReaderInD
from scenarioind import ScenarioInD


def draw_scene(scenario: ScenarioInD, frame_id: int, fig=None, ax=None):
    """Visualize all objects of ``frame_id`` on the orthophoto.

        Parameters
        ----------
        scenario : ScenarioInD
            Scenario instance containing loaded tracks.
        frame_id : int
            Frame to plot.
        """

    data_reader = scenario.data_reader

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    ax.clear()

    if not os.path.exists(data_reader.background_path):
        raise FileNotFoundError(f"Background image not found: {data_reader.background_path}")

    # Load and convert the background image to RGB for matplotlib
    img = cv2.imread(data_reader.background_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(
            f"Failed to read background image {data_reader.background_path}"
        )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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

        # Draw vehicle ID at center
        ax.text(x, y, str(veh_id), color='blue', fontsize=6, ha='center', va='center')

        # Draw heading arrow
        arrow_length = w  # adjust arrow length as needed
        dx = arrow_length * np.cos(heading_rad)
        dy = arrow_length * np.sin(heading_rad)
        ax.arrow(x, y, dx, dy, head_width=1.5, head_length=1.5, fc='green', ec='green', linewidth=1)

    if area_cfg:
        x_lim = [val / SCALE_DOWN_FACTOR for val in area_cfg['x_lim']]
        y_lim = [val / SCALE_DOWN_FACTOR for val in area_cfg['y_lim']]
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

    ax.set_title(f"Frame {frame_id}")
    plt.tight_layout()
    fig.canvas.draw()


def play_scene(scenario: ScenarioInD, interval: int = 2):
    """Play all frames by repeatedly clearing and drawing the scene."""

    min_frame = min(v.initial_frame for v in scenario.vehicles.values())
    max_frame = max(v.final_frame for v in scenario.vehicles.values())

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.ion()

    for frame_id in range(min_frame, max_frame + 1):
        draw_scene(scenario, frame_id, fig, ax)
        plt.pause(interval / 2000.0)

    plt.ioff()
    plt.show()


# ========== 使用示例 ==========
if __name__ == "__main__":
    prefix_number = "00"
    data_path = "/Users/delvin/Desktop/programs/跨文化返修/inD"

    data_reader = DataReaderInD(prefix_number, data_path)
    scenario = ScenarioInD(data_reader)
    draw_scene(scenario, 150)
    plt.show()
