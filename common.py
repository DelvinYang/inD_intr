# Tracks
import numpy as np

BBOX = "bbox"
FRAME = "frame"
TRACK_ID = "trackId"
X = "xCenter"
Y = "yCenter"
HEADING = "heading"
LENGTH = "length"
WIDTH = "width"
X_VELOCITY = "xVelocity"
Y_VELOCITY = "yVelocity"
X_ACCELERATION = "xAcceleration"
Y_ACCELERATION = "yAcceleration"
LON_VELOCITY = "lonVelocity"
LAT_VELOCITY = "latVelocity"
LON_ACCELERATION = "lonAcceleration"
LAT_ACCELERATION = "latAcceleration"

# STATIC FILE
INITIAL_FRAME = "initialFrame"
FINAL_FRAME = "finalFrame"
NUM_FRAMES = "numFrames"
CLASS = "class"

# META FILE
FRAME_RATE = "frameRate"
SPEED_LIMIT = "speedLimit"
ORTHO_PX_TO_METER = "orthoPxToMeter"
AREA_ID = "locationId"


SCALE_DOWN_FACTOR = 12

RELEVANT_AREAS = {
    "1": {
        "x_lim": [2000, 11500],
        "y_lim": [9450, 0]
    },
    "2": {
        "x_lim": [0, 12500],
        "y_lim": [7400, 0]
    },
    "3": {
        "x_lim": [0, 11500],
        "y_lim": [9365, 0]
    },
    "4": {
        "x_lim": [2700, 15448],
        "y_lim": [9365, 0]
    }
}


class State(object):
    def __init__(self, this_id, x, y, lon, lat, width, height, heading, vehicle_type):
        self.id = this_id
        self.x = x
        self.y = y
        self.lon = lon
        self.lat = lat
        self.width = width
        self.height = height
        self.heading = heading
        self.vehicle_type = vehicle_type
        self.position = np.array([x[0], y[0]], dtype=float)
        self.heading_vis = (-self.heading) % 360



def get_rotated_bbox(x_center: np.ndarray, y_center: np.ndarray,
                     length: np.ndarray, width: np.ndarray, heading: np.ndarray) -> np.ndarray:
    """
    Calculate the corners of a rotated bbox from the position, shape and heading for every timestamp.

    :param x_center: x coordinates of the object center positions [num_timesteps]
    :param y_center: y coordinates of the object center positions [num_timesteps]
    :param length: objects lengths [num_timesteps]
    :param width: object widths [num_timesteps]
    :param heading: object heading (rad) [num_timesteps]
    :return: Numpy array in the shape [num_timesteps, 4 (corners), 2 (dimensions)]
    """
    centroids = np.column_stack([x_center, y_center])

    # Precalculate all components needed for the corner calculation
    l = length / 2
    w = width / 2
    c = np.cos(heading)
    s = np.sin(heading)

    lc = l * c
    ls = l * s
    wc = w * c
    ws = w * s

    # Calculate all four rotated bbox corner positions assuming the object is located at the origin.
    # To do so, rotate the corners at [+/- length/2, +/- width/2] as given by the orientation.
    # Use a vectorized approach using precalculated components for maximum efficiency
    rotated_bbox_vertices = np.empty((centroids.shape[0], 4, 2))

    # Front-right corner
    rotated_bbox_vertices[:, 0, 0] = lc - ws
    rotated_bbox_vertices[:, 0, 1] = ls + wc

    # Rear-right corner
    rotated_bbox_vertices[:, 1, 0] = -lc - ws
    rotated_bbox_vertices[:, 1, 1] = -ls + wc

    # Rear-left corner
    rotated_bbox_vertices[:, 2, 0] = -lc + ws
    rotated_bbox_vertices[:, 2, 1] = -ls - wc

    # Front-left corner
    rotated_bbox_vertices[:, 3, 0] = lc + ws
    rotated_bbox_vertices[:, 3, 1] = ls - wc

    # Move corners of rotated bounding box from the origin to the object's location
    rotated_bbox_vertices = rotated_bbox_vertices + np.expand_dims(centroids, axis=1)
    return rotated_bbox_vertices
