# Tracks
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
        self.center = self.calc_center()

    def calc_center(self):
        return self.x[0] + (self.width / 2), self.y[0] + (self.height / 2)