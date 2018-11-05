from __future__ import print_function

import numpy as np

from sklearn.utils.linear_assignment_ import linear_assignment
from utils.kalman_filter import KalmanFilter

SOFT_IOU_ADD = 0.25


def iou(a, b, inc_size, soft_iou_k):
    """
    Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    """
    w_a = a[2] - a[0]
    w_b = b[2] - b[0]
    h_a = a[3] - a[1]
    h_b = b[3] - b[1]
    ratio = (w_a * h_a) / (w_b * h_b)
    if 1 / inc_size <= ratio <= inc_size:
        x1 = np.maximum(a[0], b[0])
        y1 = np.maximum(a[1], b[1])
        x2 = np.minimum(a[2], b[2])
        y2 = np.minimum(a[3], b[3])
        w_c = np.maximum(0., x2 - x1)
        h_c = np.maximum(0., y2 - y1)
        s_c = w_c * h_c
        if s_c > 0:
            return SOFT_IOU_ADD + (1 - SOFT_IOU_ADD) * s_c / ((a[2] - a[0]) * (a[3] - a[1])
                                                              + (b[2] - b[0]) * (b[3] - b[1]) - s_c)
        elif soft_iou_k > 0:
            dx = np.abs((a[0] + a[2]) - (b[0] + b[2])) / 2
            dy = np.abs((a[1] + a[3]) - (b[1] + b[3])) / 2
            soft_iou = SOFT_IOU_ADD * (1 - max(dx / (w_a + w_b), dy / (h_a + h_b)) / soft_iou_k)
            if soft_iou > 0:
                return soft_iou
    return 0


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    return np.array([x, y, w, h]).reshape((4, 1))


def convert_x_to_bbox(x):
    """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
    w = x[2]
    h = x[3]
    return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((4, 1))


class Box:
    def __init__(self, track_id, left, bottom, right, top, frame, phantom, confidence, person):
        self.track_id = track_id
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self.frame = frame
        self.phantom = phantom
        self.confidence = confidence
        self.person = person


class Track:
    def __init__(self, track_id, box, vx, vy, ax, ay):
        self.track_id = track_id
        self.boxes = [box]
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, vx, 0, 0, 0],
             [0, 1, 0, 0, 0, vy, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, ax, 0],
             [0, 0, 0, 0, 0, 1, 0, ay],
             [0, 0, 0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = convert_bbox_to_z(np.array([box.left, box.bottom, box.right, box.top]))
        self.kf.predict()
        self.last_frame = box.frame
        self.real_boxes = 1
        self.counted = False
        self.phantom_boxes = 0
        self.max_phantom_boxes = 0

    def update_phantom(self):
        predicted = self.kf.x
        b = convert_x_to_bbox(predicted)
        new_box = Box(self.track_id, b[0], b[1], b[2], b[3], self.last_frame, True, 0, "")
        self.last_frame += 1
        self.boxes.append(new_box)
        self.kf.update(predicted[:4])
        self.kf.predict()
        self.phantom_boxes += 1
        if self.phantom_boxes > self.max_phantom_boxes:
            self.max_phantom_boxes = self.phantom_boxes

    def update_real(self, box, non_decrease, real_detection):
        if len(self.boxes) > 0:
            prev = self.boxes[-1]
            ratio = (box.right - box.left) * (box.top - box.bottom) / (
                    (prev.right - prev.left) * (prev.top - prev.bottom))
            if ratio < non_decrease:
                predicted = convert_x_to_bbox(self.kf.x[:4])
                box = Box(self.track_id, predicted[0], predicted[1], predicted[2], predicted[3], self.last_frame, True, 0, "")

        self.boxes.append(box)
        if box.confidence >= real_detection:
            self.real_boxes += 1
        self.phantom_boxes = 0
        self.last_frame = box.frame
        self.kf.update(convert_bbox_to_z(np.array([box.left, box.bottom, box.right, box.top])))
        self.kf.predict()

    def get_prediction(self):
        return convert_x_to_bbox(self.kf.x[:4])

    def get_max_phantoms(self):
        return self.max_phantom_boxes


class PhantomSortTracker:
    def __init__(self, min_x, min_y, max_x, max_y, detection_thresh, nms_thresh,
                 detections_count, track_creation_score, min_phantom_omit, max_phantom_omit,
                 phantom_coef, non_decrease, inc_size, vx, vy, ax, ay, soft_iou_k, real_detection):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.detection_thresh = detection_thresh  # Threshold for detection probability
        self.nms_thresh = nms_thresh  # X
        self.detections_count = detections_count  # How much to be a person V
        self.track_creation_score = track_creation_score  # Follow person if detection_thresh < P < track_creation_score
        self.min_phantom_omit = min_phantom_omit  # About phantoms
        self.max_phantom_omit = max_phantom_omit  # About phantoms
        self.phantom_coef = phantom_coef  # About phantoms
        self.non_decrease = non_decrease  # Times
        self.inc_size = inc_size  # Times for far and close
        self.soft_iou_k = soft_iou_k  # X
        self.vx = vx  # Kalman filters velocity V
        self.vy = vy  # Kalman filters velocity V
        self.ax = ax  # Kalman filters acceleration V
        self.ay = ay  # Kalman filters acceleration V
        self.real_detection = real_detection
        self.tracks = []
        self.deleted_tracks = []
        self.next_id = 0
        self.objects = 0

    @classmethod
    def from_json(cls, config):
        return PhantomSortTracker(min_x=config["min_x"], min_y=config["min_y"], max_x=config["max_x"],
                                  max_y=config["max_y"],
                                  detection_thresh=config["detection-thresh"], nms_thresh=config["nms-thresh"],
                                  detections_count=config["detections-count"],
                                  track_creation_score=config["track-creation"],
                                  min_phantom_omit=config["min-phantom"], max_phantom_omit=config["max-phantom"],
                                  phantom_coef=config["phantom-coef"], non_decrease=config["non-decrease"],
                                  vx=config["vx"], vy=config["vy"], ax=config["ax"], ay=config["ay"],
                                  inc_size=config["inc-size"], soft_iou_k=config["soft-iou-k"])

    def update_all(self, boxes, start_frame):
        colored_boxes = []
        frame = start_frame
        for detections in boxes:
            if len(detections) > 0:
                iou_matrix = np.zeros((len(detections), len(self.tracks) + len(detections)), dtype=np.float32)
                for d, det in enumerate(detections):
                    for t, track in enumerate(self.tracks):
                        iou_matrix[d, t] = -iou(det, track.get_prediction(), self.inc_size, self.soft_iou_k)
                        if iou_matrix[d, t] < 0:
                            if not track.boxes[-1].phantom:
                                iou_matrix[d, t] -= 1.0
                    if det[4] < self.track_creation_score:
                        iou_matrix[d, len(self.tracks) + d] = +0.001
                    else:
                        iou_matrix[d, len(self.tracks) + d] = -0.001
                matched_indices = linear_assignment(iou_matrix)
                old_length = len(self.tracks)
                for row in matched_indices:
                    b = detections[row[0]]
                    if row[1] >= old_length:
                        id = self.next_id
                        self.next_id += 1
                        self.tracks.append(Track(id, Box(id, b[0], b[1], b[2], b[3], frame, False, b[4], b[5]),
                                                 self.vx, self.vy, self.ax, self.ay))
                    elif iou_matrix[row[0], row[1]] < 0:
                        track = self.tracks[row[1]]
                        box = Box(track.track_id, b[0], b[1], b[2], b[3], frame, False, b[4], b[5])
                        track.update_real(box, self.non_decrease, self.real_detection)
                        if not track.counted and track.real_boxes >= self.detections_count:
                            self.objects += 1
                            track.counted = True

            boxes = []
            active_tracks = []
            for track in self.tracks:
                if track.last_frame < frame:
                    track.update_phantom()
                    phantom_threshold = np.minimum(self.max_phantom_omit,
                                                   np.maximum(self.min_phantom_omit,
                                                              self.phantom_coef * track.get_max_phantoms()))
                    box = track.boxes[-1]
                    if track.phantom_boxes > phantom_threshold \
                            or box.left > self.max_x \
                            or box.right < self.min_x \
                            or box.top < self.min_y \
                            or box.bottom > self.max_y:
                        self.deleted_tracks.append(track)
                    else:
                        active_tracks.append(track)
                else:
                    active_tracks.append(track)
                box = track.boxes[-1]
                colored_box = [int(box.left), int(box.bottom), int(box.right), int(box.top), int(track.track_id),
                               int(box.phantom)]
                boxes.append(colored_box)
            colored_boxes.append(boxes)
            self.tracks = active_tracks
            frame += 1

        return colored_boxes
