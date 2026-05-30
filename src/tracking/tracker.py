import numpy as np
from filterpy.kalman import KalmanFilter

class Track:
    def __init__(self, bbox, track_id):
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        
        # Initialize Kalman Filter for tracking [x_center, y_center, width, height, dx, dy]
        self.kf = KalmanFilter(dim_x=6, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]
        ])
        
        self.kf.x[:4, 0] = bbox
        self.history = []

    def camera_compensate(self, M):
        """Compensates the Kalman Filter state using the GMC Affine Matrix 'M'"""
        dx = M[0, 2]
        dy = M[1, 2]
        self.kf.x[0, 0] += dx
        self.kf.x[1, 0] += dy

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.kf.x[:4, 0]

    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.kf.update(bbox)


class DroneTracker:
    def __init__(self, max_age=30, min_hits=3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []
        self.frame_count = 0
        self.next_id = 1

    def update(self, detections, transformation_matrix):
        self.frame_count += 1

        # 1. Compensate existing tracks for camera movement
        for track in self.tracks:
            track.camera_compensate(transformation_matrix)

        # 2. Get predictions for current frame
        predicted_boxes = []
        for track in self.tracks:
            predicted_boxes.append(track.predict())

        # 3. Associate detections to tracked boxes
        matched, unmatched_dets, unmatched_tracks = self._associate_detections(detections, predicted_boxes)

        # 4. Update matched tracks
        for m in matched:
            self.tracks[m[1]].update(detections[m[0]][:4])

        # 5. Create new tracks for unmatched detections
        for i in unmatched_dets:
            track = Track(detections[i][:4], self.next_id)
            self.next_id += 1
            self.tracks.append(track)

        # 6. Clean out dead tracks
        ret = []
        for i, track in enumerate(self.tracks):
            if track.time_since_update < 1 and (track.hits >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((track.kf.x[:4, 0], [track.track_id])).astype(np.float32))
        
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return ret

    def _associate_detections(self, detections, predicted_boxes, iou_threshold=0.3):
        if len(predicted_boxes) == 0:
            return np.empty((0, 2), dtype=int), list(range(len(detections))), []

        iou_matrix = np.zeros((len(detections), len(predicted_boxes)), dtype=np.float32)
        for d, det in enumerate(detections):
            for t, trk in enumerate(predicted_boxes):
                iou_matrix[d, t] = self._box_iou(det[:4], trk)

        local_matches = []
        if iou_matrix.size > 0:
            max_indices = np.argmax(iou_matrix, axis=1)
            for det_idx, trk_idx in enumerate(max_indices):
                if iou_matrix[det_idx, trk_idx] >= iou_threshold:
                    local_matches.append([det_idx, trk_idx])

        np_matches = np.array(local_matches, dtype=int)
        if len(np_matches) == 0:
            np_matches = np.empty((0, 2), dtype=int)

        unmatched_detections = [d for d in range(len(detections)) if len(np_matches) == 0 or d not in np_matches[:, 0]]
        unmatched_tracks = [t for t in range(len(predicted_boxes)) if len(np_matches) == 0 or t not in np_matches[:, 1]]

        return np_matches, unmatched_detections, unmatched_tracks

    def _box_iou(self, box1, box2):
        xx1 = max(box1[0], box2[0])
        yy1 = max(box1[1], box2[1])
        xx2 = min(box1[2], box2[2])
        yy2 = min(box1[3], box2[3])
        w = max(0, xx2 - xx1)
        h = max(0, yy2 - yy1)
        wh = w * h
        o = wh / ((box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - wh + 1e-6)
        return o