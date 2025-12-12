!pip install filterpy
!pip install norfair

import numpy as np
import pandas as pd
from norfair import Detection, Tracker
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

#@title Tracker Nearest-neighbour
def link_detections(detections_per_frame: list[list[tuple[int, int]]],
                    max_dist: float = 7.0) -> pd.DataFrame:
    """Link detections across frames into tracks using nearest neighbour association.

    Args:
        detections_per_frame: a list where each element is a list of (x,y)
            detections for the corresponding frame index.
        max_dist: maximum allowed distance in pixels between a detection and
            an existing track’s last position for association.  If no
            detection falls within this radius the track is terminated and
            a new track is started for the unmatched detection.

    Returns:
        A pandas DataFrame with columns ['frame','x','y','track_id'] containing
        the linked tracks.
    """
    next_track_id = 0
    active_tracks: dict[int, tuple[int, int, int]] = {}  # track_id -> (x, y, last_frame)
    records: list[dict[str, int]] = []
    for frame_idx, detections in enumerate(detections_per_frame):
        assigned = [False] * len(detections)
        detection_track_id: list[int | None] = [None] * len(detections)
        updated_tracks: dict[int, tuple[int, int, int]] = {}
        # attempt to match existing tracks to current detections
        for track_id, (tx, ty, last_frame) in list(active_tracks.items()):
            best_dist = max_dist
            best_idx: int | None = None
            for i, (x, y) in enumerate(detections):
                if assigned[i]:
                    continue
                dist = math.hypot(x - tx, y - ty)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            if best_idx is not None:
                # assign detection to this track
               assigned[best_idx] = True
               detection_track_id[best_idx] = track_id
               updated_tracks[track_id] = (detections[best_idx][0], detections[best_idx][1], frame_idx)
            # tracks with no assignment are dropped (no occlusion handling)
        # start new tracks for unmatched detections
        for i, (x, y) in enumerate(detections):
            if not assigned[i]:
                track_id = next_track_id
                next_track_id += 1
                detection_track_id[i] = track_id
                updated_tracks[track_id] = (x, y, frame_idx)
        #update active tracks
        active_tracks = updated_tracks
        #record detections with assigned track ids
        for i, (x, y) in enumerate(detections):
            tid = detection_track_id[i]
            records.append({'frame': frame_idx, 'x': x, 'y': y, 'track_id': tid})
    return pd.DataFrame(records)


#@title Tracker Kalmanův filtr + Hungarian algoritmus

# Track object
class Track:
    def __init__(self, x, y, track_id):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([[1,0,1,0],
                              [0,1,0,1],
                              [0,0,1,0],
                              [0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0],
                              [0,1,0,0]])
        self.kf.R *= 0.5
        self.kf.P *= 10

        # initial position
        self.kf.x[:2] = np.array([[x],[y]])

        self.id = track_id
        self.time_since_update = 0

    def predict(self):
        self.kf.predict()
        x, y = self.kf.x[0][0], self.kf.x[1][0]
        return (x, y)

    def update(self, z):
        self.kf.update(z)
        self.time_since_update = 0


# Hungarian algoritm
def hungarian_association(tracks, detections, max_dist=6):

  """
    max_dist = maximum object displacement between frames
    tracks = list of Track objects
    detections = list of (x, y) tuples
  """

  if len(tracks)==0:
        # no existing tracks → everything is new
        return [], list(range(len(detections))), []

  if len(detections)==0:
        # no detections → all tracks unmatched
        return list(range(len(tracks))), [], []

  cost = np.zeros((len(tracks), len(detections)))

    # compute distance matrix
  for i, t in enumerate(tracks):
        tx, ty = t.predict()
        for j, (x, y) in enumerate(detections):
            cost[i, j] = np.hypot(tx - x, ty - y) #eucleidian

  row_ind, col_ind = linear_sum_assignment(cost)

  matches = []
  unmatched_tracks = list(range(len(tracks)))
  unmatched_detections = list(range(len(detections)))

  for r, c in zip(row_ind, col_ind):
        if cost[r, c] < max_dist:
            matches.append((r, c))
            unmatched_tracks.remove(r)
            unmatched_detections.remove(c)

  return unmatched_tracks, unmatched_detections, matches


# tracking loop
def track_ccp_sequence(detections_per_frame, max_dist):
    tracks = []
    next_id = 0
    records = []

    for frame, detections in enumerate(detections_per_frame):

        # pre-prediction for all tracks
        _ = [t.predict() for t in tracks]

        unmatched_tracks, unmatched_dets, matches = hungarian_association(tracks, detections, max_dist)

        # update matched tracks
        for t_idx, d_idx in matches:
            x, y = detections[d_idx]
            tracks[t_idx].update(np.array([x, y]))
            records.append({
                "frame": frame,
                "x": x,
                "y": y,
                "track_id": tracks[t_idx].id
            })

        # new tracks for unmatched detections
        for d_idx in unmatched_dets:
            x, y = detections[d_idx]
            new_track = Track(x, y, next_id)
            next_id += 1
            tracks.append(new_track)
            records.append({
                "frame": frame,
                "x": x,
                "y": y,
                "track_id": new_track.id
            })

    return pd.DataFrame(records)

#@title Noirfar tracker
def track_with_norfair(detections_per_frame, max_dist=8):
    """
    Track objects using the Norfair multi-object tracker.
    """
    # Initialize Norfair tracker with L1 (Manhattan) distance
    tracker = Tracker(
    distance_function="cityblock",
    distance_threshold=max_dist
)
   
    results = []

    for frame_idx, dets in enumerate(detections_per_frame):

        # Create Norfair detections
        norfair_dets = [
            Detection(points=np.array([[x, y]], dtype=float))
            for (x, y) in dets
        ]

        # Update tracker
        tracked = tracker.update(norfair_dets)

        # Save output
        for t in tracked:
            x, y = t.estimate[0]  # one point

            results.append({
                "frame": frame_idx,
                "x": float(x),
                "y": float(y),
                "track_id": int(t.id)
            })

    return pd.DataFrame(results)
