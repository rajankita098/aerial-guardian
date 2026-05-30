import cv2
import collections
import numpy as np

class DroneVisualizer:
    def __init__(self, max_trail_length=30):
        # Keep track of center-point history for drawing trails
        # Format: { track_id: deque([(x1, y1), (x2, y2), ...], maxlen=max_trail_length) }
        self.track_history = collections.defaultdict(lambda: collections.deque(maxlen=max_trail_length))

    def draw(self, frame, active_tracks, fps=None):
        """
        Draws boxes, IDs, trajectory trails, and pipeline telemetry performance metrics.
        
        active_tracks: list of tracks where each track is [x1, y1, x2, y2, track_id]
        """
        # Copy frame to ensure we don't accidentally mutate the baseline image buffer
        out_frame = frame.copy()

        for track in active_tracks:
            x1, y1, x2, y2, track_id = map(int, track)
            
            # Calculate object frame center point
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Record history center point
            self.track_history[track_id].append((center_x, center_y))

            # 1. Draw Tracking "Tail" Trajectory Lines
            trail_points = self.track_history[track_id]
            for i in range(1, len(trail_points)):
                # Calculate fading thickness or color based on age
                thickness = int(np.sqrt(i / float(len(trail_points))) * 2.5) + 1
                cv2.line(out_frame, trail_points[i - 1], trail_points[i], (0, 255, 255), thickness)

            # 2. Draw Target Person Bounding Box
            # Using neon green for high-visibility contrast against aerial terrain
            cv2.rectangle(out_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 3. Draw Track Label Box (Unique ID Assignment)
            label = f"ID: {track_id}"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Bound text label boundaries cleanly above the main target rectangle
            cv2.rectangle(out_frame, (x1, y1 - text_h - 4), (x1 + text_w, y1), (0, 255, 0), -1)
            # Corrected arguments structure: explicit font Scale, color, thickness, line type
            cv2.putText(out_frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # 4. Draw Pipeline FPS Counter (Required Optimization Telemetry)
        if fps is not None:
            fps_label = f"Pipeline Processing Speed: {fps:.1f} FPS"
            cv2.putText(out_frame, fps_label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        return out_frame

    def clear_dead_history(self, active_tracks):
        """Housekeeping to prevent memory leak build-up over long tracking sequences"""
        active_ids = {int(track[4]) for track in active_tracks}
        current_recorded_ids = list(self.track_history.keys())
        
        for k in current_recorded_ids:
            if k not in active_ids:
                del self.track_history[k]