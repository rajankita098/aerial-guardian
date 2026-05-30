import cv2
import numpy as np

class GlobalMotionCompensation:
    def __init__(self):
        # Using ORB feature detector - lightweight and perfect for real-time edge use
        self.detector = cv2.ORB_create(nfeatures=2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None

    def apply(self, frame):
        """
        Calculates the transformation matrix representing the drone's ego-motion
        between the previous frame and the current frame.
        """
        # Convert frame to grayscale for feature extraction
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Initialize transformation matrix as Identity (no movement)
        transformation_matrix = np.eye(2, 3, dtype=np.float32)

        # First frame processing initialization
        if self.prev_frame is None:
            self.prev_frame = gray
            self.prev_keypoints, self.prev_descriptors = self.detector.detectAndCompute(gray, None)
            return transformation_matrix

        # 1. Detect features in the current frame
        kp, des = self.detector.detectAndCompute(gray, None)

        if des is not None and self.prev_descriptors is not None:
            # 2. Match descriptors between previous and current frames
            matches = self.matcher.match(self.prev_descriptors, des)
            matches = sorted(matches, key=lambda x: x.distance)

            # Extract matching feature locations
            pts_prev = np.float32([self.prev_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            pts_curr = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            if len(matches) > 10:
                # 3. Compute rigid Affine transformation matrix (Translation + Rotation + Scaling)
                # Using RANSAC to eliminate noisy moving objects (like other cars/people) from the calculation
                matrix, inliers = cv2.estimateAffinePartial2D(pts_prev, pts_curr, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                
                if matrix is not None:
                    transformation_matrix = matrix

        # Update historical frame variables for the next iteration
        self.prev_frame = gray
        self.prev_keypoints = kp
        self.prev_descriptors = des

        return transformation_matrix