import cv2
import time
import os

# Import our custom modules
from src.detection.detector import DroneDetector
from src.tracking.tracker import DroneTracker
from src.utils.gmc import GlobalMotionCompensation
from src.utils.visualizer import DroneVisualizer

def run_pipeline(video_path, output_path, model_weights='yolov8s.pt'):
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 1. Initialize our specialized components
    print("[INFO] Loading detection model...")
    detector = DroneDetector(model_path=model_weights, conf_threshold=0.25)
    gmc = GlobalMotionCompensation()
    tracker = DroneTracker(max_age=30, min_hits=3)
    visualizer = DroneVisualizer(max_trail_length=30)

    # 2. Open Video Capture Streams
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open input video path: {video_path}")
        return

    # Extract source spatial and framerate configurations
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30.0

    # Initialize video output writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"[INFO] Processing video: {video_path} ({width}x{height} @ {fps} FPS)")
    print("[INFO] Press 'q' to stop execution early.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video streamReached

        start_time = time.perf_counter()

        # Step A: Compute Camera Ego-Motion Transformation Matrix
        transformation_matrix = gmc.apply(frame)

        # Step B: Extract Small-Scale Bounding Box Detections
        detections = detector.detect(frame)

        # Step C: Feed Frame Matrix & Hits into Tracker Engine
        active_tracks = tracker.update(detections, transformation_matrix)

        # Step D: Compute Telemetry Speeds (Frames Per Second)
        end_time = time.perf_counter()
        current_fps = 1.0 / (end_time - start_time)

        # Step E: Render Bounding Boxes, IDs, and Motion Tails
        annotated_frame = visualizer.draw(frame, active_tracks, fps=current_fps)
        visualizer.clear_dead_history(active_tracks)

        # Write output frame to storage file
        out.write(annotated_frame)

        # Optional: Display processing window in real time
        cv2.imshow("The Aerial Guardian - Processing Pipeline", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Execution halted prematurely by user request.")
            break

    # Clean execution memory loops
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[SUCCESS] Processed clip saved safely to: {output_path}")

if __name__ == "__main__":
    # Example relative execution paths
    # Replace these with actual test video variables when your VisDrone sequence is chosen
    INPUT_VIDEO = "data/test_input.mp4"
    OUTPUT_VIDEO = "data/output_tracked.mp4"
    
    # Run the pipeline using baseline weights (or your fine-tuned checkpoint)
    run_pipeline(INPUT_VIDEO, OUTPUT_VIDEO, model_weights='yolov8s.pt')