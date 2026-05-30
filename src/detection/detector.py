import torch
from ultralytics import YOLO

class DroneDetector:
    def __init__(self, model_path='yolov8s.pt', conf_threshold=0.25):
        """
        Initializes the YOLO small-object detector.
        Defaulting to a lightweight setup that dynamically selects CUDA if present.
        """
        self.conf_threshold = conf_threshold
        
        # Select device automatically (CUDA GPU or CPU)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[INFO] Initializing Detector on device: {self.device.upper()}")
        
        # Load the network model
        self.model = YOLO(model_path).to(self.device)

    def detect(self, frame):
        """
        Runs inference on an incoming frame and parses outputs.
        Returns a list of bounding boxes: [[x1, y1, x2, y2, confidence]]
        """
        # Run inference using optimal image size adjustments for tiny aerial views
        # Setting verbose=False keeps the console clean during high-speed tracking
        results = self.model(frame, imgsz=1280, conf=self.conf_threshold, verbose=False)[0]
        
        detections = []
        
        # Check if any bounding boxes were found
        if results.boxes is not None:
            for box in results.boxes:
                # Extract class index (0 is typically person in custom training)
                cls_id = int(box.cls[0].item())
                
                # Filter strictly for 'Person' class (Class 0)
                if cls_id == 0:
                    # Extract coordinates and confidence score
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].item()
                    
                    detections.append([x1, y1, x2, y2, conf])
                    
        return detections