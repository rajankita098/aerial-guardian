import cv2
import os

def images_to_video(sequence_folder, output_video_path, fps=30):
    # Get all jpg images from the sequence folder and sort them numerically
    images = [img for img in os.listdir(sequence_folder) if img.endswith(".jpg")]
    images.sort()

    if not images:
        print(f"No images found in {sequence_folder}")
        return

    # Read the first image to get the frame dimensions
    first_image_path = os.path.join(sequence_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Initialize the video writer to save as an .mp4 file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"Compiling {len(images)} frames into a video. Please wait...")
    
    for img_name in images:
        img_path = os.path.join(sequence_folder, img_name)
        video.write(cv2.imread(img_path))

    video.release()
    print(f"Success! Video saved to: {output_video_path}")

if __name__ == "__main__":
    # Change this path to point to one of your extracted sequence folders
    # Example:
    SOURCE_FOLDER = "data/sequences/uav0000137_00458_v"
    
    # Destination path where your inference.py expects the video
    OUTPUT_PATH = "data/test_input.mp4"
    
    images_to_video(SOURCE_FOLDER, OUTPUT_PATH, fps=30)