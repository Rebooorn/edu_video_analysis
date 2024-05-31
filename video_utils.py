import cv2
from pathlib import Path

def extract_frames_with_timestamp(video_path, output_folder):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video file is opened successfully
    if not cap.isOpened():
        print("Error: Could not open the video file.")
        return
    
    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get the frames per second (fps) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Loop through each frame
    for frame_num in range(total_frames):
        # Read the frame
        ret, frame = cap.read()
        
        # Check if the frame is read successfully
        if not ret:
            print(f"Error: Could not read frame {frame_num}")
            break
        
        # Calculate the timestamp of the current frame
        timestamp_ms = frame_num * (1000 / fps)
        timestamp_sec = timestamp_ms / 1000.0
        
        # Save the frame
        frame_filename = f"{output_folder}/frame_{frame_num}.jpg"
        cv2.imwrite(frame_filename, frame)
        
        # Print or save the timestamp (you can modify this part as needed)
        print(f"Frame {frame_num}: Timestamp - {timestamp_sec} seconds")

        break
    
    # Release the video capture object
    cap.release()
    print("Frames extracted successfully!")

# Example usage
video_path = "T4_CXY_en.mp4"
output_folder = "output_frames_with_timestamp"
Path(Path(__file__).parent / output_folder).mkdir(exist_ok=True)
extract_frames_with_timestamp(video_path, output_folder)

