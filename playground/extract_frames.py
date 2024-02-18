import cv2 as cv
import os

# Path to the video file
video_path = '../data/videos/testrec.mp4'

# Directory to save the frames
frames_dir = '../data/frames'

# Create the directory if it doesn't exist
if not os.path.exists(frames_dir):
    os.makedirs(frames_dir)

# Open the video file
cap = cv.VideoCapture(video_path)

# Read the video frame by frame
frame_count = 0

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    # Read frame by frame
    ret, frame = cap.read()

    # If frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Save frame as JPEG file
    save_path = os.path.join(frames_dir, f'frame_{frame_count:04d}.jpg')
    cv.imwrite(save_path, frame)
    print(f'Saved {save_path}')
    frame_count += 1

# When everything done, release the capture
cap.release()
print(f'Extracted {frame_count} frames and saved to {frames_dir}')