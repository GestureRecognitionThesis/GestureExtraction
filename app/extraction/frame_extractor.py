import cv2 as cv


def extract(video_path: str) -> list:
    # Open the video file
    cap = cv.VideoCapture(video_path)

    # Read the video frame by frame
    frame_count : int = 0

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Create a list to store the frames
    frames : list = []

    while True:
        # Read frame by frame
        ret, frame = cap.read()

        # If frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Save frame as JPEG file
        frames.append(frame)
        frame_count += 1

    # When everything done, release the capture
    cap.release()
    print(f'Extracted {frame_count} frames')
    return frames
