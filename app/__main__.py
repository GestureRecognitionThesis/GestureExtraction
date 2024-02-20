from extraction import extract
from processing import process_mp

if __name__ == '__main__':
    # i want to use frame 23
    wanted_frame = 42
    video_path = '../data/videos/testrec.mp4'
    extracted_frames = extract(video_path)
    frame_data: list = process_mp(extracted_frames[wanted_frame])
    print(frame_data[3].relative)
