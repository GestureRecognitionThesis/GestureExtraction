from extraction import extract
from processing import process

if __name__ == '__main__':
    # i want to use frame 23
    wanted_frame = 42
    video_path = '../data/videos/testrec.mp4'
    extracted_frames = extract(video_path)
    process(extracted_frames[wanted_frame])
