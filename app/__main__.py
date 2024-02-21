from extraction import extract
from processing import process_mp
from processing import calculate_line_equation

if __name__ == '__main__':
    # i want to use frame 23
    wanted_frame = 24
    video_path = '../data/videos/testrec.mp4'
    extracted_frames = extract(video_path)
    frame_data1: list = process_mp(extracted_frames[wanted_frame])
    frame_data2: list = process_mp(extracted_frames[wanted_frame + 1])
    #print(frame_data[3].relative[2])
    equation = calculate_line_equation(frame_data1[3], frame_data2[3])
    print(equation)
