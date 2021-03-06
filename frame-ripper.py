import getopt
import cv2
from pathlib import Path
from os import mkdir, cpu_count
from tqdm import tqdm
from multiprocessing import Pool
from sys import argv

DIFFERENCE_THRESHOLD = 0.17

def multiprocess_unique_frames(video_folder: str, num_processes = None):
    folder_path = Path(video_folder)

    if not folder_path.is_dir():
        return -1

    video_paths = folder_path.glob("**/*.mp4")

    if num_processes is None:
        num_processes = cpu_count()

    with Pool(num_processes) as pool:
        pool.map(get_unique_frames, [str(x) for x in video_paths])

    return

def get_unique_frames(video_file: str, threshold=DIFFERENCE_THRESHOLD, show_progress_bar=False):
    video_path = Path(video_file)
    video_name = video_path.name.split('.')[0]
    images_folder = video_path.with_name(video_name)
    print(images_folder)

    video = cv2.VideoCapture(video_file)

    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

    valid_frame, frame = video.read()
    current_frame_count: int = 0

    # Make sure we can actually read the first frame from this video
    if valid_frame:
        try:
            mkdir(images_folder)
        except FileExistsError:
            pass
        cv2.imwrite(images_folder.joinpath(f"{video_name}-0h0m0s0.png").as_posix(), frame)
        last_unique_frame = frame
    else:
        return

    valid_frame, frame = video.read()
    current_frame_count += 1

    # Reset all timestamp vars
    frame_time_ms: int = 0
    frame_time_s: int = 0
    frame_time_m: int = 0
    frame_time_h: int = 0

    if(show_progress_bar):
        pbar = tqdm(total=total_frames)
    else:
        pbar = None

    while valid_frame:
        #Calculate current time from fps and number of frames
        total_time_s = current_frame_count / fps
        frame_time_s = int(total_time_s) % 60
        frame_time_m = int((total_time_s / 60)) % 60
        frame_time_h = int(total_time_s / 360)
        frame_time_ms = int(total_time_s * 1000) % 1000

        # Calculate similarity between the last unique frame and this frame using template matching
        similarity = cv2.matchTemplate(frame, last_unique_frame, cv2.TM_CCOEFF_NORMED)
        _, max_v, _, _ = cv2.minMaxLoc(similarity)

        #print(f"max_v:{max_v}")

        # When a unique frame is found, save it as a .png with its timestamp in the file title
        if max_v < threshold:
            print(f"{video_name} {frame_time_h}h{frame_time_m}m{frame_time_s}s{frame_time_ms}ms: {round(max_v * 100, 3)}% similar")
            cv2.imwrite(images_folder.joinpath(f"{video_name}-{frame_time_h}h{frame_time_m}m{frame_time_s}s{frame_time_ms}.png").as_posix(), frame)
            last_unique_frame = frame

        if(show_progress_bar):
            pbar.update(1)

        current_frame_count += 1
        valid_frame, frame = video.read()

    if(show_progress_bar):
        pbar.close()

    print(f"{current_frame_count} frames at {round(fps, 2)} fps processed for a calculated length of {frame_time_h}h{frame_time_m}m{frame_time_s}s{frame_time_ms}ms")

    return

if __name__ == "__main__":

    if len(argv) < 2:
        print('frame-ripper.py [-t threshold] video_folder_path')

    try:
        opts, args = getopt.getopt(argv, "t:")
    except getopt.GetoptError:
        print('frame-ripper.py [-t threshold] video_folder_path')
        exit(2)

    for opt, arg in opts:
        if opt == 't':
            try:
                DIFFERENCE_THRESHOLD = float(arg)
            except ValueError:
                print("Threshold value must be a float")
                exit(2)

    video_folder = Path(argv[len(argv) - 1])
    if not video_folder.exists():
        print(f"could not open \"{video_folder}\"")

    exit(multiprocess_unique_frames(video_folder))