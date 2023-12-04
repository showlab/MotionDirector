import cv2
import imageio


def get_total_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames


def extract_frames(input_path, output_path, target_fps, selected_frames):
    total_frames = get_total_frames(input_path)

    video_reader = imageio.get_reader(input_path)
    fps = video_reader.get_meta_data()['fps']

    target_total_frames = selected_frames
    frame_interval = max(1, int(fps / target_fps))
    selected_indices = [int(i * frame_interval) for i in range(target_total_frames)]

    target_frames = [video_reader.get_data(i) for i in selected_indices]
    with imageio.get_writer(output_path, fps=target_fps) as video_writer:
        for frame in target_frames:
            video_writer.append_data(frame)


if __name__ == "__main__":
    input_video_path = "../test_data/zoom/zoom_out.mp4"
    output_video_path = "../test_data/zoom/zoom_out_16.mp4"
    target_fps = 8
    selected_frames = 16

    extract_frames(input_video_path, output_video_path, target_fps, selected_frames)


