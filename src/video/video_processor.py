class VideoProcessor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.video_capture = None

    def open_video(self):
        import cv2
        self.video_capture = cv2.VideoCapture(self.video_path)

    def extract_frames(self):
        frames = []
        while self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if not ret:
                break
            frames.append(frame)
        self.video_capture.release()
        return frames

    def annotate_frames(self, frames, annotations):
        annotated_frames = []
        for i, frame in enumerate(frames):
            # Assuming annotations is a list of text to overlay on each frame
            cv2.putText(frame, annotations[i], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            annotated_frames.append(frame)
        return annotated_frames

    def extract_clips(self, start_time, end_time):
        import ffmpeg
        out_filename = f'clip_{start_time}_{end_time}.mp4'
        ffmpeg.input(self.video_path, ss=start_time, to=end_time).output(out_filename).run()
        return out_filename

    def get_video_info(self):
        import cv2
        self.open_video()
        width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        self.video_capture.release()
        return {'width': width, 'height': height, 'fps': fps}
