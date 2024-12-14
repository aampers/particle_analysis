import cv2
import numpy as np

def analyze_video(video_path, frame_step):
    """
    Analyze a video to extract particle characteristics and motion patterns.
    
    Args:
    - video_path (str): Path to the video file.
    - frame_step (int): Number of frames to skip between extractions for analysis.
    
    Returns:
    - particle_data (dict): Analysis results including particle positions, sizes, and opacities.
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps
    print(f"Analyzing video: {video_path}")
    print(f"Total frames: {frame_count}, FPS: {fps}, Duration: {duration:.2f} seconds")

    particle_data = {"positions": [], "sizes": [], "opacities": []}
    
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret or i % frame_step != 0:
            continue
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        thresh_frame = cv2.adaptiveThreshold(
            blurred_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        contours, _ = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        positions = [cv2.boundingRect(contour)[:2] for contour in contours]
        sizes = [cv2.contourArea(contour) for contour in contours]
        particle_data["positions"].append(positions)
        particle_data["sizes"].append(np.mean(sizes) if sizes else 0)
        particle_data["opacities"].append(np.mean(gray_frame[thresh_frame > 0]) if sizes else 0)

    cap.release()
    return particle_data

