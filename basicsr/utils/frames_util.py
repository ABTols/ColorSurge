import cv2
import os
import glob
def video2grayscale_frames_and_txtimg2tensor():
    pass

def video2grayscale_frames_and_txt(video_path, image_dir, txt_out='frames_paths.txt'):
    os.makedirs(image_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    idx = 0
    fps = cap.get(cv2.CAP_PROP_FPS)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_save_path = os.path.join(image_dir, f"frame_{idx:04d}.jpg")
        cv2.imwrite(img_save_path, gray)
        frame_paths.append(img_save_path)
        idx += 1
    cap.release()

    num_frames = len(frame_paths)
    assert num_frames >= 16, f'The video has only {num_frames} frames, less than 16, cannot proceed.'

    # Group every 16 frames; pad the last group with the last 16 frames if needed
    groups = []
    i = 0
    while i < num_frames:
        group = frame_paths[i:i+16]
        if len(group) < 16:
            group = frame_paths[-16:]
        groups.append(group)
        i += 16

    with open(txt_out, 'w', encoding='utf-8') as f:
        for g in groups:
            f.write(str(g) + '\n')

    print(f"Grayscale frames are saved to {image_dir}.")
    print(f"Frame path groups are written to {txt_out}.")
    print(f"Total groups: {len(groups)}")
    return fps


def images_to_video(img_folder, output_path, img_ext='jpg', fps=25):
    img_files = sorted(glob.glob(os.path.join(img_folder, f'*.{img_ext}')))
    if not img_files:
        print(f"No images with extension {img_ext} found!")
        return

    # Read the first image to get frame size
    frame = cv2.imread(img_files[0])
    if frame is None:
        print("Cannot read the first image.")
        return
    h, w, _ = frame.shape

    # Define the video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Write all images to the video
    for img_file in img_files:
        img = cv2.imread(img_file)
        if img is not None:
            out.write(img)
        else:
            print(f"Warning: Could not read image {img_file}, skipped.")
    out.release()
    print(f"Video created successfully: {output_path}")