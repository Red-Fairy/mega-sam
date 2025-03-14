import os
import argparse
import cv2
import subprocess

def read_video(video_path, save_root, frame_interval=0, num_frames=37, fps=None, full_sampling=False):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f'Cannot open video file {video_path}')
    fps_this = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if full_sampling:
        frame_interval = (total_frames - 1) // (num_frames - 1)
    elif fps is not None:
        frame_interval = max(round(fps_this / fps), 1)
        

    images = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        images.append(frame)
        if num_frames is not None and len(images) == num_frames:
            break
        for _ in range(frame_interval - 1):
            cap.grab()
    cap.release()

    if num_frames is not None and len(images) < num_frames:
       images += [images[-1]] * (num_frames - len(images))

    for i, image in enumerate(images):
        save_path = os.path.join(save_root, f'{i:06d}.png')
        cv2.imwrite(save_path, image)


def run_mono_depth(image_file_root, seq_name, conda_env=None):

    print(image_file_root, seq_name)

    cmd = [
        'python', 'Depth-Anything/run_videos.py',
        "--encoder", "vitl",
        "--load-from", "Depth-Anything/checkpoints/depth_anything_vitl14.pth",
        "--img-path", os.path.join(image_file_root, seq_name),
        "--outdir", f"Depth-Anything/video_visualization/{seq_name}"
    ]

    if conda_env is not None:
        cmd = ['conda', 'run', '-n', conda_env] + cmd

    subprocess.run(cmd, check=True)

    # do env setup
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{env.get('PYTHONPATH', '')}:{os.getcwd()}/UniDepth"

    cmd = [
        'python', 'UniDepth/scripts/demo_mega-sam.py',
        "--scene-name", seq_name,
        "--img-path", os.path.join(image_file_root, seq_name),
        "--outdir", "UniDepth/outputs"
    ]

    if conda_env is not None:
        cmd = ['conda', 'run', '-n', conda_env] + cmd

    subprocess.run(cmd, check=True, env=env)


def camera_pose_estimation(image_file_root, seq_name, output_dir, conda_env=None):

    cmd = [
        'python', 'camera_tracking_scripts/test_demo.py',
        "--datapath", os.path.join(image_file_root, seq_name),
        "--weights", "checkpoints/megasam_final.pth",
        "--scene_name", seq_name,
        "--mono_depth_path", "Depth-Anything/video_visualization",
        "--metric_depth_path", "UniDepth/outputs",
        "--save_root", output_dir,
        "--disable_vis"
    ]

    if conda_env is not None:
        cmd = ['conda', 'run', '-n', conda_env] + cmd

    subprocess.run(cmd, check=True)

def video_depth_optimization(image_file_root, seq_name, output_dir='outputs_cvd', conda_env=None):

    '''
    1. run RAFT
    python cvd_opt/preprocess_flow.py \
    --datapath=$DATA_PATH/$seq \
    --model=cvd_opt/raft-things.pth \
    --scene_name $seq --mixed_precision

    2. run cvd optimization
    python cvd_opt/cvd_opt.py \
    --scene_name $seq \
    --w_grad 2.0 --w_normal 5.0
    '''

    cmd = [
        'python', 'cvd_opt/preprocess_flow.py',
        "--datapath", os.path.join(image_file_root, seq_name),
        "--model", "cvd_opt/raft-things.pth",
        "--scene_name", seq_name,
        "--mixed_precision"
    ]

    if conda_env is not None:
        cmd = ['conda', 'run', '-n', conda_env] + cmd

    subprocess.run(cmd, check=True)

    cmd = [
        'python', 'cvd_opt/cvd_opt.py',
        "--scene_name", seq_name,
        "--w_grad", "2.0",
        "--w_normal", "5.0",
        "--output_dir", output_dir
    ]

    if conda_env is not None:
        cmd = ['conda', 'run', '-n', conda_env] + cmd

    subprocess.run(cmd, check=True)


def main():

    parser = argparse.ArgumentParser(description='Run trajectory prediction')
    parser.add_argument('input_file_path_or_dir', type=str, help='Input file path or directory')
    parser.add_argument('output_pose_dir', type=str, help='Output pose directory')
    parser.add_argument('--temp_image_dir', type=str, default='cache', help='Temporary image directory')
    parser.add_argument('--fps', type=float, default=8, help='Output video fps')
    parser.add_argument('--num_frames', type=int, default=None, help='Number of frames to process')
    parser.add_argument('--full_sampling', action='store_true', help='Use full sampling')
    parser.add_argument('--num_processes', type=int, default=1, help='Number of processes to run in parallel')
    parser.add_argument('--process_id', type=int, default=0, help='Process ID')
    parser.add_argument('--skip_mono_depth', action='store_true', help='Skip mono-depth')
    parser.add_argument('--output_depth_dir', type=str, default=None, help='Output depth directory')
    parser.add_argument('--mono_depth_env', type=str, default=None, help='Mono-depth environment')
    parser.add_argument('--camera_pose_env', type=str, default=None, help='Camera pose environment')
    parser.add_argument('--video_depth_env', type=str, default=None, help='Video depth environment')

    args = parser.parse_args()

    if args.output_depth_dir is None:
        assert args.skip_mono_depth, 'Output depth directory is required if mono-depth is not skipped'

    if os.path.isdir(args.input_file_path_or_dir):
        video_files = [f for f in os.listdir(args.input_file_path_or_dir) if f.endswith('.mp4')]
        video_files = [os.path.join(args.input_file_path_or_dir, f) for f in video_files]
    else:
        video_files = [args.input_file_path_or_dir]

    video_files = video_files[args.process_id::args.num_processes]

    for video_file in video_files:
        video_name = os.path.basename(video_file)
        video_name = video_name[:video_name.rfind('.')]

        image_file_root = os.path.join(args.temp_image_dir, video_name)
        os.makedirs(image_file_root, exist_ok=True)

        print(f'Processing {video_file}...')
        read_video(video_file, image_file_root, fps=args.fps, num_frames=args.num_frames)

        print('Running mono-depth...')
        run_mono_depth(args.temp_image_dir, video_name, conda_env=args.mono_depth_env)

        print('Running camera pose estimation...')
        camera_pose_estimation(args.temp_image_dir, video_name, args.output_pose_dir, conda_env=args.camera_pose_env)

        if not args.skip_mono_depth:
            print('Running video depth optimization...')
            video_depth_optimization(args.temp_image_dir, video_name, args.output_depth_dir, conda_env=args.video_depth_env)

        print('Done!')
        os.system(f'rm -r {image_file_root}')

if __name__ == '__main__':
    main()
