from utils.utils import argument_parser_video, process_video
from pathlib import Path


save_folder_path = Path('results')

if __name__ == '__main__':
    args = argument_parser_video()

    output_video_path = save_folder_path / f"{Path(args.video_path).stem}_output.avi"

    # video processing
    process_video(args.video_path, str(output_video_path))

    print(f"Video saved in: {output_video_path}")