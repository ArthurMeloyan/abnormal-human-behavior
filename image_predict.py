from utils.utils import argument_parser_image, process_image
from pathlib import Path

save_folder_path = Path('results')

if __name__ == '__main__':
    args = argument_parser_image()

    output_image_path = save_folder_path / f"{Path(args.image_path).stem}_output.jpg"

    # image processing
    process_image(args.image_path, str(output_image_path))

    print(f"Image saved in: {output_image_path}")