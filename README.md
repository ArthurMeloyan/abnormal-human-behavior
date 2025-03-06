# Abnormal Human Behavior Detection

This project focuses on detecting abnormal human behavior using computer vision techniques. The model leverages the pre-trained CLIP model from OpenAI, fine-tuned for the specific task of classifying and detecting abnormal behavior in video and image data. Rather than training from scratch, the project uses the power of CLIP’s image-text embeddings to interpret human actions and classify abnormal behavior directly from frames in videos and still images.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)

## Installation

To get started with the project, you need to clone the repository and install the required dependencies. Follow the instructions below:

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/abnormal-human-behavior.git
    cd abnormal-human-behavior
    ```

2. Set up a Python virtual environment:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

Once you have the environment set up, you can start using the project to process images and videos for abnormal human behavior detection.

### Predicting with Images

To predict the behavior in an image, run the following command:

```bash
python image_predict.py --image-path path_to_your_image.jpg
```
By default, the result will be saved in the results folder with _output appended to the filename.
### Predicting with Videos
To predict the behavior in a video, run:
```bash
python video_predict.py --video-path path_to_your_video.mp4
```
By default, the result will be saved in the results folder with _output appended to the filename.

## Project Structure
The project is organized as follows:
```bash
abnormal-human-behavior/
│
├── data/                    # Folder for input data (images, videos)
│
├── results/                 # Folder for output data (processed images, videos)
│
├── models/                  # Folder containing the model and related code
│   └── model.py             # The main model for behavior detection
│
├── utils/                   # Utility functions (e.g., image/video processing)
│   └── utils.py             # Functions for argument parsing, processing, etc.
│
├── settings/ 
│   └── settings.yaml        # Configuration file for project settings (paths, model params, etc.)
│
├── requirements.txt         # Python dependencies
├── .gitignore               # Files and directories to be ignored by Git
├── image_predict.py         # Image processing script
├── video_predict.py         # Video processing script
└── README.md                # Project description and usage

```
## Dependencies
This project requires the following dependencies:

- Python 3.11
- OpenCV
- NumPy
- Matplotlib
- Other dependencies listed in `requirements.txt`
```bash
pip install -r requirements.txt

```
