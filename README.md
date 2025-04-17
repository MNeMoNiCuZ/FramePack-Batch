# FramePack Batch Processor

FramePack Batch Processor is a command-line tool that processes a folder of images and transforms them into animated videos using the FramePack I2V model. This tool enables you to batch process multiple images without needing to use the Gradio web interface, and it also allows you to extract and use the prompt used in your original image, if it's saved in the EXIF metadata (like A1111 or other tools does).

### Original Repository
[https://github.com/lllyasviel/FramePack](https://github.com/lllyasviel/FramePack)

### Sample Image to Video
![2024-11-29 - 12 31 35](https://github.com/user-attachments/assets/8a2fe5eb-a0d9-4947-b0b5-89960fd92738)

https://github.com/user-attachments/assets/3cadc2a8-4285-414e-947f-bb6cfb77b532

_Sample image and video, will add a stronger example later._

## Features

- Process multiple images in a single command
- Generate smooth animations from static images
- Customize video length, quality, and other parameters
- Extract prompts from image metadata (optional)
- Works in both high and low VRAM environments
- Skip files that already have generated videos
- Final videos will be copied to the input folder, matching the same name as the input image

## Requirements

- Python 3.10
- PyTorch with CUDA support
- Hugging Face Transformers
- Diffusers
- VRAM: 6GB minimum (works better with 12GB+)

## Installation

1. Clone or download the [original repository](https://github.com/lllyasviel/FramePack)
2. Clone or download the scripts and files from this repository into the same directory
3. Run `venv_create.bat` to set up your environment:
   - Choose your Python version when prompted
   - Accept the default virtual environment name (venv) or choose your own
   - Allow pip upgrade when prompted
   - Allow installation of dependencies from requirements.txt
4. Install the new requirements by running `pip install -r requirements-batch.txt` in your virtual environment

The script will create:
- A virtual environment
- `venv_activate.bat` for activating the environment
- `venv_update.bat` for updating pip

## Usage

1. Place your images in the `input` folder
2. Activate the virtual environment:
   ```
   venv_activate.bat
   ```
3. Run the script with desired parameters:
   ```
   python batch.py [optional input arguments]
   ```
4. Generated videos will be saved in both the `outputs` folder and alongside the original images

### Command Line Options (Input Arguments)

```
--input_dir PATH      Directory containing input images (default: ./input)
--output_dir PATH     Directory to save output videos (default: ./outputs)
--prompt TEXT         Prompt to guide the generation (default: "")
--seed NUMBER         Random seed, -1 for random (default: -1)
--use_teacache        Use TeaCache - faster but may affect hand quality (default: True)
--video_length FLOAT  Total video length in seconds, range 1-120 (default: 1.0)
--steps NUMBER        Number of sampling steps, range 1-100 (default: 5)
--distilled_cfg FLOAT Distilled CFG scale, range 1.0-32.0 (default: 10.0)
--gpu_memory FLOAT    GPU memory preservation in GB, range 6-128 (default: 6.0)
--use_image_prompt    Use prompt from image metadata if available (default: True)
--overwrite           Overwrite existing output videos (default: False)
```

## Examples

### Basic Usage
Process all images in the input folder with default settings:
```
python batch.py
```

### Customizing Output
Generate longer videos with more sampling steps:
```
python batch.py --video_length 10 --steps 25
```

### Using a Custom Prompt
Apply the same prompt to all images:
```
python batch.py --prompt "A character doing some simple body movements"
```

### Using Image Metadata Prompts
Extract and use prompts embedded in image metadata:
```
python batch.py --use_image_prompt
```

### Overwriting Existing Videos
By default, the processor skips images that already have corresponding videos. To regenerate them:
```
python batch.py --overwrite
```

### Processing a Custom Folder
Process images from a different folder:
```
python batch.py --input_dir "my_images" --output_dir "my_videos"
```

## Memory Optimization

The script automatically detects your available VRAM and adjusts its operation mode:
- **High VRAM Mode** (>60GB): All models are kept in GPU memory for faster processing
- **Low VRAM Mode** (<60GB): Models are loaded/unloaded as needed to conserve memory

You can adjust the amount of preserved memory with the `--gpu_memory` option if you encounter out-of-memory errors.

## Tips

- For best results, use square or portrait images with clear subjects
- Increase `steps` for higher quality animations (but slower processing)
- Use `--video_length` to control the duration of the generated videos
- If experiencing hand/finger issues, try disabling TeaCache with `--use_teacache false`
- The first image takes longer to process as models are being loaded
- Use the default skip behavior to efficiently process new images in a folder

## Limitations

- The processor currently only supports a simplified subset of the full FramePack features
- Animation quality may vary depending on the input image and subject position
- High VRAM mode requires a powerful GPU (NVIDIA RTX 3090 or better recommended)

## Future Improvements

- GUI interface for easier use
- Additional customization options
- Support for image conditioning
- Batch size options for faster processing on high-end GPUs
