import os
import argparse
import torch
import numpy as np
import traceback
import shutil
from pathlib import Path
import random
from tqdm import tqdm
from PIL import Image
import subprocess

# Set environment variable for HuggingFace models cache
os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

import torch
import einops

from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket

# Prompt selection order:
# 1. prompt_list.txt (if use_prompt_list is True). One prompt per line in this .txt-file
# 2. Per-image .txt file (if exists). The .txt-file should share name with the image-file.
# 3. Image metadata (if use_image_prompt is True)
# 4. fallback_prompt. The same will be used for each generation

prompt_list_file   = 'prompt_list.txt'  # File with one prompt per line for batch processing
use_prompt_list_file = False            # Enable to use prompt_list_file for prompts
use_image_prompt   = True               # Use image metadata as prompt if available
fallback_prompt    = ""                 # Fallback prompt if no other prompt source is found

# Other settings
input_dir          = 'input'            # Directory containing input images
output_dir         = 'output'           # Directory to save output videos
seed               = -1                 # Random seed; -1 means random each run
use_teacache       = True               # Use TeaCache for faster processing (may affect hand quality)
video_length       = 5                  # Video length in seconds (range: 1-120)
steps              = 25                 # Number of sampling steps per video
distilled_cfg      = 10.0               # Distilled CFG scale for model guidance
gpu_memory         = 6.0                # GPU memory to preserve (GB)
overwrite          = False              # Overwrite existing output files if True
fix_encoding       = True               # Re-encode video for web compatibility
copy_to_input      = True               # Copy final video to input folder

def get_image_files(directory):
    """Get all image files from directory"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend([f for f in Path(directory).glob(f'*{ext}') if f.is_file()])
        image_files.extend([f for f in Path(directory).glob(f'*{ext.upper()}') if f.is_file()])
    
    # Filter out any non-image files that might have been caught
    image_files = [f for f in image_files if f.suffix.lower() in image_extensions]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_image_files = []
    for f in image_files:
        if f not in seen:
            seen.add(f)
            unique_image_files.append(f)
    
    return sorted(unique_image_files)

def get_image_prompt(image_path):
    """Extract the prompt from an image's metadata"""
    try:
        with Image.open(image_path) as img:
            exif_data = img.info
            if not exif_data:
                return None
                
            # Look for parameters in different possible metadata fields
            prompt = None
            
            # Check standard 'parameters' field (common in SD outputs)
            if 'parameters' in exif_data:
                params = exif_data['parameters']
                # Extract just positive prompt if there's a negative prompt section
                positive_end = params.find('Negative prompt:')
                if positive_end != -1:
                    prompt = params[:positive_end].strip()
                else:
                    prompt = params.strip()
                    
            # Check for other common metadata fields if parameters wasn't found
            elif 'prompt' in exif_data:
                prompt = exif_data['prompt']
            elif 'Comment' in exif_data:
                # Some tools store in Comment field
                prompt = exif_data['Comment']
                
            # Handle case where metadata exists but prompt is empty
            if prompt and len(prompt.strip()) == 0:
                return None
                
            return prompt
                
    except Exception as e:
        print(f"Warning: Error extracting metadata from {image_path}: {e}")
        return None

def fix_video_encoding(input_path):
    """Re-encode video to ensure web compatibility with minimal quality loss using FFmpeg"""
    try:
        input_path = Path(input_path)
        output_path = input_path.with_stem(input_path.stem + "_fixed")
        
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", str(input_path),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "17",  # Lower CRF for high quality
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            "-pix_fmt", "yuv420p",
            "-y",
            str(output_path)
        ]
        
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
        print(f"Successfully fixed encoding: {input_path} -> {output_path}")
        return output_path
        
    except subprocess.CalledProcessError as e:
        print(f"Error fixing encoding for {input_path}: {e.stderr}")
        return None
    except FileNotFoundError:
        print("Error: FFmpeg is not installed or not found in PATH. Please install FFmpeg.")
        return None
    except Exception as e:
        print(f"Unexpected error fixing encoding for {input_path}: {str(e)}")
        return None

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Batch process images to generate videos")
    
    parser.add_argument("--input_dir", type=str, default=input_dir, 
                        help=f"Directory containing input images (default: {input_dir})")
    parser.add_argument("--output_dir", type=str, default=output_dir, 
                        help=f"Directory to save output videos (default: {output_dir})")
    parser.add_argument("--prompt", type=str, default=fallback_prompt,
                        help=f"Prompt to guide the generation (fallback: '{fallback_prompt}')")
    parser.add_argument("--seed", type=int, default=seed,
                        help=f"Random seed, -1 for random (default: {seed})")
    parser.add_argument("--use_teacache", action="store_true", default=use_teacache,
                        help=f"Use TeaCache - faster but may affect hand quality (default: {use_teacache})")
    parser.add_argument("--video_length", type=float, default=video_length,
                        help=f"Total video length in seconds, range 1-120 (default: {video_length})")
    parser.add_argument("--steps", type=int, default=steps,
                        help=f"Number of sampling steps, range 1-100 (default: {steps})")
    parser.add_argument("--distilled_cfg", type=float, default=distilled_cfg,
                        help=f"Distilled CFG scale, range 1.0-32.0 (default: {distilled_cfg})")
    parser.add_argument("--gpu_memory", type=float, default=gpu_memory,
                        help=f"GPU memory preservation in GB, range 6-128 (default: {gpu_memory})")
    parser.add_argument("--use_image_prompt", action="store_true", default=use_image_prompt,
                        help="Use image metadata for prompt if available")
    parser.add_argument("--overwrite", action="store_true", default=overwrite,
                        help=f"Whether to overwrite existing output files (default: {overwrite})")
    parser.add_argument("--fix_encoding", action="store_true", default=fix_encoding,
                        help=f"Fix video encoding for web compatibility (default: {fix_encoding})")
    parser.add_argument("--use_prompt_list_file", action="store_true", default=use_prompt_list_file,
                        help=f"Use prompt list file (default: {use_prompt_list_file})")
    parser.add_argument("--prompt_list_file", type=str, default=prompt_list_file,
                        help=f"Path to prompt list file (default: '{prompt_list_file}')")
    parser.add_argument("--copy_to_input", action="store_true", default=copy_to_input,
                        help=f"Copy final video to input folder (default: {copy_to_input})")
    
    return parser.parse_args()

@torch.no_grad()
def process_single_image(image_path, output_dir, prompt="", n_prompt="", seed=-1, 
                         video_length=5.0, steps=25, gs=10.0, gpu_memory=6.0, 
                         use_teacache=True, high_vram=False, 
                         text_encoder=None, text_encoder_2=None, tokenizer=None, tokenizer_2=None,
                         vae=None, feature_extractor=None, image_encoder=None, transformer=None,
                         fix_encoding=True, copy_to_input=True):
    """Process a single image to generate a video"""
    
    job_id = generate_timestamp()
    filename = Path(image_path).stem
    
    # Use random seed if seed is -1
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
        print(f"Using random seed: {seed}")
    
    # Calculate total latent sections based on video length
    latent_window_size = 9  # Fixed parameter
    total_latent_sections = (video_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))
    
    try:
        # Clean GPU
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

        # Text encoding
        print("Text encoding...")
        if not high_vram:
            fake_diffusers_current_device(text_encoder, gpu)
            load_model_as_complete(text_encoder_2, target_device=gpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        # Fixed CFG parameter
        cfg = 1.0
        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # Processing input image
        print("Image processing...")
        input_image = np.array(Image.open(image_path).convert('RGB'))
        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)

        # Save resized reference image
        os.makedirs(output_dir, exist_ok=True)
        Image.fromarray(input_image_np).save(os.path.join(output_dir, f'{job_id}.png'))

        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        # VAE encoding
        print("VAE encoding...")
        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)

        start_latent = vae_encode(input_image_pt, vae)

        # CLIP Vision encoding
        print("CLIP Vision encoding...")
        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)

        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        # Convert tensors to appropriate dtype
        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # Sampling
        print("Starting sampling...")
        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3

        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        total_generated_latent_frames = 0

        latent_paddings = reversed(range(total_latent_sections))

        if total_latent_sections > 4:
            # In theory the latent_paddings should follow the above sequence, but it seems that duplicating some
            # items looks better than expanding it when total_latent_sections > 4
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        for latent_padding in tqdm(list(latent_paddings), desc=f"Processing {Path(image_path).name}"):
            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size

            print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}')

            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            clean_latents_pre = start_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            if not high_vram:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory)

            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            # Fixed guidance_rescale parameter
            rs = 0.0

            def callback(d):
                current_step = d['i'] + 1
                if current_step % 5 == 0:  # Show progress every 5 steps
                    print(f"Step {current_step}/{steps} - Total frames: {int(max(0, total_generated_latent_frames * 4 - 3))}")
                return

            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=num_frames,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=gpu,
                dtype=torch.bfloat16,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )

            if is_last_section:
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

            if not high_vram:
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(vae, target_device=gpu)

            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                overlapped_frames = latent_window_size * 4 - 3

                current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

            if not high_vram:
                unload_complete_models()

            temp_output_filename = os.path.join(output_dir, f'{job_id}_{total_generated_latent_frames}.mp4')
            save_bcthw_as_mp4(history_pixels, temp_output_filename, fps=30)

            print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')

            if is_last_section:
                break

        # Create final output with original filename
        final_output_filename = os.path.join(output_dir, f'{filename}.mp4')
        
        # Copy the last temp file to the final output filename
        shutil.copy2(temp_output_filename, final_output_filename)
        
        # Handle encoding fix and copying to input folder
        input_dir = str(Path(image_path).parent)
        input_output_filename = os.path.join(input_dir, f'{filename}.mp4')
        
        if copy_to_input:
            try:
                # Check if file exists and is locked
                if os.path.exists(input_output_filename):
                    try:
                        with open(input_output_filename, 'a'):
                            pass
                    except:
                        print(f"Warning: Output file {input_output_filename} is locked or in use. Skipping copy to input folder.")
                        return final_output_filename
                
                # Fix encoding if enabled
                if fix_encoding:
                    fixed_output = fix_video_encoding(final_output_filename)
                    if fixed_output:
                        # Use the fixed video for copying
                        shutil.copy2(fixed_output, input_output_filename)
                        print(f"✅ Successfully processed and fixed {image_path} -> {input_output_filename}")
                        # Remove the fixed temporary file
                        os.remove(fixed_output)
                    else:
                        print(f"Warning: Encoding fix failed. Copying original video to {input_output_filename}")
                        shutil.copy2(final_output_filename, input_output_filename)
                        print(f"✅ Successfully processed {image_path} -> {input_output_filename}")
                else:
                    shutil.copy2(final_output_filename, input_output_filename)
                    print(f"✅ Successfully processed {image_path} -> {input_output_filename}")
                    
            except PermissionError:
                print(f"Warning: Could not copy to {input_output_filename} due to permission error. Output is still available at {final_output_filename}")
            except Exception as e:
                print(f"Warning: Could not copy to input folder: {e}. Output is still available at {final_output_filename}")
            
        else:
            if fix_encoding:
                fixed_output = fix_video_encoding(final_output_filename)
                if fixed_output:
                    # Replace the original output with the fixed version
                    shutil.move(fixed_output, final_output_filename)
                    print(f"✅ Successfully processed and fixed {image_path} -> {final_output_filename}")
                else:
                    print(f"Warning: Encoding fix failed. Keeping original video at {final_output_filename}")
                    print(f"✅ Successfully processed {image_path} -> {final_output_filename}")
            else:
                print(f"✅ Successfully processed {image_path} -> {final_output_filename}")
                
        return final_output_filename
        
    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")
        traceback.print_exc()
        
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )
        return None

def main():
    args = parse_args()
    
    # Check if input directory exists
    if not os.path.isdir(args.input_dir):
        print(f"Input directory {args.input_dir} does not exist. Creating...")
        os.makedirs(args.input_dir, exist_ok=True)
        print(f"Please add images to {args.input_dir} and run again.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all image files from input directory
    image_files = get_image_files(args.input_dir)
    if not image_files:
        print(f"No image files found in {args.input_dir}")
        return
    
    print(f"Found {len(image_files)} image files:")
    for i, img in enumerate(image_files):
        print(f"  {i+1}. {img.name}")
        
    # Check for existing output files if overwrite is disabled
    if not args.overwrite:
        skipped_files = []
        files_to_process = []
        
        for img_path in image_files:
            # Check if output MP4 already exists
            output_mp4 = os.path.join(args.input_dir, f"{img_path.stem}.mp4")
            if os.path.exists(output_mp4):
                skipped_files.append(img_path)
            else:
                files_to_process.append(img_path)
        
        if skipped_files:
            print(f"\nSkipping {len(skipped_files)} files that already have MP4 outputs:")
            for i, img in enumerate(skipped_files):
                print(f"  {i+1}. {img.name}")
            
        image_files = files_to_process
        
        if not image_files:
            print(f"\nNo files to process. All images already have corresponding MP4 files.")
            print(f"Use --overwrite to regenerate videos for existing files.")
            return
    
    # Print batch processing settings
    print("\nBatch Processing Settings:")
    print(f"  Input Directory: {args.input_dir}")
    print(f"  Output Directory: {args.output_dir}")
    # Determine prompt source for settings printout (accurate to per-image .txt, prompt list, image metadata, or fallback)
    prompt_desc = None
    per_image_txt_exists = all(os.path.exists(str(img.with_suffix('.txt'))) for img in image_files)
    if per_image_txt_exists and len(image_files) > 0:
        prompt_desc = "(Using per-image .txt files)"
    elif args.use_prompt_list_file and os.path.exists(args.prompt_list_file):
        prompt_desc = f"(Using prompt list: {args.prompt_list_file})"
    elif args.use_image_prompt:
        prompt_desc = "(Using image metadata)"
    elif args.prompt:
        prompt_desc = args.prompt
    else:
        prompt_desc = f"(Fallback: '{fallback_prompt}')"
    print(f"  Prompt: {prompt_desc}")
    print(f"  Video Length: {args.video_length} seconds")
    print(f"  Steps: {args.steps}")
    print(f"  Seed: {args.seed if args.seed != -1 else 'Random'}")
    print(f"  Distilled CFG: {args.distilled_cfg}")
    print(f"  TeaCache: {args.use_teacache}")
    print(f"  GPU Memory: {args.gpu_memory} GB")
    print(f"  Overwrite Existing: {args.overwrite}")
    print(f"  Fix Encoding: {args.fix_encoding}")
    print(f"  Copy to Input: {args.copy_to_input}")

    print(f"\nProcessing {len(image_files)} images...")
    
    # Check VRAM and set high_vram mode
    free_mem_gb = get_cuda_free_memory_gb(gpu)
    high_vram = free_mem_gb > 60
    
    print(f'Free VRAM {free_mem_gb} GB')
    print(f'High-VRAM Mode: {high_vram}')
    
    # Load models - following the same pattern as demo_gradio.py
    print("Loading models...")
    text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
    text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
    tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
    tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
    vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()

    feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
    image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()

    transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()

    # Set models to evaluation mode
    vae.eval()
    text_encoder.eval()
    text_encoder_2.eval()
    image_encoder.eval()
    transformer.eval()

    # Configure models for low VRAM mode
    if not high_vram:
        vae.enable_slicing()
        vae.enable_tiling()

    # Set high quality output for transformer
    transformer.high_quality_fp32_output_for_inference = True
    print('transformer.high_quality_fp32_output_for_inference = True')

    # Set appropriate data types for models
    transformer.to(dtype=torch.bfloat16)
    vae.to(dtype=torch.float16)
    image_encoder.to(dtype=torch.float16)
    text_encoder.to(dtype=torch.float16)
    text_encoder_2.to(dtype=torch.float16)

    # Disable gradient calculation for all models
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    image_encoder.requires_grad_(False)
    transformer.requires_grad_(False)

    # Install DynamicSwap for models in low VRAM mode
    if not high_vram:
        DynamicSwapInstaller.install_model(transformer, device=gpu)
        DynamicSwapInstaller.install_model(text_encoder, device=gpu)
    else:
        # Load all models to GPU for high VRAM mode
        text_encoder.to(gpu)
        text_encoder_2.to(gpu)
        image_encoder.to(gpu)
        vae.to(gpu)
        transformer.to(gpu)

    # Priority 1: Prompt list (prompt_list.txt, if use_prompt_list and prompt_list_file exists)
    prompt_list = None
    prompt_list_path = None
    if args.use_prompt_list_file and os.path.exists(args.prompt_list_file):
        prompt_list_path = args.prompt_list_file
    if prompt_list_path is not None:
        with open(prompt_list_path, 'r', encoding='utf-8') as f:
            prompt_list = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(prompt_list)} prompts from prompts.txt (project root)")

    # Process each image
    for i, image_path in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] Processing {image_path}")
        actual_prompt = None
        prompt_source = None

        # Priority 1: Project-wide prompts.txt
        if prompt_list is not None:
            if i < len(prompt_list):
                actual_prompt = prompt_list[i]
                prompt_source = f"project prompts.txt line {i+1}"
            else:
                actual_prompt = ""
                prompt_source = "project prompts.txt (no line, using empty prompt)"
        else:
            # Priority 2: Per-image .txt file
            image_txt_path = image_path.with_suffix('.txt')
            if image_txt_path.exists():
                with open(image_txt_path, 'r', encoding='utf-8') as f:
                    actual_prompt = f.read().strip()
                prompt_source = f"per-image .txt ({image_txt_path.name})"
            # Priority 3: Image metadata
            if actual_prompt is None and args.use_image_prompt:
                image_prompt = get_image_prompt(image_path)
                if image_prompt:
                    actual_prompt = image_prompt
                    prompt_source = "image metadata"
            # Priority 4: Fallback prompt
            if actual_prompt is None:
                actual_prompt = fallback_prompt
                if fallback_prompt:
                    prompt_source = "fallback prompt"
                else:
                    prompt_source = "empty prompt"

        print(f"Using prompt from: {prompt_source}")
        if actual_prompt:
            print(f"Prompt: {actual_prompt[:100]}{'...' if len(actual_prompt) > 100 else ''}")
        else:
            print("Prompt: (empty)")

        # Process the image
        process_single_image(
            image_path=image_path,
            output_dir=args.output_dir,
            prompt=actual_prompt,
            n_prompt="",
            seed=args.seed,
            video_length=args.video_length,
            steps=args.steps,
            gs=args.distilled_cfg,
            gpu_memory=args.gpu_memory,
            use_teacache=args.use_teacache,
            high_vram=high_vram,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            vae=vae,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            transformer=transformer,
            fix_encoding=args.fix_encoding,
            copy_to_input=args.copy_to_input
        )

    print("\nAll images processed!")

if __name__ == "__main__":
    main()
