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

# Default settings
DEFAULT_INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input')
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
DEFAULT_PROMPT = ""
DEFAULT_USE_IMAGE_PROMPT = True     # Whether to extract prompts from image metadata (DEFAULT_PROMPT entry take priority)
DEFAULT_SEED = -1                    # -1 = random
DEFAULT_USE_TEACACHE = False          # TeaCache: faster but may affect hand quality
DEFAULT_VIDEO_LENGTH = 5.0           # Video length in seconds (range: 1-120)
DEFAULT_STEPS = 25                   # Number of sampling steps
DEFAULT_DISTILLED_CFG = 10.0         # Distilled CFG scale
DEFAULT_GPU_MEMORY = 6.0             # GPU inference memory preservation (GB)
DEFAULT_OVERWRITE = False            # Whether to overwrite existing output files

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

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Batch process images to videos")
    
    parser.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_DIR, 
                        help=f"Directory containing input images (default: {DEFAULT_INPUT_DIR})")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, 
                        help=f"Directory to save output videos (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT,
                        help=f"Prompt to guide the generation (default: '{DEFAULT_PROMPT}')")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed, -1 for random (default: {DEFAULT_SEED})")
    parser.add_argument("--use_teacache", action="store_true", default=DEFAULT_USE_TEACACHE,
                        help=f"Use TeaCache - faster but may affect hand quality (default: {DEFAULT_USE_TEACACHE})")
    parser.add_argument("--video_length", type=float, default=DEFAULT_VIDEO_LENGTH,
                        help=f"Total video length in seconds, range 1-120 (default: {DEFAULT_VIDEO_LENGTH})")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help=f"Number of sampling steps, range 1-100 (default: {DEFAULT_STEPS})")
    parser.add_argument("--distilled_cfg", type=float, default=DEFAULT_DISTILLED_CFG,
                        help=f"Distilled CFG scale, range 1.0-32.0 (default: {DEFAULT_DISTILLED_CFG})")
    parser.add_argument("--gpu_memory", type=float, default=DEFAULT_GPU_MEMORY,
                        help=f"GPU memory preservation in GB, range 6-128 (default: {DEFAULT_GPU_MEMORY})")
    parser.add_argument("--use_image_prompt", action="store_true", default=DEFAULT_USE_IMAGE_PROMPT,
                        help=f"Use prompt from image metadata if available (default: {DEFAULT_USE_IMAGE_PROMPT})")
    parser.add_argument("--overwrite", action="store_true", default=DEFAULT_OVERWRITE,
                        help=f"Whether to overwrite existing output files (default: {DEFAULT_OVERWRITE})")
    
    return parser.parse_args()

@torch.no_grad()
def process_single_image(image_path, output_dir, prompt="", n_prompt="", seed=-1, 
                         video_length=5.0, steps=25, gs=10.0, gpu_memory=6.0, 
                         use_teacache=True, high_vram=False, 
                         text_encoder=None, text_encoder_2=None, tokenizer=None, tokenizer_2=None,
                         vae=None, feature_extractor=None, image_encoder=None, transformer=None):
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
        
        # Also copy to the input folder, with error handling for permission denied
        input_dir = str(Path(image_path).parent)
        input_output_filename = os.path.join(input_dir, f'{filename}.mp4')
        try:
            # Check if file exists and is locked (likely being written to or read from)
            if os.path.exists(input_output_filename):
                # Try to open the file to check if it's accessible
                try:
                    with open(input_output_filename, 'a'):
                        pass
                except:
                    print(f"Warning: Output file {input_output_filename} is locked or in use. Skipping copy to input folder.")
                    return final_output_filename
                    
            shutil.copy2(temp_output_filename, input_output_filename)
            print(f"✅ Successfully processed {image_path} -> {input_output_filename}")
        except PermissionError:
            print(f"Warning: Could not copy to {input_output_filename} due to permission error. Output is still available at {final_output_filename}")
        except Exception as e:
            print(f"Warning: Could not copy to input folder: {e}. Output is still available at {final_output_filename}")
            
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
    print(f"  Prompt: {args.prompt if args.prompt else '(Using image metadata)' if args.use_image_prompt else '(Empty)'}")
    print(f"  Video Length: {args.video_length} seconds")
    print(f"  Steps: {args.steps}")
    print(f"  Seed: {args.seed if args.seed != -1 else 'Random'}")
    print(f"  Distilled CFG: {args.distilled_cfg}")
    print(f"  TeaCache: {args.use_teacache}")
    print(f"  GPU Memory: {args.gpu_memory} GB")
    print(f"  Overwrite Existing: {args.overwrite}")
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

    # Process each image
    for i, image_path in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] Processing {image_path}")
        
        # Get prompt from image metadata if enabled
        image_prompt = None
        if args.use_image_prompt:
            image_prompt = get_image_prompt(image_path)
            if image_prompt:
                print(f"Found prompt in image metadata: {image_prompt[:100]}..." if len(image_prompt) > 100 else image_prompt)
            else:
                print(f"No prompt found in image metadata.")
        
        # Use image metadata prompt if available and enabled, otherwise use CLI prompt
        actual_prompt = args.prompt
        if args.use_image_prompt and image_prompt and args.prompt == DEFAULT_PROMPT:
            actual_prompt = image_prompt
            print(f"Using prompt from image metadata")
        else:
            if args.prompt:
                print(f"Using command-line prompt: {args.prompt}")
            else:
                print(f"Using empty prompt (no metadata prompt found and no command-line prompt provided)")
                # Default fallback prompt for completely empty cases - just provides natural motion
                if actual_prompt == "":
                    print(f"Using empty prompt")
                
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
            transformer=transformer
        )

    print("\nAll images processed!")

if __name__ == "__main__":
    main() 
