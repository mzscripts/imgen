#!/usr/bin/env python3
"""
Instagram Image Generator - Local Python Script
Batch generate high-quality images from prompts using Stable Diffusion

Requirements:
- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- ~8GB RAM minimum, 12GB+ recommended

Installation:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate pillow requests matplotlib

Usage:
python imgen.py
"""

import json
import os
import sys
from pathlib import Path
import requests
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from datetime import datetime
import gc
import time
import argparse
import matplotlib.pyplot as plt

class InstagramImageGenerator:
    def __init__(self, output_dir="./img", model_id="stabilityai/sd-turbo"):
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_dir = Path(output_dir)
        self.model_id = model_id
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"🖥️  Using device: {self.device}")
        print(f"📁 Output directory: {self.output_dir.absolute()}")
        
        if self.device == "cpu":
            print("⚠️  No CUDA GPU detected. Generation will be slower on CPU.")
        
    def setup_pipeline(self):
        """Initialize the Stable Diffusion pipeline"""
        print("🔄 Loading Stable Diffusion model...")
        print("   (This may take a few minutes on first run)")
        
        try:
            # Load pipeline with optimizations
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,  # Disable for faster inference
                requires_safety_checker=False,
                cache_dir="./models"  # Local model cache
            )
            
            # Use DPM++ scheduler for better quality
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )
            
            # Move to device and enable optimizations
            self.pipe = self.pipe.to(self.device)
            
            if self.device == "cuda":
                try:
                    self.pipe.enable_memory_efficient_attention()
                    self.pipe.enable_xformers_memory_efficient_attention()
                    print("✅ CUDA optimizations enabled")
                except Exception as e:
                    print(f"⚠️  Could not enable all CUDA optimizations: {e}")
            
            print("✅ Pipeline ready!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading pipeline: {e}")
            print("💡 Try: pip install --upgrade torch torchvision diffusers transformers")
            return False
    
    def enhance_prompt(self, description):
        """Enhance the prompt for better image quality"""
        quality_modifiers = [
            "high quality", "detailed", "sharp focus", "8k resolution",
            "professional photography", "vibrant colors", "masterpiece",
            "best quality", "ultra detailed"
        ]
        
        # Add quality modifiers
        enhanced = f"{description}, {', '.join(quality_modifiers)}"
        
        # Negative prompt for better quality
        negative_prompt = (
            "blurry, low quality, pixelated, distorted, ugly, "
            "bad anatomy, watermark, signature, text, logo, "
            "worst quality, low resolution, jpeg artifacts"
        )
        
        return enhanced, negative_prompt
    
    def generate_image(self, prompt_data, num_images=1):
        """Generate Instagram-formatted images from prompt data"""
        if self.pipe is None:
            raise ValueError("Pipeline not initialized. Call setup_pipeline() first.")
        
        prompt_id = prompt_data.get("prompt_id", "unknown")
        description = prompt_data.get("description", "")
        
        if not description:
            raise ValueError("No description found in prompt data")
        
        print(f"🎨 Generating image for prompt {prompt_id}...")
        print(f"📝 Description: {description[:100]}...")
        
        # Enhance prompt
        enhanced_prompt, negative_prompt = self.enhance_prompt(description)
        
        # Instagram aspect ratios
        width, height = 1024, 1024  # Square format (1:1)
        
        generated_images = []
        
        for i in range(num_images):
            print(f"🖼️  Generating image {i+1}/{num_images}...")
            
            try:
                # Generate image with high quality settings
                if self.device == "cuda":
                    with torch.autocast(self.device):
                        image = self.pipe(
                            prompt=enhanced_prompt,
                            negative_prompt=negative_prompt,
                            width=width,
                            height=height,
                            num_inference_steps=50,
                            guidance_scale=7.5,
                            generator=torch.Generator(device=self.device).manual_seed(42 + i)
                        ).images[0]
                else:
                    # CPU generation
                    image = self.pipe(
                        prompt=enhanced_prompt,
                        negative_prompt=negative_prompt,
                        width=width,
                        height=height,
                        num_inference_steps=30,  # Reduced for CPU
                        guidance_scale=7.5,
                        generator=torch.Generator(device=self.device).manual_seed(42 + i)
                    ).images[0]
                
                # Save image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"prompt_{prompt_id}_{timestamp}_{i+1}.png"
                filepath = self.output_dir / filename
                
                # Save as high-quality PNG
                image.save(filepath, "PNG", quality=95, optimize=True)
                generated_images.append(filepath)
                
                print(f"✅ Saved: {filename}")
                
            except Exception as e:
                print(f"❌ Error generating image {i+1}: {e}")
                continue
        
        # Clean up GPU memory
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        
        return generated_images

def load_prompts_from_json(file_path):
    """Load prompts from a JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            prompts = data
        elif isinstance(data, dict) and 'prompts' in data:
            prompts = data['prompts']
        else:
            raise ValueError("JSON must be a list or have a 'prompts' key")
        
        print(f"📄 Loaded {len(prompts)} prompts from {file_path}")
        return prompts
    except Exception as e:
        print(f"❌ Error loading JSON: {str(e)}")
        return []

def batch_generate(generator, prompt_list, images_per_prompt=1, save_progress=True):
    """
    Generate images from a list of prompt dictionaries with progress tracking
    """
    all_generated = []
    failed_prompts = []
    progress_file = generator.output_dir / "batch_progress.json"
    
    # Load existing progress if available
    completed_ids = set()
    if save_progress and progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
                completed_ids = set(progress_data.get('completed_ids', []))
                print(f"📊 Resuming batch: {len(completed_ids)} prompts already completed")
        except:
            print("⚠️  Could not load progress file, starting fresh")
    
    total_prompts = len(prompt_list)
    print(f"\n🚀 Starting batch generation: {total_prompts} prompts")
    print(f"🖼️  {images_per_prompt} image(s) per prompt")
    print(f"📱 Instagram format (1024x1024)")
    print("-" * 50)
    
    start_time = time.time()
    
    for i, prompt_dict in enumerate(prompt_list):
        prompt_id = prompt_dict.get("prompt_id", f"unknown_{i}")
        
        # Skip if already completed
        if prompt_id in completed_ids:
            print(f"⏭️  Skipping prompt {prompt_id} (already completed)")
            continue
        
        print(f"\n📋 Processing prompt {i+1}/{total_prompts} (ID: {prompt_id})")
        print(f"📝 Description: {prompt_dict.get('description', '')[:80]}...")
        
        try:
            generated = generator.generate_image(prompt_dict, images_per_prompt)
            if generated:
                all_generated.extend(generated)
                completed_ids.add(prompt_id)
                print(f"✅ Success: Generated {len(generated)} image(s)")
                
                # Save progress
                if save_progress:
                    progress_data = {
                        'completed_ids': list(completed_ids),
                        'total_generated': len(all_generated),
                        'last_updated': datetime.now().isoformat()
                    }
                    with open(progress_file, 'w') as f:
                        json.dump(progress_data, f)
            else:
                failed_prompts.append(prompt_dict)
                print(f"❌ Failed to generate images for prompt {prompt_id}")
                
        except Exception as e:
            failed_prompts.append(prompt_dict)
            print(f"❌ Error with prompt {prompt_id}: {str(e)}")
        
        # Progress update with time estimate
        elapsed = time.time() - start_time
        avg_time_per_prompt = elapsed / max(1, i + 1 - len([p for p in prompt_list[:i+1] if p.get("prompt_id") in completed_ids]))
        remaining = total_prompts - (i + 1)
        eta_minutes = (remaining * avg_time_per_prompt) / 60
        
        print(f"📊 Progress: {i+1}/{total_prompts} | ETA: {eta_minutes:.1f}min")
    
    # Final summary
    total_time = (time.time() - start_time) / 60
    print("\n" + "="*50)
    print("🏁 BATCH GENERATION COMPLETE!")
    print(f"✅ Successfully generated: {len(all_generated)} images")
    print(f"❌ Failed prompts: {len(failed_prompts)}")
    print(f"⏱️  Total time: {total_time:.1f} minutes")
    
    if failed_prompts:
        print("\n⚠️  Failed prompts (you can retry these):")
        for fp in failed_prompts:
            print(f"   - ID {fp.get('prompt_id')}: {fp.get('description', '')[:50]}...")
    
    return all_generated, failed_prompts

def display_images(output_dir, max_images=6):
    """Display generated images"""
    try:
        import matplotlib.pyplot as plt
        
        image_files = list(Path(output_dir).glob("*.png"))
        
        if not image_files:
            print("No images found to display")
            return
        
        # Sort by creation time and show the most recent
        image_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        recent_images = image_files[:max_images]
        
        # Calculate grid size
        cols = 3
        rows = (len(recent_images) + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, img_path in enumerate(recent_images):
            if i < len(axes):
                img = Image.open(img_path)
                axes[i].imshow(img)
                axes[i].set_title(img_path.name, fontsize=8)
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(recent_images), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("matplotlib not available for image display")
    except Exception as e:
        print(f"Error displaying images: {e}")

def main():
    parser = argparse.ArgumentParser(description='Generate Instagram images from prompts')
    parser.add_argument('--prompts', type=str, help='Path to JSON file with prompts')
    parser.add_argument('--output', type=str, default='./img', help='Output directory')
    parser.add_argument('--images-per-prompt', type=int, default=1, help='Images per prompt')
    parser.add_argument('--model', type=str, default='runwayml/stable-diffusion-v1-5', help='Model ID')
    parser.add_argument('--no-display', action='store_true', help='Skip image display')
    
    args = parser.parse_args()
    
    # Sample prompts if no file provided
    if args.prompts:
        prompts = load_prompts_from_json(args.prompts)
        if not prompts:
            print("No valid prompts loaded, using sample prompts")
            prompts = sample_prompts()
    else:
        prompts = sample_prompts()
    
    # Initialize generator
    generator = InstagramImageGenerator(args.output, args.model)
    
    # Setup pipeline
    if not generator.setup_pipeline():
        print("❌ Failed to initialize pipeline")
        sys.exit(1)
    
    # Generate images
    print(f"\n🎯 Ready to generate {len(prompts)} prompts")
    generated_images, failed = batch_generate(
        generator, 
        prompts, 
        images_per_prompt=args.images_per_prompt
    )
    
    # Display results
    if not args.no_display and generated_images:
        print("\n🖼️  Displaying generated images...")
        display_images(generator.output_dir)
    
    # Final summary
    print(f"\n📊 FINAL SUMMARY:")
    print(f"✅ Total images generated: {len(generated_images)}")
    print(f"❌ Failed prompts: {len(failed)}")
    print(f"💾 All images saved to: {generator.output_dir.absolute()}")

def sample_prompts():
    """Sample prompts for testing"""
    return [
        {
            "prompt_id": 1,
            "description": "A cyberpunk street at night with neon lights reflecting on wet pavement, featuring a lone figure in a trench coat"
        },
        {
            "prompt_id": 2,
            "description": "A magical forest clearing with bioluminescent flowers and floating fairy lights dancing around ancient oak trees"
        },
        {
            "prompt_id": 3,
            "description": "A steampunk airship floating above Victorian London, with brass gears and steam billowing from copper pipes"
        },
        {
            "prompt_id": 4,
            "description": "A pixelated jungle on a distant planet, where a tiny explorer in a pith helmet rides a robotic dinosaur, collecting glowing pixelated gems while neon vines pulse in the background."
        },
        {
            "prompt_id": 5,
            "description": "An underwater city made of coral and glass, with mermaids swimming between towering spires while schools of luminescent fish create patterns in the water"
        }
    ]

if __name__ == "__main__":
    print("🎨 Instagram Image Generator - Local Version")
    print("=" * 50)
    
    # Check requirements
    try:
        import torch
        import diffusers
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ Diffusers version: {diffusers.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name()}")
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("\n💡 Install requirements:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("pip install diffusers transformers accelerate pillow requests matplotlib")
        sys.exit(1)
    
    main()