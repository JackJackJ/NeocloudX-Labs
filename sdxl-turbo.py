#!/usr/bin/env python3 
import torch 
import random
import os
from datetime import datetime
from diffusers import AutoPipelineForText2Image 
import gradio as gr 

# --- CONFIGURATION --- 
MODEL_ID = "stabilityai/sdxl-turbo" 
OUTPUT_FOLDER = "outputs"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print(f"Loading {MODEL_ID}...") 

# SDXL Turbo works best with float16
dtype = torch.float16 

pipe = AutoPipelineForText2Image.from_pretrained( 
    MODEL_ID, 
    torch_dtype=dtype, 
    variant="fp16",
    use_safetensors=True
).to("cuda") 

def generate(prompt, seed, randomize_seed, width, height): 
    # SDXL Turbo is designed for 1 step. 
    steps = 1
    
    if randomize_seed:
        seed = random.randint(0, 2147483647)
        
    print(f"Generating: {prompt} | Steps: {steps} | Seed: {seed}") 
     
    generator = torch.Generator("cuda").manual_seed(int(seed)) 
     
    image = pipe( 
        prompt=prompt, 
        num_inference_steps=steps,  
        guidance_scale=0.0, # SDXL Turbo requires guidance_scale 0.0
        width=int(width), 
        height=int(height), 
        generator=generator 
    ).images[0] 
    
    # Auto-Save Logic
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"sdxl-turbo_{timestamp}_seed-{seed}.png"
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    
    image.save(file_path)
    print(f"Saved image to: {file_path}")

    return image, seed

# --- UI SETUP ---
with gr.Blocks(title="SDXL-Turbo App") as demo:
    gr.Markdown("# SDXL-Turbo")
    gr.Markdown(f"Running SDXL Turbo (Fixed 1 Step). Images are auto-saved to `{OUTPUT_FOLDER}/`.")
    
    with gr.Row():
        # LEFT COLUMN (30% width)
        with gr.Column(scale=3):
            prompt_input = gr.Textbox(
                label="Prompt", 
                lines=5, 
                placeholder="Enter your prompt here...",
                value="A cinematic shot of a futuristic city with neon lights, rain on pavement, cyberpunk aesthetic, highly detailed, 8k resolution."
            )
            with gr.Row():
                seed_input = gr.Number(value=42, label="Seed", precision=0)
                random_check = gr.Checkbox(value=True, label="Randomize Seed")
            
            width_slider = gr.Slider(512, 1024, value=512, step=64, label="Width") 
            height_slider = gr.Slider(512, 1024, value=512, step=64, label="Height") 
            
            run_btn = gr.Button("Generate", variant="primary")
        
        # RIGHT COLUMN (70% width)
        with gr.Column(scale=7):
            result_image = gr.Image(label="SDXL Turbo Generation")
            seed_output = gr.Number(label="Seed Used")

    # Connect the inputs and outputs
    run_btn.click(
        fn=generate,
        inputs=[prompt_input, seed_input, random_check, width_slider, height_slider],
        outputs=[result_image, seed_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True, server_port=7860)
