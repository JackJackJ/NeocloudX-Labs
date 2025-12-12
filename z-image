#!/usr/bin/env python3 
import torch 
import random
import os
from datetime import datetime
from diffusers import AutoPipelineForText2Image 
import gradio as gr 

# --- CONFIGURATION --- 
MODEL_ID = "Tongyi-MAI/Z-Image-Turbo" 
OUTPUT_FOLDER = "outputs"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print(f"Loading {MODEL_ID}...") 

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16 

pipe = AutoPipelineForText2Image.from_pretrained( 
    MODEL_ID, 
    torch_dtype=dtype, 
    use_safetensors=True
).to("cuda") 

def generate(prompt, seed, randomize_seed, width, height): 
    steps = 8
    
    if randomize_seed:
        seed = random.randint(0, 2147483647)
        
    print(f"Generating: {prompt} | Steps: {steps} | Seed: {seed}") 
     
    generator = torch.Generator("cuda").manual_seed(int(seed)) 
     
    image = pipe( 
        prompt=prompt, 
        num_inference_steps=steps,  
        guidance_scale=0.0, 
        width=int(width), 
        height=int(height), 
        generator=generator 
    ).images[0] 
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"z-image_{timestamp}_seed-{seed}.png"
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    
    image.save(file_path)
    print(f"Saved image to: {file_path}")

    return image, seed

with gr.Blocks(title="Z-Image-Turbo") as demo:
    gr.Markdown("# Z-Image-Turbo")
    gr.Markdown(f"Running Z-Image Turbo (Fixed 8 Steps). Images are auto-saved to `{OUTPUT_FOLDER}/`.")
    
    with gr.Row():
        with gr.Column():
            # Inputs
            prompt_input = gr.Textbox(
                label="Prompt", 
                lines=3, 
                value="A sunlit field in late summer, tall grass swaying in a gentle breeze, an abandoned picnic blanket and a forgotten book, warm golden tones. Photorealistic, 8K, sentimental, nostalgic feel."
            )
            with gr.Row():
                seed_input = gr.Number(value=42, label="Seed", precision=0)
                random_check = gr.Checkbox(value=True, label="Randomize Seed")
            
            width_slider = gr.Slider(512, 1536, value=1024, step=64, label="Width") 
            height_slider = gr.Slider(512, 1536, value=1024, step=64, label="Height") 
            
            run_btn = gr.Button("Generate", variant="primary")
        
        with gr.Column():
            # Outputs
            result_image = gr.Image(label="Z-Image Generation")
            seed_output = gr.Number(label="Seed Used")

    # Connect the inputs and outputs
    run_btn.click(
        fn=generate,
        inputs=[prompt_input, seed_input, random_check, width_slider, height_slider],
        outputs=[result_image, seed_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True, server_port=7860)
