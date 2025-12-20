#!/usr/bin/env python3
import torch
import gradio as gr
import random
import re
from diffusers import AutoPipelineForText2Image
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# --- CONFIGURATION ---
IMAGE_MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
LLM_MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507" 

# System prompt
SYSTEM_PROMPT_STORY = (
    "You are a visual storyteller. I will give you a topic and a number of scenes. "
    "You must generate a story broken down into exactly that many scenes. "
    "For EACH scene, strictly follow this format:\n\n"
    "SCENE_START\n"
    "NARRATIVE: [The text that tells the story to the reader]\n"
    "VISUAL: [A detailed physical description of the image to generate]\n"
    "SCENE_END\n\n"
    "Do not include any other text, introductions, or conclusions."
)

# --- HARDWARE SETUP ---
device = "cuda"
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

print(f"--- HARDWARE DETECTED ---")
print(f"Device: {device}")
print(f"Precision: {dtype}")

# 1. LOAD IMAGE MODEL
print(f"Loading Image Model: {IMAGE_MODEL_ID}...")
pipe = AutoPipelineForText2Image.from_pretrained(
    IMAGE_MODEL_ID,
    torch_dtype=dtype, 
    use_safetensors=True,
    trust_remote_code=True
).to(device)

# 2. LOAD LLM (INSTRUCT)
print(f"Loading Instruct LLM: {LLM_MODEL_ID}...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              
    bnb_4bit_quant_type="nf4",       
    bnb_4bit_compute_dtype=dtype    
)

try:
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, trust_remote_code=True)
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        quantization_config=bnb_config, 
        device_map="auto",              
        trust_remote_code=True
    )
except Exception as e:
    print(f"Error loading LLM: {e}")
    exit(1)

# --- GENERATION LOGIC ---

def generate_scene_data(topic, num_scenes):
    """Uses the LLM to write the story and split it into N scenes."""
    
    prompt = f"Topic: {topic}\nNumber of Scenes: {num_scenes}"
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_STORY},
        {"role": "user", "content": prompt}
    ]
    
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text_input], return_tensors="pt").to(llm_model.device)
    
    print(f"Generating Story Logic for '{topic}' ({num_scenes} scenes)...")
    
    with torch.no_grad():
        gen_ids = llm_model.generate(
            **model_inputs, 
            max_new_tokens=2048,
            temperature=0.8,
            top_p=0.9
        )
        output_text = tokenizer.decode(gen_ids[0][len(model_inputs.input_ids[0]):], skip_special_tokens=True)

    # Parse the output using specific delimiters
    scenes = []
    raw_scenes = output_text.split("SCENE_START")
    
    for raw in raw_scenes:
        if "SCENE_END" not in raw:
            continue
            
        try:
            # Extract Narrative and Visual parts 
            narrative_part = raw.split("NARRATIVE:")[1].split("VISUAL:")[0].strip()
            visual_part = raw.split("VISUAL:")[1].split("SCENE_END")[0].strip()
            scenes.append({"narrative": narrative_part, "visual": visual_part})
        except IndexError:
            continue
            
    return scenes[:int(num_scenes)] # Ensure we don't return more than requested

def generate_image(prompt, width=1024, height=1024):
    """Helper to generate a single image."""
    seed = random.randint(0, 2**32 - 1)
    generator = torch.Generator(device).manual_seed(seed)
    
    image = pipe(
        prompt=prompt,
        num_inference_steps=8,  # Turbo model settings
        guidance_scale=0.0,
        width=width,
        height=height,
        generator=generator
    ).images[0]
    
    return image

def process_story(topic, num_scenes, progress=gr.Progress()):
    """Main workflow: Generate Story -> Loop Scenes -> Generate Images -> Output Gallery."""
    
    if not topic:
        return None, "Please enter a topic."

    progress(0.1, desc="Writing Story...")
    
    # 1. Generate Text Scenarios
    scenes = generate_scene_data(topic, num_scenes)
    
    if not scenes:
        return None, "Error: LLM failed to format the story correctly. Please try again."

    gallery_results = []
    log_output = f"## Generated Story: {topic}\n\n"
    
    # 2. Loop through scenes and generate images
    total = len(scenes)
    for i, scene in enumerate(scenes):
        progress((i + 1) / (total + 1), desc=f"Generating Scene {i+1}/{total}...")
        
        narrative = scene["narrative"]
        visual_prompt = scene["visual"]
        
        # Log text
        log_output += f"**Scene {i+1}:** {narrative}\n*Prompt: {visual_prompt}*\n---\n"
        
        # Generate Image
        try:
            img = generate_image(visual_prompt)
            # Gallery expects list of tuples: (image, caption)
            gallery_results.append((img, narrative)) 
        except Exception as e:
            print(f"Error on scene {i}: {e}")

    progress(1.0, desc="Done!")
    return gallery_results, log_output

# --- UI ---
custom_css = """
#gallery_container { min-height: 600px; }
.caption-label { font-size: 1.1em; font-weight: bold; }
"""

with gr.Blocks(title="Story Slideshow Generator", css=custom_css) as demo:
    gr.Markdown("# 📖 AI Story Slideshow Generator")
    gr.Markdown("Enter a topic, choose the number of scenes, and watch the AI build a visual narrative.")
    
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            topic_input = gr.Textbox(
                label="Story Topic / Prompt", 
                placeholder="e.g., A robot discovering a flower in a wasteland...",
                lines=3
            )
            
            scene_slider = gr.Slider(
                minimum=3, 
                maximum=10, 
                value=5, 
                step=1, 
                label="Number of Scenes"
            )
            
            generate_btn = gr.Button("Generate Slideshow", variant="primary", size="lg")
            
            # Log area to see what the LLM actually wrote vs the prompt
            log_area = gr.Markdown("### Story Log")
            
        with gr.Column(scale=3):
            # Gallery component allows scrolling through images with captions
            gallery = gr.Gallery(
                label="Story Slideshow", 
                show_label=False, 
                elem_id="gallery_container",
                columns=[1], 
                rows=[1],
                object_fit="contain",
                height="auto"
            )

    generate_btn.click(
        fn=process_story,
        inputs=[topic_input, scene_slider],
        outputs=[gallery, log_area]
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", share=True, server_port=7860)
