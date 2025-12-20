#!/usr/bin/env python3
import torch
import gradio as gr
import random
import re
from diffusers import AutoPipelineForText2Image
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# --- CONFIGURATION ---
IMAGE_MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
LLM_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct" 

# Define available visual styles
VISUAL_STYLES = [
    "Cinematic Photography",
    "Pixar 3D Animation",
    "Japanese Anime",
    "Detailed Digital Art",
    "Comic Book Graphic Novel",
    "Classical Oil Painting",
    "Watercolor Illustration",
    "Retro Sci-Fi Poster Art",
    "Realistic Documentary Style"
]

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

def generate_scene_data(topic, num_scenes, visual_style):
    """Uses the LLM to write the story, incorporating the chosen style."""
    
    # Dynamic system prompt that includes the chosen style and instructs continuity.
    system_prompt_dynamic = (
        f"You are an expert visual storyteller specializing in the '{visual_style}' style. "
        f"I will give you a topic and a number of scenes. "
        f"You must generate a continuous, cohesive story broken down into exactly that many scenes. "
        f"Crucially, ensure that visual elements (characters, environment details, lighting, atmosphere) remain consistent from scene to scene to build a continuous narrative flow.\n\n"
        f"For EACH scene, strictly follow this format:\n\n"
        f"SCENE_START\n"
        f"NARRATIVE: [The engaging text that tells this specific part of the story to the reader.]\n"
        f"VISUAL: [A detailed image prompt describing the scene in the '{visual_style}' style. Mention the subjects, action, environment, lighting, and specific artistic elements of the style.]\n"
        f"SCENE_END\n\n"
        f"Do not include any other text, introductions, or conclusions. Output exactly {num_scenes} blocks."
    )
    
    user_prompt = f"Topic: {topic}\nNumber of Scenes: {num_scenes}"
    
    messages = [
        {"role": "system", "content": system_prompt_dynamic},
        {"role": "user", "content": user_prompt}
    ]
    
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text_input], return_tensors="pt").to(llm_model.device)
    
    print(f"Generating Story Logic for '{topic}' ({num_scenes} scenes, Style: {visual_style})...")
    
    with torch.no_grad():
        gen_ids = llm_model.generate(
            **model_inputs, 
            max_new_tokens=3072, # Increased tokens slightly for more detailed style descriptions
            temperature=0.7, # Slightly lower temperature for better adherence to structure
            top_p=0.9,
            repetition_penalty=1.05
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
            print(f"Warning: Could not parse a scene block correctly. Skipping chunk.")
            continue
            
    return scenes[:int(num_scenes)] 

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

def process_story(topic, num_scenes, visual_style, progress=gr.Progress()):
    """Main workflow: Generate Story -> Loop Scenes -> Generate Images -> Output Gallery."""
    
    if not topic:
        return None, "Please enter a topic."

    progress(0.1, desc=f"Writing Story ({visual_style})...")
    
    # 1. Generate Text Scenarios
    scenes = generate_scene_data(topic, num_scenes, visual_style)
    
    if not scenes or len(scenes) < 1:
        return None, "Error: LLM failed to format the story correctly. Please try again or try a different topic."

    gallery_results = []
    log_output = f"## Generated Story: {topic}\n### Style: {visual_style}\n\n"
    
    # 2. Loop through scenes and generate images
    total = len(scenes)
    for i, scene in enumerate(scenes):
        progress((i + 1) / (total + 1), desc=f"Generating Scene {i+1}/{total}...")
        
        narrative = scene["narrative"]
        visual_prompt = scene["visual"]
        
        # Log text
        log_output += f"**Scene {i+1} Narrative:** {narrative}\n**Visual Prompt:** *{visual_prompt}*\n---\n"
        
        # Generate Image
        try:
            img = generate_image(visual_prompt)
            # Gallery expects list of tuples: (image, caption)
            gallery_results.append((img, narrative)) 
        except Exception as e:
            print(f"Error generating image for scene {i+1}: {e}")
            log_output += f"\n*Error generating image for Scene {i+1}*\n"

    progress(1.0, desc="Done!")
    return gallery_results, log_output

# --- UI ---
custom_css = """
#gallery_container { min-height: 600px; }
.caption-label { font-size: 1.1em; font-weight: bold; }
"""

with gr.Blocks(title="NeocloudX Labs Storyteller", css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📖 NeocloudX Labs Storyteller")
    gr.Markdown("Enter a topic, choose a visual style, and watch the model generate a narrated slideshow.")
    
    with gr.Row():
        with gr.Column(scale=1, min_width=350):
            topic_input = gr.Textbox(
                label="Story Topic / Prompt", 
                placeholder="e.g., A lone astronaut discovering an ancient ruin on Mars...",
                lines=4
            )
            
            style_select = gr.Dropdown(
                choices=VISUAL_STYLES,
                value=VISUAL_STYLES[0],
                label="Visual Style",
                interactive=True
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
            log_area = gr.Markdown("### Story Generation Log", elem_classes=["log-container"])
            
        with gr.Column(scale=3):
            # Gallery component allows scrolling through images with captions
            gallery = gr.Gallery(
                label="Story Slideshow", 
                show_label=False, 
                elem_id="gallery_container",
                columns=[1], 
                rows=[1],
                object_fit="contain",
                height="auto",
                preview=True
            )

    generate_btn.click(
        fn=process_story,
        inputs=[topic_input, scene_slider, style_select],
        outputs=[gallery, log_area]
    )

if __name__ == "__main__":
    # Increased timeout for slower GPUs generating many images
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0", 
        share=True, 
        server_port=7860
    )
