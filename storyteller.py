#!/usr/bin/env python3
import torch
import gradio as gr
import random
import re
from threading import Thread
from diffusers import AutoPipelineForText2Image
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer

# --- CONFIGURATION ---
IMAGE_MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
LLM_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"

# Standard presets
VISUAL_STYLES = [
    "Cinematic Photography",
    "Pixar 3D Animation",
    "Japanese Anime",
    "Detailed Digital Art",
    "Comic Book Graphic Novel",
    "Classical Oil Painting",
    "Watercolor Illustration",
    "Retro Sci-Fi Poster Art",
    "Realistic Documentary Style",
    "Dark Fantasy Concept Art"
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

def generate_scene_data(topic, num_scenes, visual_style, progress_callback):
    """
    Uses the LLM to write the story. 
    Uses a Streamer to update the progress bar every time a new scene starts.
    """
    
    # System prompt embedding the style and structure
    system_prompt_dynamic = (
        f"You are an expert visual storyteller specializing in the '{visual_style}' style. "
        f"I will give you a topic and a number of scenes. "
        f"You must generate a continuous, cohesive story broken down into exactly that many scenes. "
        f"Crucially, ensure that visual elements (characters, environment details, lighting, atmosphere) remain consistent from scene to scene.\n\n"
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
    
    # Setup Streamer
    streamer = TextIteratorStreamer(tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True)
    
    # Run generation in a separate thread so we can consume the stream
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=3072,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.05
    )
    t = Thread(target=llm_model.generate, kwargs=generate_kwargs)
    t.start()

    # --- Consume Stream & Update Progress ---
    output_text = ""
    scenes_found = 0
    
    for new_text in streamer:
        output_text += new_text
        
        # Check how many scenes we have written so far
        current_count = output_text.count("SCENE_START")
        if current_count > scenes_found:
            scenes_found = current_count
            # Update Gradio Progress Bar dynamically
            # We map the writing phase to 0.0 -> 0.4 range of the total progress
            prog_val = 0.05 + (0.35 * (scenes_found / num_scenes))
            progress_callback(prog_val, desc=f"Writing Story (Scene {scenes_found}/{num_scenes})...")

    # Parse the final collected text
    scenes = []
    raw_scenes = output_text.split("SCENE_START")
    
    for raw in raw_scenes:
        if "SCENE_END" not in raw:
            continue
            
        try:
            narrative_part = raw.split("NARRATIVE:")[1].split("VISUAL:")[0].strip()
            visual_part = raw.split("VISUAL:")[1].split("SCENE_END")[0].strip()
            scenes.append({"narrative": narrative_part, "visual": visual_part})
        except IndexError:
            continue
            
    return scenes[:int(num_scenes)]

def generate_image(prompt, width=1024, height=1024):
    """Helper to generate a single image."""
    seed = random.randint(0, 2**32 - 1)
    generator = torch.Generator(device).manual_seed(seed)
    
    image = pipe(
        prompt=prompt,
        num_inference_steps=8, 
        guidance_scale=0.0,
        width=width,
        height=height,
        generator=generator
    ).images[0]
    
    return image

def process_story(topic, num_scenes, dropdown_style, custom_style, progress=gr.Progress()):
    """Main workflow handling logic."""
    
    if custom_style and custom_style.strip():
        final_style = custom_style.strip()
    else:
        final_style = dropdown_style

    if not topic:
        return None, "Please enter a topic."

    # 1. Generate Text Scenarios (Streaming Updates)
    # We pass the 'progress' object into the generator
    scenes = generate_scene_data(topic, num_scenes, final_style, progress)
    
    if not scenes or len(scenes) < 1:
        return None, "Error: LLM failed to format the story correctly. Please try again."

    gallery_results = []
    log_output = f"## Generated Story: {topic}\n### Style: {final_style}\n\n"
    
    # 2. Loop through scenes and Generate Images
    # We map the Image phase to 0.4 -> 1.0 range of total progress
    total = len(scenes)
    for i, scene in enumerate(scenes):
        # Calculate progress starting from 0.4
        current_prog = 0.4 + (0.6 * ((i + 1) / total))
        progress(current_prog, desc=f"Generating Image {i+1}/{total}...")
        
        narrative = scene["narrative"]
        raw_visual_prompt = scene["visual"]
        
        # Force style in prompt
        final_image_prompt = f"{final_style} style. {raw_visual_prompt}"
        
        log_output += f"**Scene {i+1} Narrative:** {narrative}\n"
        log_output += f"**Final Prompt:** *{final_image_prompt}*\n---\n"
        
        try:
            img = generate_image(final_image_prompt)
            gallery_results.append((img, narrative)) 
        except Exception as e:
            print(f"Error generating image for scene {i+1}: {e}")
            log_output += f"\n*Error generating image for Scene {i+1}*\n"

    progress(1.0, desc="Done!")
    return gallery_results, log_output

# --- UI ---
custom_css = """
/* 1. Wrap the text and center it */
.caption-label {
    white-space: pre-wrap !important;
    overflow-wrap: break-word !important;
    text-align: center;
    line-height: 1.5;
    
    /* 2. Create a nice translucent box for readability */
    background-color: rgba(0, 0, 0, 0.6) !important;
    color: white !important;
    border-radius: 8px;
    padding: 10px 15px !important;
    margin: 0 auto;
    
    /* 3. Position it so it doesn't overlap the bottom thumbnails */
    /* This pushes the caption UP away from the thumbnail strip */
    margin-bottom: 60px !important; 
    
    /* 4. Ensure it floats above the image but below modal controls if needed */
    position: relative;
    z-index: 1000;
    max-width: 90%;
}

/* Force specific Gradio gallery captions to behave */
#gallery_container span.caption, 
#gallery_container .caption {
    white-space: pre-wrap !important;
    height: auto !important;
    overflow: visible !important;
}

.log-container { max-height: 400px; overflow-y: scroll; }
"""
with gr.Blocks(title="NeocloudX Labs Storyteller", css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📖 NeocloudX Labs Storyteller")
    gr.Markdown("Enter a topic, choose a visual style, and watch the AI generate a narrated slideshow.")
    
    with gr.Row():
        with gr.Column(scale=1, min_width=350):
            topic_input = gr.Textbox(
                label="Story Topic / Prompt", 
                placeholder="e.g., A lone astronaut discovering an ancient ruin on Mars...",
                lines=3
            )
            
            with gr.Group():
                style_select = gr.Dropdown(
                    choices=VISUAL_STYLES,
                    value=VISUAL_STYLES[0],
                    label="Visual Style Preset",
                    interactive=True
                )
                custom_style_input = gr.Textbox(
                    label="Custom Style (Optional)",
                    placeholder="Type here to override preset (e.g., '1980s VHS Horror')",
                    lines=1
                )

            scene_slider = gr.Slider(
                minimum=3, 
                maximum=10, 
                value=5, 
                step=1, 
                label="Number of Scenes"
            )
            
            generate_btn = gr.Button("Generate Slideshow", variant="primary", size="lg")
            
            log_area = gr.Markdown("### Story Generation Log", elem_classes=["log-container"])
            
        with gr.Column(scale=3):
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
        inputs=[topic_input, scene_slider, style_select, custom_style_input],
        outputs=[gallery, log_area]
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0", 
        share=True, 
        server_port=7860
    )
