#!/usr/bin/env python3
import torch
import gradio as gr
import random
from diffusers import AutoPipelineForText2Image
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, BitsAndBytesConfig
from threading import Thread

# --- CONFIGURATION ---
IMAGE_MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"

# This is the base model. We will compress it to 4-bit instantly upon loading.
LLM_MODEL_ID = "Qwen/Qwen2.5-72B-Instruct" 

SYSTEM_PROMPT_INSTRUCT = "You are a helpful AI assistant."
SYSTEM_PROMPT_THINKING = "You are a deep thinking AI. Break down the problem step-by-step."
SYSTEM_PROMPT_IMAGE = (
    "You are an expert AI art prompt engineer. "
    "Rewrite the user description into a detailed text-to-image prompt. "
    "Output ONLY the raw English prompt."
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

# 2. LOAD LLM
print(f"Loading LLM: {LLM_MODEL_ID}...")

# Define the 4-bit Quantization Config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              # Enable 4-bit loading
    bnb_4bit_quant_type="nf4",      # Normalized Float 4 (Best for accuracy)
    bnb_4bit_compute_dtype=dtype    # Compute in bfloat16 (Speed of H100)
)

try:
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, trust_remote_code=True)
    
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        quantization_config=bnb_config,  # Inject the config here
        device_map="auto",               # Automatically spreads across H100
        trust_remote_code=True
    )
except Exception as e:
    print(f"Error loading LLM: {e}")
    exit(1)


# --- GENERATION LOGIC ---

def generate_image_from_prompt(prompt):
    """Generates image, saves to temp file, returns path."""
    seed = random.randint(0, 2**32 - 1)
    generator = torch.Generator(device).manual_seed(seed)
    
    print(f"Generating Image with Seed {seed}...")
    
    # Using BF16 natively (no VAE casting needed)
    image = pipe(
        prompt=prompt,
        num_inference_steps=4, 
        guidance_scale=0.0,    
        width=1024,
        height=1024,
        generator=generator
    ).images[0]
    
    output_path = f"/tmp/z_image_{seed}.png"
    image.save(output_path)
    return output_path, seed

def format_history_for_llm(history, system_prompt):
    """Converts Gradio dict history to strict LLM format (STRINGS ONLY)."""
    messages = [{"role": "system", "content": system_prompt}]

    # remember the last 20 messages
    recent_history = history[-20:] if len(history) > 20 else history
    
    for msg in recent_history:
        role = msg.get("role")
        content = msg.get("content")
        
        # Skip non-string content (Images)
        if not isinstance(content, str):
            continue
            
        messages.append({"role": role, "content": content})
            
    return messages

def chat_response(message, history, mode):
    # 1. IMAGE GENERATION
    if mode == "Image Generation":
        refine_msgs = [{"role": "system", "content": SYSTEM_PROMPT_IMAGE}, {"role": "user", "content": str(message)}]
        text_input = tokenizer.apply_chat_template(refine_msgs, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text_input], return_tensors="pt").to(llm_model.device)
        
        with torch.no_grad():
            gen_ids = llm_model.generate(**model_inputs, max_new_tokens=200)
            refined_prompt = tokenizer.decode(gen_ids[0][len(model_inputs.input_ids[0]):], skip_special_tokens=True)
        
        # Add text response
        history.append({"role": "assistant", "content": f"**Refined Prompt:**\n*{refined_prompt}*\n\nGenerating..."})
        yield history
        
        # Generate Image
        try:
            img_path, seed = generate_image_from_prompt(refined_prompt)
            
            # Use dictionary format for image content (Fixed the Gradio error)
            image_message = {
                "role": "assistant", 
                "content": {"path": img_path, "alt_text": f"Seed: {seed}"}
            }
            history.append(image_message)
            yield history
            
        except Exception as e:
            history.append({"role": "assistant", "content": f"Error: {e}"})
            yield history

    # 2. INSTRUCT / THINKING
    else:
        sys_prompt = SYSTEM_PROMPT_THINKING if mode == "Thinking" else SYSTEM_PROMPT_INSTRUCT
        
        context_msgs = format_history_for_llm(history[:-1], sys_prompt)
        context_msgs.append({"role": "user", "content": str(message)})
        
        text_input = tokenizer.apply_chat_template(context_msgs, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text_input], return_tensors="pt").to(llm_model.device)
        
        streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
        t = Thread(target=llm_model.generate, kwargs=dict(model_inputs, streamer=streamer, max_new_tokens=2048, temperature=0.7))
        t.start()

        history.append({"role": "assistant", "content": ""})
        partial_resp = ""
        for new_token in streamer:
            partial_resp += new_token
            history[-1]["content"] = partial_resp
            yield history

# --- UI ---
custom_css = """
#chatbot { height: 700px; overflow: auto; }
footer { display: none !important; }
"""

with gr.Blocks(title="NeocloudX Labs", css=custom_css) as demo:
    gr.Markdown("# NeocloudX Labs Open Chat")
    
    with gr.Row():
        with gr.Column(scale=1):
            mode_select = gr.Radio(["Instruct", "Thinking", "Image Generation"], value="Instruct", label="Mode")
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(elem_id="chatbot", avatar_images=(None, "https://api.iconify.design/fluent-emoji:robot.svg"))
            msg = gr.Textbox(show_label=False, placeholder="Type here...")
            clear = gr.Button("Clear")

    def user_turn(user_msg, history):
        if history is None: history = []
        return "", history + [{"role": "user", "content": str(user_msg)}]

    def bot_turn(history, mode):
        last_user_msg = history[-1]["content"]
        yield from chat_response(last_user_msg, history, mode)

    msg.submit(user_turn, [msg, chatbot], [msg, chatbot], queue=False).then(bot_turn, [chatbot, mode_select], chatbot)
    clear.click(lambda: [], None, chatbot, queue=False)

demo.queue().launch(server_name="0.0.0.0", share=True, server_port=7860)
