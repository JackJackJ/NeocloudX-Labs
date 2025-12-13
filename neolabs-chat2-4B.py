#!/usr/bin/env python3
import torch
import gradio as gr
import random
from diffusers import AutoPipelineForText2Image
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, BitsAndBytesConfig
from threading import Thread

# --- CONFIGURATION ---
IMAGE_MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
LLM_MODEL_ID_INSTRUCT = "Qwen/Qwen3-4B-Instruct-2507" 
LLM_MODEL_ID_THINKING = "Qwen/Qwen3-4B-Thinking-2507"

DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."
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

# 2. LOAD LLMS (INSTRUCT & THINKING)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              
    bnb_4bit_quant_type="nf4",       
    bnb_4bit_compute_dtype=dtype    
)

try:
    # --- Load Instruct Model ---
    print(f"Loading Instruct LLM: {LLM_MODEL_ID_INSTRUCT}...")
    tokenizer_instruct = AutoTokenizer.from_pretrained(LLM_MODEL_ID_INSTRUCT, trust_remote_code=True)
    model_instruct = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID_INSTRUCT,
        quantization_config=bnb_config, 
        device_map="auto",              
        trust_remote_code=True
    )

    # --- Load Thinking Model ---
    print(f"Loading Thinking LLM: {LLM_MODEL_ID_THINKING}...")
    tokenizer_thinking = AutoTokenizer.from_pretrained(LLM_MODEL_ID_THINKING, trust_remote_code=True)
    model_thinking = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID_THINKING,
        quantization_config=bnb_config, 
        device_map="auto",              
        trust_remote_code=True
    )

except Exception as e:
    print(f"Error loading LLMs: {e}")
    exit(1)


# --- GENERATION LOGIC ---

def generate_image_from_prompt(prompt, resolution_str):
    """Generates image, saves to temp file, returns path."""
    try:
        width, height = map(int, resolution_str.split('x'))
    except:
        width, height = 1024, 1024

    seed = random.randint(0, 2**32 - 1)
    generator = torch.Generator(device).manual_seed(seed)
    
    print(f"Generating Image ({width}x{height}) with Seed {seed}...")
    
    image = pipe(
        prompt=prompt,
        num_inference_steps=8, 
        guidance_scale=0.0,    
        width=width,
        height=height,
        generator=generator
    ).images[0]
    
    output_path = f"/tmp/z_image_{seed}.png"
    image.save(output_path)
    return output_path, seed

def format_history_for_llm(history, system_prompt):
    """Converts history to strict LLM format (STRINGS ONLY)."""
    messages = [{"role": "system", "content": system_prompt}]

    recent_history = history[-20:] if len(history) > 20 else history
    
    for msg in recent_history:
        role = msg.get("role")
        content = msg.get("content")
        
        # Skip non-string content (Images stored as dicts)
        if not isinstance(content, str):
            continue
            
        messages.append({"role": role, "content": content})
            
    return messages

def chat_response(message, history_state, mode, custom_sys_prompt, resolution, refine_enabled, thinking_enabled):
    # 1. IMAGE GENERATION
    if mode == "Image":
        target_prompt = str(message)
        
        # --- CONDITIONAL REFINEMENT ---
        if refine_enabled:
            # Use the INSTRUCT model for prompt engineering
            refine_msgs = [{"role": "system", "content": SYSTEM_PROMPT_IMAGE}, {"role": "user", "content": target_prompt}]
            text_input = tokenizer_instruct.apply_chat_template(refine_msgs, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer_instruct([text_input], return_tensors="pt").to(model_instruct.device)
            
            with torch.no_grad():
                gen_ids = model_instruct.generate(**model_inputs, max_new_tokens=200)
                target_prompt = tokenizer_instruct.decode(gen_ids[0][len(model_inputs.input_ids[0]):], skip_special_tokens=True)
            
            history_state.append({"role": "assistant", "content": f"**Refined Prompt:**\n*{target_prompt}*\n\nGenerating ({resolution})..."})
        else:
            history_state.append({"role": "assistant", "content": f"**Generating Raw:**\n*{target_prompt}*\n\n({resolution})..."})
            
        yield history_state
        
        # Generate Image
        try:
            img_path, seed = generate_image_from_prompt(target_prompt, resolution)
            
            image_message = {
                "role": "assistant", 
                "content": {"path": img_path, "alt_text": f"Seed: {seed}"}
            }
            history_state.append(image_message)
            yield history_state
            
        except Exception as e:
            history_state.append({"role": "assistant", "content": f"Error: {e}"})
            yield history_state

    # 2. TEXT GENERATION (INSTRUCT OR THINKING)
    else:
        sys_prompt = custom_sys_prompt if custom_sys_prompt.strip() else DEFAULT_SYSTEM_PROMPT
        
        # Determine which model to use based on toggle
        if thinking_enabled:
            active_model = model_thinking
            active_tokenizer = tokenizer_thinking
            # Prepend a thought indicator if desired, or let the model handle it naturally
        else:
            active_model = model_instruct
            active_tokenizer = tokenizer_instruct
        
        # Shared Context Pool: format_history_for_llm pulls from the same history_state
        context_msgs = format_history_for_llm(history_state[:-1], sys_prompt)
        context_msgs.append({"role": "user", "content": str(message)})
        
        text_input = active_tokenizer.apply_chat_template(context_msgs, tokenize=False, add_generation_prompt=True)
        model_inputs = active_tokenizer([text_input], return_tensors="pt").to(active_model.device)
        
        streamer = TextIteratorStreamer(active_tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
        t = Thread(target=active_model.generate, kwargs=dict(model_inputs, streamer=streamer, max_new_tokens=2048, temperature=0.7))
        t.start()

        history_state.append({"role": "assistant", "content": ""})
        partial_resp = ""
        
        for new_token in streamer:
            partial_resp += new_token
            history_state[-1]["content"] = partial_resp
            yield history_state

# --- UI ---
custom_css = """
#chatbot { height: 85vh !important; overflow: auto; }
footer { display: none !important; }
.caption-container { margin-top: 5px; margin-bottom: 20px; }
.caption { font-size: 0.85em; color: #6b7280; line-height: 1.2; }
"""

with gr.Blocks(title="NeocloudX Labs", css=custom_css, fill_height=True) as demo:
    gr.Markdown("# NeocloudX Labs Open Chat")
    
    # Internal Memory (List of Dicts)
    history_state = gr.State([])

    with gr.Row():
        # --- LEFT SIDEBAR ---
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### Settings")
            
            mode_select = gr.Radio(
                ["Text", "Image"], 
                value="Text", 
                label="Mode"
            )
            
            resolution_select = gr.Dropdown(
                ["1024x1024", "1152x896", "896x1152", "1280x720", "720x1280"],
                value="1024x1024",
                label="Image Resolution",
                interactive=True
            )

            # --- TOGGLE SECTION ---
            with gr.Group():
                refine_prompts_toggle = gr.Checkbox(
                    value=True, 
                    label="Refine Prompts (Image Mode)"
                )
                
                thinking_toggle = gr.Checkbox(
                    value=False,
                    label="Enable Thinking Model (Text Mode)"
                )

                with gr.Column(elem_classes=["caption-container"]):
                    gr.Markdown(
                        "*Thinking Mode switches to Qwen-Thinking logic.*\n*Context is shared across all modes.*", 
                        elem_classes=["caption"]
                    )
            # ---------------------------

            sys_prompt_input = gr.Textbox(
                value=DEFAULT_SYSTEM_PROMPT,
                label="System Prompt",
                lines=5,
                placeholder="Enter custom instructions..."
            )
            
            with gr.Row():
                stop = gr.Button("Stop", variant="stop")
                clear = gr.Button("Clear Chat", variant="secondary")

        # --- RIGHT SIDE ---
        with gr.Column(scale=6):
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                avatar_images=(None, "https://neocloudx.com/logomain.svg")
            )
            msg = gr.Textbox(show_label=False, placeholder="Type your message here...", container=False)

    # --- EVENT HANDLERS ---
    
    def user_turn(user_msg, h_state):
        if h_state is None: h_state = []
        h_state.append({"role": "user", "content": str(user_msg)})
        return "", h_state, h_state

    def bot_turn(h_state, mode, sys_prompt, res, refine_enabled, thinking_enabled):
        last_user_msg = h_state[-1]["content"]
        # Yield updated state directly to chatbot
        for updated_state in chat_response(last_user_msg, h_state, mode, sys_prompt, res, refine_enabled, thinking_enabled):
            yield updated_state, updated_state

    def clear_all():
        return [], [] 

    # Capture the submit event to allow cancellation
    submit_event = msg.submit(
        user_turn, [msg, history_state], [msg, history_state, chatbot], queue=False
    ).then(
        bot_turn, 
        [history_state, mode_select, sys_prompt_input, resolution_select, refine_prompts_toggle, thinking_toggle], 
        [history_state, chatbot]
    )
    
    # Wire the stop button to cancel the submit event
    stop.click(fn=None, inputs=None, outputs=None, cancels=[submit_event])
    
    clear.click(clear_all, None, [history_state, chatbot], queue=False)

demo.queue().launch(server_name="0.0.0.0", share=True, server_port=7860)