# NeocloudX Labs: Open Chat & Image Suites

Welcome to the [**NeocloudX Labs**](https://neocloudx.com/labs) repository. This project provides local, multimodal AI interfaces powered by **Qwen 2.5** (LLM) and **Tongyi-MAI Z-Image-Turbo** (Image Generation).

We provide multiple variations of the application ranging from lightweight consumer versions to enterprise-grade flagship models, as well as a standalone image generation tool.

![Multimodal interface with integrated chat and image generation](https://neocloudx.com/labs-screenshot.webp)

## Quick Start

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/JackJackJ/NeocloudX-Labs.git](https://github.com/JackJackJ/NeocloudX-Labs.git)
    cd NeocloudX-Labs
    ```

2.  **Choose your version** (see below) and run the corresponding launch script.
    * *Example:* To run the 14B model:
        ```bash
        chmod +x launch_chat14b.sh
        ./launch_chat14b.sh
        ```

---

## Available Models & Hardware Recommendations

Choose the version that fits your available GPU VRAM. All chat models support **Text-to-Text** and **Text-to-Image** (via prompt refinement).

### 1. Flagship (72B)
* **File:** `neolabs-chat72b.py`
* **Launcher:** `launch_chat72b.sh`
* **Description:** The most powerful version. Uses **Qwen2.5-72B-Instruct**, paired with Z-Image-Turbo.
* **Hardware Requirement:** **Enterprise Grade**
    * **Recommended:** NVIDIA H100 (80GB) or A100 (80GB).
    * **Minimum:** 2x NVIDIA A6000 (48GB) or a multi-GPU setup with at least 60GB+ combined VRAM.
    * *Note: The 72B model (even in 4-bit) is massive. Do not attempt to run this on consumer cards like a 4090.*

### 2. Balanced (14B)
* **File:** `neolabs-chat14b.py`
* **Launcher:** `launch_chat14b.sh`
* **Description:** The Goldilocks model. Uses **Qwen2.5-14B-Instruct**. Good logic and instruction following, but efficient enough for high-end consumer hardware.
* **Hardware Requirement:** **High-End Consumer**
    * **Recommended:** NVIDIA RTX 3090 / 4090 (24GB VRAM).
    * **Minimum:** NVIDIA A10G or any card with 24GB VRAM.

### 3. Lightweight (7B)
* **File:** `neolabs-chat7b.py`
* **Launcher:** `launch_chat7b.sh`
* **Description:** Optimized for speed and lower memory usage. Uses **Qwen2.5-7B-Instruct**. Good for quick questions.
* **Hardware Requirement:** **Mid-Range Consumer**
    * **Recommended:** NVIDIA RTX 4070 Ti / 3080 Ti (16GB VRAM).
    * **Minimum:** NVIDIA T4 (16GB) or RTX 3060 (12GB) *might* run it tight.

### 4. Z-Image Turbo (Standalone)
* **File:** `z-image.py`
* **Launcher:** `launch_z-image.sh`
* **Description:** An interface purely for image generation (no LLM). Very fast with surprisingly great quality and realism. (8-step generation).
* **Hardware Requirement:** **Entry-Level**
    * **Recommended:** Any GPU with 8GB+ VRAM (RTX 3060, 2070, etc.).
 
### 5. SDXL Turbo (Standalone)
* **File:** `sdxl-turbo.py`
* **Launcher:** `launch_sdxl-turbo.sh`
* **Description:** An interface purely for image generation (no LLM). Extremely fast (1-step generation). Not very realistic, but very performant and produces okay illustration-style images, especially paintings.
* **Hardware Requirement:** **Entry-Level**
    * **Recommended:** Any GPU, high end CPUs even work.

---

## Installation Details

The included `.sh` scripts handle installation automatically. However, if you prefer to install manually:

**Requirements:**
* Python 3.10+
* CUDA Toolkit (12.1+ recommended)

**Manual Install:**
```bash
pip install torch torchvision torchaudio gradio diffusers transformers bitsandbytes accelerate sentencepiece protobuf
```

---

## Common Bugs & Troubleshooting

If the application fails to start or crashes, check the list below:

### 1. `CUDA out of memory` (OOM)
* **Symptom:** The script crashes immediately after "Loading LLM" or during generation.
* **Cause:** Your GPU does not have enough VRAM for the selected model.
* **Fix:**
    * If trying to run **72B**, switch to **14B** or **7B**.
    * Close other GPU-intensive applications (games, other renders).
    * *Advanced:* Edit the python file and change `load_in_4bit=True` to `load_in_8bit=False` (though this usually *increases* memory usage, 4-bit is already the most efficient).

### 2. `bitsandbytes` errors or "DLL not found"
* **Symptom:** Error complaining about `libbitsandbytes` or CUDA setup.
* **Cause:** Usually happens on Windows or if CUDA paths are missing.
* **Fix:**
    * **Linux:** Ensure you have the CUDA toolkit installed (`nvcc --version`).
    * **Windows:** You may need to run `pip install bitsandbytes-windows` (unofficial) or ensure your NVIDIA drivers are up to date. *Note: These scripts are optimized for Linux/WSL.*

### 3. Download Timeouts / "Connection Reset"
* **Symptom:** The script hangs or errors while downloading files (especially for the 72B model).
* **Cause:** The 72B model is nearly 50GB. Weak internet connections may drop.
* **Fix:**
    * Restart the script; Hugging Face caching will resume where it left off.
    * Use `huggingface-cli download` to download the weights manually before running the script.

### 4. "Safetensors not found" or Corrupt Weights
* **Symptom:** `OSError: Error no file named...`
* **Fix:** Delete the cache folder at `~/.cache/huggingface/hub` and re-run the script to force a fresh download.

---

## Future Updates & Requests

We plan to add more models and variations in the future.

If you have a specific model or feature request, please open an issue in this repository or email me at jack__at__neocloudx.com
