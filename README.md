# **Multimodal-Edge-Node**

Multimodal-Edge-Node is an experimental, node-based visual reasoning and multimodal inference canvas. It provides a unique, deeply customized web interface where users can visually connect input images, select from a diverse arsenal of 10 state-of-the-art vision-language models, define task parameters (such as query, caption, point, or detect), and observe real-time streaming outputs. The application goes beyond simple text generation by featuring a dedicated "Grounding Visualiser" node, which automatically parses JSON coordinate outputs from the models and renders precise bounding boxes or point markers directly onto the source image. Operating entirely on CUDA-enabled GPUs, this suite acts as a powerful, interactive sandbox for testing spatial grounding, optical character recognition, and instruction-following capabilities across various edge-optimized and unredacted models.

https://github.com/user-attachments/assets/fdbef908-1977-4765-b53e-664a2fdb289d

### **Key Features**

* **Interactive Node-Based Canvas:** Abandons standard UI layouts for a bespoke, draggable node interface with bezier curve wire connections, creating a visual flow from Image Input -> Model Selection -> Task Configuration -> Output & Grounding.
* **10x Vision Models:** Access a curated selection of advanced multimodal models, including the Qwen3-VL series, specialized unredacted variants, and LiquidAI's LFM models.
* **Real-Time Streaming Output:** Responses are generated and streamed token-by-token directly into the Output Stream node via FastAPI and Server-Sent Events (SSE).
* **Automatic Visual Grounding:** For "Point" and "Detect" tasks, the backend automatically parses the model's structured JSON output and renders an annotated image overlay (bounding boxes or point markers) in the Grounding node.
* **Custom Ubuntu/Dark Theme:** Features a highly styled, responsive dark theme utilizing the `JetBrains Mono` font for a sleek, developer-centric aesthetic.

### **Models Included**

The application supports the following vision-language models:

1.  **Qwen3-VL-2B-Instruct** (`Qwen/Qwen3-VL-2B-Instruct`)
2.  **Qwen3-VL-4B-Instruct** (`Qwen/Qwen3-VL-4B-Instruct`)
3.  **Qwen3.5-4B-Unredacted-MAX** (`prithivMLmods/Qwen3.5-4B-Unredacted-MAX`)
4.  **Qwen3.5-4B** (`Qwen/Qwen3.5-4B`)
5.  **Qwen3.5-2B** (`Qwen/Qwen3.5-2B`)
6.  **LFM2.5-VL-450M** (`LiquidAI/LFM2.5-VL-450M`)
7.  **Gemma4-E2B-it** (`google/gemma-4-E2B-it`)
8.  **LFM2.5-VL-1.6B** (`LiquidAI/LFM2.5-VL-1.6B`)
9.  **Qwen3.5-2B-Unredacted-MAX** (`prithivMLmods/Qwen3.5-2B-Unredacted-MAX`)
10. **Qwen2.5-VL-3B-Instruct** (`Qwen/Qwen2.5-VL-3B-Instruct`)

### **Repository Structure**

```text
Multimodal-Edge-Node/
├── .python-version
├── app.py
├── LICENSE.txt
├── main.py
├── pre-requirements.txt
├── pyproject.toml
├── README.md
├── requirements.txt
└── uv.lock
```

### **Installation and Requirements**

To run Multimodal-Edge-Node locally, you must configure a Python 3.14 environment with the following dependencies. A CUDA-enabled GPU is required to load and execute the models.

**Standard PIP Installation**
1. Update pip:
```bash
pip install pip>=26.1
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```

#### **Running with `uv` (Recommended)**

`uv` is an extremely fast Python package and project manager, written in Rust.

**Step 1 — Install `uv`**
*   **macOS / Linux:** `curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh`
*   **Windows:** `powershell -c "irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"`

**Step 2 — Clone the repository**
```bash
git clone https://github.com/PRITHIVSAKTHIUR/Multimodal-Edge-Node.git
cd Multimodal-Edge-Node
```

**Step 3 — Initialize the project and install dependencies**
Ensure you are using Python 3.14 as specified in the `.python-version` file.
```bash
uv sync
```

**Step 4 — Run the script**
```bash
uv run app.py
```

### **Usage**

After launching `app.py`, open your browser to the provided local address (typically `[http://127.0.0.1:7860/](http://127.0.0.1:7860/)`).
1. **Upload an Image:** Drag and drop an image into the "Input Image" node.
2. **Select a Model:** Choose one of the 10 available models from the "Model Selector" node.
3. **Configure Task:** In the "Task Config" node, select an action category (e.g., Query, Caption, Detect) and enter a specific prompt.
4. **Execute:** Click "Execute" to run the inference. The model's raw output will stream into the "Output Stream" node, and if grounding was requested, the annotated image will appear in the "View Grounding" node.

---

### **License and Source**

* **License:** Apache License 2.0 (Available at [LICENSE.txt](https://github.com/PRITHIVSAKTHIUR/Multimodal-Edge-Node/blob/main/LICENSE.txt))
* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/Multimodal-Edge-Node.git](https://github.com/PRITHIVSAKTHIUR/Multimodal-Edge-Node.git)
