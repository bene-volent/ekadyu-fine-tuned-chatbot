---
# Fine-Tuned Chatbot Service
A powerful conversational AI chatbot service built with FastAPI that provides text responses, audio generation, and specialized content creation features.
---
## **Features**

- 💬 Conversational AI: Natural dialogue with context awareness
- 🔊 Text-to-Speech: Stream audio responses for accessibility
- 📑 Content Generation: Create bullet points and presentation outlines
- 🔐 Authentication: JWT-based security with timestamp verification
- 🔄 Streaming Responses: Real-time token-by-token or chunked responses

---

## **Installation**

### Prerequisites

* Python 3.9+
* Git
* 7-Zip or WinRAR (for model extraction)

### Setup

1. Clone the repository:

2. Set up a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Download and extract the model:

   * Download the model files from [Folder](https://drive.google.com/drive/folders/1DPq3ULDF7jF8In1BuKHy9BfHLsgiaHul?usp=drive_link)
   * Extract the RAR file using WinRAR or 7-Zip
   * Place the extracted `deepseek-1.5B-finetuned` folder in the project root directory

5. Verify folder structure:

   ```
   fine-tuned-chatbot-test/
   ├── app.py
   ├── chatbot.py
   ├── auth/
   ├── deepseek-1.5B-finetuned/
   │   ├── model.safetensors
   │   ├── config.json
   │   └── ...
   ├── requirements.txt
   ```

---

## **Quick Start**

1. **Activate your virtual environment**:

   ```bash
   # On Windows
   venv\Scripts\activate

   # On Linux/Mac
   source venv/bin/activate
   ```

2. **Start the server**:

   ```bash
   python app.py
   ```

3. The server will start with default settings and initialize the model.

---

## **Hardware Requirements**

For optimal performance:

* 16GB+ RAM
* CUDA-compatible GPU with 8GB+ VRAM (recommended)
* SSD storage

---

## **License**

This project is available under the **MIT License**. See the `LICENSE` file for more information.

---

## **Acknowledgements**

* Built with **FastAPI** and **Hugging Face Transformers**
* Uses **DeepSeek-Coder 1.5B base model** with custom fine-tuning

---
