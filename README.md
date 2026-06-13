# 🏨 AI Medical Expert — Phi-2 Fine-Tuned

A medical Q&A assistant powered by **Microsoft Phi-2** fine-tuned on the
[Medical-o1 Reasoning SFT dataset](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)
using **Unsloth + LoRA** for 2× faster, memory-efficient training.

> ⚠️ **Disclaimer:** This is a research project. It is NOT a substitute for
> professional medical advice, diagnosis, or treatment.

---

## 📁 Project Structure

```
ai-doctor/
├── train.py          # Fine-tuning script (Unsloth + LoRA + SFTTrainer)
├── app.py            # Gradio web UI (streaming inference)
├── requirements.txt  # All Python dependencies
├── .gitignore        # Files excluded from version control
└── README.md         # This file
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/<your-username>/ai-doctor.git
cd ai-doctor

pip install -r requirements.txt
```

> **GPU required.** Training needs ~10 GB VRAM minimum (tested on T4 / A100).
> A free Kaggle or Google Colab GPU instance works well.

### 2. Fine-tune the Model

```bash
python train.py \
  --hf_token  YOUR_HF_WRITE_TOKEN \
  --max_steps 60 \
  --push_to_hub
```

| Argument        | Default                          | Description                            |
|-----------------|----------------------------------|----------------------------------------|
| `--hf_token`    | *(required)*                     | Hugging Face write-access token        |
| `--max_steps`   | `60`                             | Number of training steps               |
| `--push_to_hub` | `False`                          | Upload adapter weights to HF Hub       |
| `--save_local`  | `phi2_medical_lora`              | Local directory to save the model      |

### 3. Launch the Web UI

```bash
python app.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser.

---

## 🤗 Deploy on Hugging Face Spaces

1. Create a new **Space** (SDK: Gradio) at [huggingface.co/spaces](https://huggingface.co/spaces).
2. Push this repository to the Space:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
   git push space main
   ```
3. If your model repo is private, add `HF_TOKEN` as a **Secret** in the Space settings.

---

## 🧠 Model Details

| Property        | Value                                      |
|-----------------|--------------------------------------------|
| Base model      | `unsloth/phi-2` (Microsoft Phi-2)          |
| Fine-tuning     | LoRA (rank 16) via Unsloth                 |
| Dataset         | Medical-o1 Reasoning SFT (500 examples)   |
| Training steps  | 60                                         |
| Quantisation    | 4-bit (QLoRA)                              |
| HF model repo   | `priyanshuxsinha/phi2-medical-expert`      |

---

## 📊 Training Highlights

- **Unsloth** provides 2× faster training and ~60% less VRAM than vanilla HuggingFace.
- **Chain-of-Thought (CoT)** prompting teaches the model to reason step-by-step before answering.
- **Streaming** inference in the UI shows the answer token by token for a responsive feel.

---

## 👤 Author

**Priyanshu Sinha**  
ATRISI Intern · K-tech MeitY Nasscom CoE IoT & AI, Bengaluru

---

## 📄 License

This project is released under the **MIT License**.  
The base model (Phi-2) is subject to Microsoft's
[Research License](https://huggingface.co/microsoft/phi-2/blob/main/LICENSE).
