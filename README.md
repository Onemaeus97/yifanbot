# Mimic-Me Bot

## 📌 Overview
**Mimic-Me Bot** is a personal AI assistant that:
- Mimics your **tone and personality** via LoRA fine-tuning on your own chat/email/post history.
- Recalls **facts and personal knowledge** via Retrieval-Augmented Generation (RAG) with a vector database.
- Runs **cheaply** on CPU or small GPU using a quantized open-source model.
- Has **hallucination safeguards** for accurate factual recall.

---

## 🚀 Features
- **Tone Mimicry** — Fine-tuned on your style dataset (`tone.jsonl`).
- **Knowledge Recall** — Retrieves personal facts from a Chroma vector DB (`facts.jsonl`).
- **Factual Mode** — Deterministic decoding for precision queries like “What’s my cat’s name?”
- **Canonicalization** — Corrects repeated/mutated named entities (e.g., `Tuantuantuan` → `Tuantuan`).
- **Low-Cost Deployment** — Runs on Hugging Face Spaces (CPU free tier) with LoRA adapter.

---

## 🧠 Model
| Component       | Choice |
|-----------------|--------|
| Base Model      | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |
| Fine-Tuning     | LoRA via `peft` + QLoRA (4-bit, `nf4`, `bfloat16` compute) |
| Vector Store    | [ChromaDB](https://www.trychroma.com/) |
| Embeddings      | `sentence-transformers/all-MiniLM-L6-v2` |
| Quantization    | 4-bit inference with `bitsandbytes` |
| Hosting         | Hugging Face Spaces (Gradio UI) |

---

## 📂 Data Format

### Style Data (`tone.jsonl`)
```json
{"instruction": "Explain your PhD topic.", "output": "I research DBMS tuning with AI to improve performance."}
{"instruction": "What's your favorite sport?", "output": "Basketball — I usually play as a guard."}
```

### Facts Data (`facts.jsonl`)
```json
{"text": "My cat's name is Tuantuan."}
{"text": "My wife is named Yuanyuan."}
```

---

## ⚙️ Fine-Tuning

```python
from peft import LoraConfig
from transformers import Trainer, TrainingArguments

lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
)

training_args = TrainingArguments(
    output_dir="./lora-Qwen2.5-7B",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    gradient_checkpointing=True,
    bf16=True
)
```

---

## 🔍 Retrieval-Augmented Generation (RAG)

**Pipeline:**
1. Embed `facts.jsonl` with `all-MiniLM-L6-v2`.
2. Store in Chroma vector DB.
3. At query time, retrieve top-k facts (`k=3`).
4. Inject into system prompt:
```
SYSTEM: You are Mimic-Me Bot...  
PINNED FACTS:  
- My cat's name is Tuantuan.  
- My wife is named Yuanyuan.  

CONTEXT:  
- <retrieved fact 1>  
- <retrieved fact 2>
```

---

## 🛡️ Hallucination Mitigation
- **Pinned Facts** — Always included, override retrieved data.
- **Factual Mode** — Keyword trigger (`name`, `cat`, `wife`, `email`…), disables sampling.
- **Canonicalization** — Regex cleanup for entity spelling consistency.
- **Strict Stop Sequences** — Ends output on `<|im_end|>` or start of next user turn.
- **Context-only Rule** — “If insufficient facts, say you’re unsure.”

---

## 💻 Inference Example
```python
print(mebot_generate("what's your wife's name", use_rag=True, stream=False))
# Yuanyuan

print(mebot_generate("summarize my PhD work", use_rag=True, stream=False))
# My PhD focuses on tuning database management systems using AI to improve performance.
```

---

## 📦 Deployment
1. Push model + LoRA adapter to Hugging Face Hub.
2. Push Chroma index (or rebuild on startup).
3. Deploy Gradio app:
```python
import gradio as gr
iface = gr.ChatInterface(fn=mebot_generate, title="Mimic-Me Bot")
iface.launch()
```

---

## 📅 Roadmap
- ✅ LoRA fine-tuning pipeline.
- ✅ Chroma RAG with post-processing.
- ✅ Hallucination safeguards.
- ⏳ Cross-encoder re-ranking for better retrieval precision.
- ⏳ Web UI toggles for “Precise Mode” & “Use RAG”.
