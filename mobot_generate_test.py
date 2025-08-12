from pathlib import Path
import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# === Paths (Windows-safe) ===
ADAPTER_DIR = Path("../data/lora-Qwen2.5-7B")      # your saved LoRA adapter
CHROMA_DIR  = Path("../data/db_chroma")            # existing Chroma DB directory
COLL_NAME   = "facts"

# === Base model (open, no gating) ===
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# 4-bit load for inference if GPU is present
bnb_config = None
if device == "cuda":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

assert ADAPTER_DIR.exists(), f"Adapter folder not found: {ADAPTER_DIR.resolve()}"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16 if device=="cuda" else torch.float32,
    device_map="auto" if device=="cuda" else None,
    quantization_config=bnb_config,   # None on CPU
)
base = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
base.eval()
print("Model + LoRA ready.")
from chromadb import PersistentClient
from chromadb.utils import embedding_functions

# Must match the embedding model used when building the DB
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

client = PersistentClient(path=str(CHROMA_DIR))
rag_coll = client.get_collection(name=COLL_NAME, embedding_function=ef)
print("Chroma loaded. Docs:", rag_coll.count())

def retrieve_chroma(query: str, k: int = 3):
    res = rag_coll.query(query_texts=[query], n_results=k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    hits = []
    for i, doc in enumerate(docs):
        meta = metas[i] if i < len(metas) else {}
        if doc:
            hits.append({"text": doc, "meta": meta})
    return hits
import torch, json, re
from pathlib import Path
from typing import List, Dict, Optional
from transformers import TextStreamer

# ---- (A) Factual detection ----
FACTY_KEYWORDS = ["name", "email", "phone", "birthday", "wife", "husband", "cat", "dog", "city", "university"]
def is_factual_query(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in FACTY_KEYWORDS)

# ---- (B) Pinned facts (optional) ----
PROFILE_PATH = Path("./profile.json")
PROFILE = {}
if PROFILE_PATH.exists():
    try:
        PROFILE = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    except Exception:
        PROFILE = {}

def pinned_facts_block(profile: dict) -> str:
    if not profile: return ""
    lines = []
    for k, v in profile.items():
        if isinstance(v, list):
            v = ", ".join(v)
        lines.append(f"- {k}: {v}")
    return "PINNED FACTS:\n" + "\n".join(lines)

# Canonical strings to preserve exactly (light post-fix)
CANONICAL_STRINGS: List[str] = []
for key in ("cat_name", "wife_name", "husband_name", "email", "phone"):
    v = PROFILE.get(key)
    if isinstance(v, str) and v.strip():
        CANONICAL_STRINGS.append(v.strip())

def canonicalize(output: str) -> str:
    out = output.strip().strip("`")
    for canon in CANONICAL_STRINGS:
        if not canon: continue
        # if exact already present (case-insensitive), leave it
        if canon.lower() in out.lower():
            continue
        base = canon[:3]
        if not base: continue
        # squash silly repetitions like "TuaTuaTua" -> "Tuantuan"
        out = re.sub(re.escape(base) + r"{2,}", canon, out, flags=re.IGNORECASE)
    return out

# ---- (C) System prompts ----
SYS_PROMPT_GENERIC = (
    "You are Mimic‑Me Bot. Be concise, pragmatic, and use 'I' for the user's experience. "
    "Use information from PINNED FACTS and CONTEXT only; if uncertain, say you're unsure briefly."
)
SYS_PROMPT_FACT = (
    "You are Mimic‑Me Bot. Answer with ONLY the exact value requested, no extra words or punctuation. "
    "Use information from PINNED FACTS and CONTEXT only; if uncertain, answer: I'm not sure."
)

def build_messages(instruction: str, user_context: str = "", retrieved: Optional[List[Dict]] = None, factual: bool = False):
    ctx_lines = [h["text"] for h in (retrieved or []) if h.get("text")]
    ctx_block = ""
    if ctx_lines:
        ctx_block = "\n\nCONTEXT:\n" + "\n".join(f"- {t}" for t in ctx_lines)

    pin_block = pinned_facts_block(PROFILE)
    extra = ""
    if pin_block: extra += "\n\n" + pin_block
    if ctx_block: extra += ctx_block

    system_content = (SYS_PROMPT_FACT if factual else SYS_PROMPT_GENERIC) + (extra or "")
    user_content = instruction
    if factual:
        user_content += "\n\nFormat: output only the exact answer, no extra words."
    elif user_context:
        user_content += f"\n\nAdditional details:\n{user_context}"

    return [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": user_content},
    ]

# ---- (D) Get special token IDs for clean stopping ----
# Qwen uses ChatML: <|im_start|>role ... <|im_end|>
IM_END_ID = tokenizer.convert_tokens_to_ids("<|im_end|>") if "<|im_end|>" in tokenizer.get_vocab() else tokenizer.eos_token_id

@torch.inference_mode()
def mebot_generate(
    instruction: str,
    user_context: str = "",
    k: int = 3,
    use_rag: bool = True,
    max_new_tokens: int = 220,
    temperature: float = 0.7,
    top_p: float = 0.9,
    stream: bool = True
):
    factual = is_factual_query(instruction)
    retrieved = retrieve_chroma(instruction, k=k) if use_rag else []
    messages = build_messages(instruction, user_context, retrieved, factual=factual)

    # Build prompt using model's chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(base.device) for k, v in inputs.items()}
    input_len = int(inputs["input_ids"].shape[1])

    if factual:
        # Deterministic decode; small token budget; stop on im_end only.
        gen_kwargs = dict(
            max_new_tokens=min(16, max_new_tokens),
            do_sample=False,
            temperature=None,
            top_p=None,
            eos_token_id=IM_END_ID,
            repetition_penalty=1.05,
        )
        out = base.generate(**inputs, **gen_kwargs)
        gen_ids = out[0][input_len:]  # <-- decode ONLY the newly generated tokens
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        # Trim at first newline just in case
        text = text.split("\n", 1)[0].strip()
        return canonicalize(text)

    # Non-factual: allow sampling; support streaming cleanly
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True, temperature=temperature, top_p=top_p,
        eos_token_id=IM_END_ID,
        no_repeat_ngram_size=3,
        repetition_penalty=1.05,
    )

    if stream:
        # Stream only the generated part
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        base.generate(**inputs, streamer=streamer, **gen_kwargs)
        print()
        return ""
    else:
        out = base.generate(**inputs, **gen_kwargs)
        gen_ids = out[0][input_len:]  # <-- decode ONLY new tokens
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return text
    
mebot_generate("In one sentence, what is my PhD about?")
mebot_generate("What is MicroTune and why is it useful? 3 bullets.", use_rag=True)
mebot_generate("List the languages you speak.", use_rag=True)
mebot_generate("what's my cat's name", use_rag=True)