# GPT-From-Scratch-with-Alpaca-Instruction-Tuning

Implementation of a decoder-only GPT-style Transformer built entirely from scratch in PyTorch.  
The model is trained using next-token prediction and fine-tuned on Alpaca-format instruction-response pairs.

---

## 🚀 Project Overview

This project demonstrates a full implementation of a Large Language Model (LLM) from first principles, without using prebuilt transformer libraries.

Key highlights:

- Decoder-only Transformer architecture (GPT-style)
- Multi-Head Self-Attention with causal masking
- Token + positional embeddings
- Next-token prediction training objective
- Cross-entropy loss optimization
- Instruction fine-tuning on Alpaca-format dataset
- Autoregressive text generation

---

## 🧠 Architecture

The model follows the standard GPT-style decoder architecture:

Input Tokens  
→ Token Embeddings  
→ Positional Embeddings  
→ Stacked Transformer Blocks  
  • Multi-Head Self-Attention  
  • Causal Masking  
  • Feed Forward Network  
  • Layer Normalization  
→ Linear Projection  
→ Logits  
→ Softmax → Next Token Prediction  

---

## 📊 Training Details

- Context Window: 1024 tokens  
- Loss Function: Cross-Entropy  
- Optimizer: AdamW  
- Gradient Clipping: Enabled  
- Learning Rate Scheduler: Cosine Decay  
- Training Objective: Autoregressive Next-Token Prediction  

---

## 🔥 Instruction Fine-Tuning

The base language model was fine-tuned using supervised instruction tuning on 1100 Alpaca-format instruction-response pairs.

### Alpaca Format:
```json
{
  "instruction": "...",
  "input": "...",
  "output": "..."
}
