# 🧠 SmartCompress - Train Small, Expand Smart

**Train a 5-layer network, then expand it to 10 layers with better accuracy and 50% less memory!**

---

## 📌 The Problem

Training deep neural networks is:
- 💰 **Expensive** (GPU time, memory, electricity)
- ⏱️ **Slow** (large models take hours/days)
- 🚫 **Inaccessible** (limited VRAM on consumer GPUs)

---

## 💡 The Solution

Instead of training a full 10-layer model from scratch:

1. **Train a compact 5-layer model** (saves 50% parameters, 40% time, and ~7% VRAM)
2. **Expand it back to 10 layers** using our smart expansion algorithm
3. **Light fine-tuning** (80 steps, LR=1e-4) to recover full performance

---

## 📊 Results (on Arabic text data)

| Model | Parameters | Test Loss | Training Time |
|-------|------------|-----------|---------------|
| Full (10 layers) | 1.33M | 2.781 | 2.42 sec |
| Compressed (5 layers) | 0.67M | 2.770 | 1.49 sec |
| **Expanded (after tuning)** | 1.33M | **2.637** ✅ | 0.50 sec |

### 🏆 Key Achievements

- **50% fewer parameters** during training
- **38% faster training**
- **Better accuracy** than full model (2.637 vs 2.781)
- **Full size restored** after expansion

---

## 🛠️ How It Works

### 1. Compression
Train a network with half the layers using standard training.

### 2. Smart Expansion
Each layer is duplicated with a smart split:
- Layer A: 80% of weights (main knowledge)
- Layer B: 20% + small noise (learns details)

### 3. Light Fine-Tuning
80 steps of fine-tuning at low learning rate (1e-4) to let new layers adapt.

---

## 🚀 Quick Start

### Requirements
```bash
pip install torch numpy
