# HeadHunt-VAD: Hunting Robust Anomaly-Sensitive Heads in MLLM for Tuning-Free Video Anomaly Detection

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-green.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/get-started/locally/)

</div>

Official implementation of the paper **"HeadHunt-VAD: Hunting Robust Anomaly-Sensitive Heads in MLLM for Tuning-Free Video Anomaly Detection"**. 

---

## 📢 News
* **[2026.02]** Code released.
* **[2025.12]** HeadHunt-VAD is accepted by **AAAI 2026 Oral**! 🚀

---

## 📝 Abstract
Video Anomaly Detection (VAD) aims to identify and localize events that deviate from normal patterns. Traditional methods often require extensive labeled data or suffer from high computational costs. We propose **HeadHunt-VAD**, a novel tuning-free paradigm that bypasses textual generation by directly probing robust, anomaly-sensitive internal attention heads within a frozen Multimodal Large Language Model (MLLM). 

Central to our method is the **Robust Head Identification (RHI)** module, which evaluates attention heads through multi-criteria saliency and stability analysis across diverse prompts. Features from these expert heads are fed into a lightweight anomaly scorer and a calibrated temporal locator, achieving state-of-the-art performance on UCF-Crime and XD-Violence benchmarks with remarkable data and computational efficiency.

---

## 🖼️ Framework
![Framework Architecture](configs/resources/framework.png)
*The overall architecture of HeadHunt-VAD. The offline phase identifies consensus expert heads, while the online phase performs real-time detection via a single forward pass.*

---

## 🔧 Preparation

### 1. Environment
```bash
# Clone the repo
git clone https://github.com/headhunt-vad/headhunt-vad.git
cd headhunt-vad

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Data Structure
Organize your datasets as follows:
```
data/
├── UCF-Crime/
│   ├── Videos/
│   │   ├── Explosion/...
│   │   └── Normal_Videos_event/...
│   └── annotations.txt
└── XD-Violence/
    ├── videos/
    │   ├── v=..._label_A.mp4
    │   └── v=..._label_B1-0-0.mp4
    └── annotations.txt
```

---

## 🚀 Getting Started

### Step 1: Feature Extraction
Extract attention head outputs from the frozen MLLM (e.g., InternVL3-8B):
```bash
python scripts/extract_features.py \
    --model.path /path/to/internvl3-8b \
    --data.video_dir data/UCF-Crime/Videos \
    --output_dir ./features/ucf_coarse \
    --prompt "Identify any abnormal events in this video."
```

### Step 2: Robust Head Identification (RHI)
Identify expert heads that are stable across different prompts:
```bash
python scripts/run_rhi.py \
    --prompt_dirs coarse=./features/ucf_coarse detailed=./features/ucf_detailed \
    --output_dir ./results/rhi_ucf \
    --top_k 5 \
    --dataset ucf_crime
```

### Step 3: Anomaly Scorer Training
Train the lightweight Logistic Regression scorer using only ~1% of the calibration set:
```bash
python scripts/train_scorer.py \
    --data_dir ./features/ucf_coarse \
    --head_indices_file ./results/rhi_ucf/robust_head_indices.json \
    --dataset ucf_crime
```

### Step 4: Online Inference
Detect and localize anomalies in unseen videos:
```bash
python scripts/inference.py \
    --model_path /path/to/internvl3-8b \
    --scorer_path ./models/anomaly_scorer.joblib \
    --video_dir data/UCF-Crime/Test_Videos \
    --output_dir ./results/inference
```

---

## 📜 Citation
If you find this work useful for your research, please cite our paper:

```bibtex
@inproceedings{headhuntvad2026,
  title={HeadHunt-VAD: Hunting Robust Anomaly-Sensitive Heads in MLLM for Tuning-Free Video Anomaly Detection},
  author={Cai, Zhaolin and Li, Fan and Zheng, Ziwei and Bi, Haixia and He, Lijun},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

---

## 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
