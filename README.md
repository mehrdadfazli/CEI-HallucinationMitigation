# 🧠 Context Embedding Injection (CEI)

This repository contains the official implementation of **Context Embedding Injection (CEI)**, a lightweight, training-free method for mitigating hallucinations in Large Vision–Language Models (LVLMs).

CEI extracts a stable **context embedding** from an initial forward pass over the image–prompt input and injects this signal into intermediate decoding layers during autoregressive generation. This intervention encourages the model to remain grounded in the original visual context, reducing object and attribute hallucinations without modifying model weights.

---

## ✨ Key Features

- **Training-free**: No fine-tuning or additional supervision required  
- **Model-agnostic**: Works with InstructBLIP, LLaVA-1.5, and LLaVA-NeXT  
- **Plug-and-play**: Integrated directly into the decoding loop  
- **Efficient**: Minimal runtime overhead (single extra forward pass)  
- **Benchmark-ready**: Supports CHAIR, AMBER, and MMHal-Bench  

---

## 🧩 Supported Models

CEI has been tested with the following 7B-scale LVLMs:

- InstructBLIP  
- LLaVA-1.5  
- LLaVA-NeXT  

Model-specific hyperparameters (e.g., injection layer, scaling schedule) are defined via JSON config files under `configs/`.

---

## 🗂️ Repository Structure

```text
cei_release/
├── requirements_lvlm.txt
├── configs/
│   ├── config_instructblip.json
│   ├── config_llava.json
│   ├── config_llavanext.json
│   └── README.md
├── eval/
│   ├── amber.py
│   ├── eval_gpt4.py
│   └── chair.py
├── run_chair.sh
├── run_mmhal.sh
├── run_amber.sh
└── src/
    ├── CEI_utils.py
    ├── model_utils.py
    ├── run_CHAIR.py
    ├── run_MMHal.py
    ├── run_AMBER.py
    └── __init__.py
```

## ⚙️ Installation

We recommend using a fresh virtual environment.

```
conda create -n cei python=3.10
conda activate cei
pip install -r requirements_lvlm.txt
```

## 🚀 Running Experiments
### CHAIR Benchmark
```
bash run_chair.sh path/to/config_file path/to/results_dir
```
or directly:
```
python src/run_CHAIR.py \
  --config configs/config_llavanext.json \
  --chair_data_path data/coco2014/val2014
```

### MMHal-Bench
```
bash run_mmhal.sh path/to/config_file path/to/results_dir
```
or:
```
python src/run_MMHal.py \
  --config configs/config_llava.json \
  --input data/MMHal-Bench/response_template.json \
  --images_root data/MMHal-Bench/images
```


### AMBER
```
bash run_amber.sh path/to/config_file path/to/results_dir
```
or:
```
python src/run_AMBER.py \
  --config configs/config_instructblip.json \
  --amber_path data/AMBER

```

## 📊 Evaluation

### CHAIR
You can evaluate CHAIR results via:

```bash
python eval/chair.py --coco_path /path/to/CHAIR/annotations --cap_file /path/to/CHAIR_output.jsonl
```

### AMBER
You can evaluate AMBER results via:
```bash
python eval/amber.py --inference_data /path/to/your/inference/file --evaluation_type g --gen_response_tag response_512
```

### MMHal-Bench
You can evaluate MMHal-Bench results via:
```
python eval/eval_gpt4.py --api-key <your-openai-api-key> --response results/MMHal-Bench/response_mymodel.json --evaluation results/MMHal-Bench/instructblip_GPT4_eval.json --gpt-model gpt-4.1-2025-04-14
```

