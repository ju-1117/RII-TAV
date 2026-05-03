# RII-TAV: Cross-domain Recommendation Intent Identification in Live-streaming Sales

This repository contains the implementation of **RII-TAV**, a cross-domain recommendation intent identification method for live-streaming sales bullet chat streams.

RII-TAV is designed to identify whether a bullet chat expresses recommendation-oriented intent or casual interaction under cross-domain streaming scenarios. The method follows an offline preparation and online streaming workflow, including target-aligned source reconstruction, auto-prompt refinement, and dynamic verbalizer update.

---

## Environment Setup

All experiments were conducted on Ubuntu with Python 3.9, PyTorch 2.2.2, CUDA 12.1, and OpenPrompt v0.1.1. The hardware environment used in the paper includes an NVIDIA GeForce RTX 4090 Founders Edition GPU, an Intel Core i9-10980XE CPU, and 125 GB RAM.

We recommend creating the environment as follows:

```bash
conda create -n rii_tav python=3.9
conda activate rii_tav
```

Install PyTorch with CUDA 12.1 support:

```bash
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
```

Install OpenPrompt:

```bash
pip install openprompt==0.1.1
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Usage Steps

1. **Prepare the Reconstructed Source Training Data**

   In RII-TAV, the source-domain training data is reconstructed with target-oriented semantic guidance.

   The reconstructed training data is stored in the `LLM_prompt_train` directory.

   For example, using the RecDY setting, copy the training set  
   from:

   ```text
   LLM_prompt_train/train.csv
   ```

   to:

   ```text
   datasets/TextClassification/rec-dy/train.csv
   ```

   You can also use the following command:

   ```bash
   cp LLM_prompt_train/train.csv datasets/TextClassification/rec-dy/train.csv
   ```

2. **Prepare the Target Stream Data**

   The target-domain bullet chat stream is stored in the `all_data` directory.

   Copy the target test set  
   from:

   ```text
   all_data/test.csv
   ```

   to:

   ```text
   datasets/veb/rec-related/all_test.csv
   ```

   You can also use the following command:

   ```bash
   cp all_data/test.csv datasets/veb/rec-related/all_test.csv
   ```

3. **Run the Pipeline**

   After completing the above steps, simply execute:

   ```bash
   python autoautorunV2.2.py
   ```

   The script will automatically perform target-stream processing, candidate word extraction, zero-shot filtering, dynamic verbalizer update, and few-shot recommendation intent identification.

---


