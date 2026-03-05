# ReCo

<p align="center">
    <img src="assets/title_imgs.png" width="400"/>
<p>

<p align="center">
    🖥️ <a href="https://github.com/HiDream-ai/ReCo">GitHub</a> &nbsp&nbsp ｜ &nbsp&nbsp  🌐 <a href="https://zhw-zhang.github.io/ReCo-page/"><b>Project Page</b></a> &nbsp&nbsp  | &nbsp&nbsp🤗 <a href="https://huggingface.co/datasets/HiDream-ai/ReCo-Data">ReCo-Data</a>&nbsp&nbsp | &nbsp&nbsp 📈 <a href="https://huggingface.co/datasets/HiDream-ai/ReCo-Bench">ReCo-Bench</a>&nbsp&nbsp | &nbsp&nbsp 🤗 <a href="https://huggingface.co/HiDream-ai/ReCo">ReCo-Models  </a> &nbsp&nbsp | &nbsp&nbsp 📖 <a href="https://arxiv.org/abs/2512.17650">Paper</a> &nbsp&nbsp 
<br>
 

[**ReCo: Region-Constraint In-Context Generation for Instructional Video Editing**](https://zhw-zhang.github.io/ReCo-page/) <be>

🔆 If you find ReCo useful, please give a ⭐ for this repo, which is important to Open-Source projects. Thanks!


Here, we will gradually release the following resources, including:

* **ReCo training dataset:** ReCo-Data
* **Evaluation code:** ReCo-Bench
* **Model weights, inference code, and training code**

 
## Video Demos

<div align="center">
  <video controls autoplay loop muted playsinline src="https://github.com/user-attachments/assets/ba530e6f-e13b-4d04-ad60-b95277cc38ce"></video>
  <p><em>Examples of different video editing tasks by our ReCo.</em></p>
</div>

## 📢 News!!!
- 2026.03.05: We are excited to see that [Kiwi-Edit (NUS)](https://github.com/showlab/Kiwi-Edit/tree/main) has further refined our **HQ-ReCo dataset** and added reference image pairs. Check out their [DATASET.md](https://github.com/showlab/Kiwi-Edit/blob/main/DATASET.md) for further instructions.

## 🔥 Updates
✅ **\[2025.12.22\]** Upload Our arXiv Paper.
✅ **\[2025.12.23\]** Release ReCo-Data and Usage code.
✅ **\[2025.12.23\]** Release ReCo-Bench and evaluation code.
✅ **\[2026.01.16\]** Release ReCo Model weights and inference code.
✅ **\[2026.01.16\]** Uploaded raw [video object masks](https://huggingface.co/datasets/HiDream-ai/ReCo-Data/tree/main/video_masks) to ReCo-Data.
✅ **\[2026.02.26\]** Release training code.




## 📊 ReCo-Data Preparation

**ReCo-Data** is a large-scale, high-quality video editing dataset consisting of **500K+ instruction–video pairs**, covering four video editing tasks: **object addition (add)**, **object removal (remove)**, **object replacement (replace)**, and **video stylization (style)**.

<p align="center">
    <img src="assets/statistic.png" width="800"/>
<p>

### Downloading ReCo-Data

Please download each task of ReCo-Data into the `./ReCo-Data` directory by running:

```bash
bash ./tools/download_dataset.sh
````

Before downloading the full dataset, you may first browse the
**[visualization examples](https://huggingface.co/datasets/HiDream-ai/ReCo-Data/blob/main/examples.tar)**.

These examples are collected by **randomly sampling 50 instances from each task**
(add, remove, replace, and style), **without any manual curation or cherry-picking**,
and are intended to help users quickly inspect and assess the overall data quality.

Note: The examples are formatted for visualization convenience and do not strictly follow the dataset format.

### Directory Structure

After downloading, please ensure that the dataset follows the directory structure below:

<details open>
<summary>ReCo-Data directory structure</summary>

```text
ReCo-Data/
├── add/
│   ├── add_data_configs.json
│   ├── src_videos/
│   │   ├── video1.mp4
│   │   ├── video2.mp4
│   │   └── ...
│   └── tar_videos/
│       ├── video1.mp4
│       ├── video2.mp4
│       └── ...
├── remove/
│   ├── remove_data_configs.json
│   ├── src_videos/
│   └── tar_videos/
├── replace/
│   ├── replace_data_configs.json
│   ├── src_videos/
│   └── tar_videos/
└── style/
    ├── style_data_configs.json
    ├── src_videos/
    │   ├── video1.mp4
    │   └── ...
    └── tar_videos/
        ├── video1-a_Van_Gogh_style.mp4
        └── ...
```

</details>

### Testing and Visualization

After downloading the dataset, you can directly test and visualize samples from **any single task** using the following script
(taking the **replace** task as an example):

```bash
python reco_data_test_single.py \
  --json_path ./ReCo-Data/replace/replace_data_configs.json \
  --video_folder ./ReCo-Data \
  --debug
```

### Mixed Task Loading

You can also load a **mixed dataset** composed of the four tasks (**add**, **remove**, **replace**, and **style**) with arbitrary ratios by running:

```bash
python reco_data_test_mix_data.py \
  --json_folder ./ReCo-Data \
  --video_folder ./ReCo-Data \
  --debug
```

### Notes

* `src_videos/` contains the original source videos.
* `tar_videos/` contains the edited target videos corresponding to each instruction.
* `*_data_configs.json` stores the instruction–video mappings and metadata for each task.



## 📈 Evaluation

### VLLM-based Evaluation Benchmark
<details close>
<summary>ReCo-Bench details</summary>

Traditional video generation metrics often struggle to accurately assess the fidelity and quality of video editing results. Inspired by recent image editing evaluation protocols, we propose a **VLLM-based evaluation benchmark** to comprehensively and effectively evaluate video editing quality.


We collect **480 video–instruction pairs** as the evaluation set, evenly distributed across four tasks: **object addition**, **object removal**, **object replacement**, and **video stylization** (120 pairs per task). All source videos are collected from the **Pexels** video platform.


For local editing tasks (add, remove, and replace), we utilize **Gemini-2.5-Flash-Thinking** to automatically generate diverse editing instructions conditioned on video content. For video stylization, we randomly select **10 source videos** and apply **12 distinct styles** to each, resulting in **120 stylization evaluation pairs**.
</details>

---

### Downloading ReCo-Bench
Please download **ReCo-Bench** into the `./ReCo-Bench` directory by running:
```bash
bash ./tools/download_ReCo-Bench.sh
````



---



### Usage



After downloading the benchmark, you can directly start the evaluation using:
```bash
cd tools
bash run_eval_via_gemini.sh
```


<details close>
<summary>This script performs the evaluation in two stages:</summary>

#### Step 1: Per-dimension Evaluation with Gemini
In the first stage, **Gemini-2.5-Flash-Thinking** is used as a VLLM evaluator to score each edited video across multiple evaluation dimensions.

Key arguments used in this step include:
* `--edited_video_folder`: Path to the folder containing the edited (target) videos generated by the model.

* `--src_video_folder`: Path to the folder containing the original source videos.

* `--base_txt_folder`: Path to the folder containing task-specific instruction configuration files.

* `--task_name`: Name of the evaluation task, one of `{add, remove, replace, style}`.



This step outputs per-video, per-dimension evaluation results in JSON format.

#### Step 2: Final Score Aggregation

After all four tasks have been fully evaluated, the second stage aggregates the evaluation results and computes the final scores.
* `--json_folder`: Path to the JSON output folder generated in Step 1

  (default: `all_results/gemini_results`)

* `--base_txt_folder`: Path to the instruction configuration folder

This step produces the final benchmark scores for each task as well as the overall performance. 


</details>



## 🏃 Inference

### 1. Environment Preparation

Create and activate the specialized Conda environment:

```bash
conda create -n reco python=3.11 -y
conda activate reco
pip install -r requirements.txt
```

### 2. Model Weights Setup

You need to prepare both the base model and our specific checkpoints.

<!-- | Model | Source | Description |
| --- | --- | --- |
| **VACE 1.3B** | [🤗 Hugging Face](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B) | Base VACE weights (Place in `./Wan-AI`) |
| **ReCo** | [🤗 Hugging Face](https://huggingface.co/HiDream-ai/ReCo) | Our ReCo checkpoint(Place in `all_ckpts/`). We will update better ckpts progressively afterward | -->

<table>
  <thead>
    <tr>
      <th width="25%" align="center">Model</th>
      <th width="25%" align="center">Source</th>
      <th align="left">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><b>Wan-2.1-VACE-1.3B</b></td>
      <td align="center"><a href="https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B">🤗 Hugging Face</a></td>
      <td>Base VACE weights. Place in <code>./Wan-AI</code></td>
    </tr>
    <tr>
      <td align="center"><b>ReCo</b></td>
      <td align="center"><a href="https://huggingface.co/HiDream-ai/ReCo">🤗 Hugging Face</a></td>
      <td>Our ReCo checkpoint. Place in <code>all_ckpts/</code>.</td>
    </tr>
  </tbody>
</table>

**Organize the files as follows:**
```text
.
├── Wan-AI/                      
├── all_ckpts/                   
│   └── 2026_01_16_v1_release.ckpt  
├── assets/                      
└── inference_reco_single.py
```

### 3. Running Inference

We provide a bash script to automate the execution of different tasks (Replace, Remove, Style, Add and Propagation). Run the following command:

```bash
bash infer_server_single.sh
```

To run a specific task manually or customize the execution, use the python command directly:

```bash
python inference_reco_single.py \
    --task_name replace \
    --test_txt_file_name assets/replace_test.txt \
    --lora_ckpt all_ckpts/2026_01_16_v1_release.ckpt
```

### 4. Key Arguments Explained

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `test_txt_file_name` | `str` | `assets/...` | Path to the `.txt` file containing test prompts/configs. |
| `task_name` | `str` | `replace` | Task type: `remove`, `replace`, `add`, `style`. Use the `_wf` suffix (e.g., `remove_wf`) for **Propagation tasks** given the first frame. |
| `base_video_folder` | `str` | `assets/test_videos` | Directory containing the source videos. |
| `base_wan_folder` | `str` | `./Wan-AI` | Path to the pre-trained Wan-AI model weights. |
| `lora_ckpt` | `str` | `all_ckpts/...` | Path to the specific LoRA checkpoint file. |



## 🚀 Training

To start training, run:

```bash
bash train.sh
```

### ⚠️ Important Notes

* Make sure to update the **pretrained model weight paths** in the script to match your local environment.
* In `train.py`, modify the dataset paths inside
  `LightningModelForTrain.train_dataloader`:

  * Update the **JSON annotation directory**
  * Update the **video data directory**

Ensure these paths point to your local dataset before launching training.


## 🌟 Star and Citation
If you find our work helpful for your research, please consider giving a star⭐ on this repository and citing our work.
```
@article{reco,
	title={{Region-Constraint In-Context Generation for Instructional Video Editing}},
	author={Zhongwei Zhang and Fuchen Long and Wei Li and Zhaofan Qiu and Wu Liu and Ting Yao and Tao Mei},
	journal={arXiv preprint arXiv:2512.17650},
	year={2025}
}
```


## 💖 Acknowledgement
<span id="acknowledgement"></span>

Our code is inspired by several works, including [WAN](https://github.com/Wan-Video/Wan2.1), [ObjectClear](https://github.com/zjx0101/ObjectClear)--a strong object remover, [VACE](https://github.com/ali-vilab/VACE), [Flux-Kontext-dev](https://github.com/black-forest-labs/flux). Thanks to all the contributors! 


