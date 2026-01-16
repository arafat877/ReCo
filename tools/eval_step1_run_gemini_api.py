import os
import base64
import threading
import concurrent.futures
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import decord
import numpy as np
import argparse
import json
import sys
from openai import OpenAI
import logging 
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)


benchmark_score_eval_sys_prompt = """
You are a professional digital artist and video quality evaluator. Your task is to evaluate an AI-generated video edit based on three major categories: **Edit Accuracy**, **Video Quality**, and **Naturalness**.
You will be given the text instruction used to create the edit and side-by-side video keyframes, where the left side shows the original video and the right side shows the edited version.
You must provide your output *only* in the following JSON format. Do not output anything else.

{
  "edit_accuracy": {
    "scores": [1, 1, 1],
    "reasoning": "..."
  },
  "video_quality": {
    "scores": [1, 1, 1],
    "reasoning": "..."
  },
  "naturalness": {
    "scores": [1, 1, 1],
    "reasoning": "..."
  }
}

(Keep each reasoning string concise and short, summarizing the scores for that category.)

---

## Category 1: Edit Accuracy

This category evaluates how well the AI understood and executed the *text instruction*.
The `scores` list for `edit_accuracy` contains three scores: **[Score_1A, Score_1B, Score_1C]**.

### Score 1A: Semantic Accuracy (Scale: 1-10)
* **What it is:** Rates if the *core concept* of the edit is correct (e.g., *what* was added, removed, replaced, or stylized).
* **1:** The core concept is completely wrong (e.g., instruction was "add a dog", but it added a cat; or "stylize as sketch", but it applied a "pixel art" style).
* **10:** The core concept of the edit perfectly matches the instruction.

### Score 1B: Scope Precision (Scale: 1-10)
* **What it is:** Rates if the *location, area, or scope* of the edit is correct (e.g., *where* the edit was applied).
* **1:** The location/area is completely wrong (e.g., edited the background instead of the instructed foreground object; or applied a local edit when a global one was requested).
* **10:** The edit is perfectly localized or globalized, exactly as instructed (e.g., *only* the specified hat was replaced; the *entire* scene was correctly stylized).

### Score 1C: Content Preservation (Scale: 1-10)
* **What it is:** Rates if the AI negatively affected areas that should *not* have been edited.
* **1:** Unedited areas are heavily distorted, changed, blurred, or contain new artifacts, losing the original content.
* **10:** All content outside the specified edit scope is perfectly preserved and identical to the original.
* **Note:** For global stylization, this evaluates if the *underlying structure* (objects, motion) is preserved.

---

## Category 2: Video Quality

This category evaluates the *technical fidelity and stability* of the edited video (the right side), focusing on artifacts and temporal consistency.
The `scores` list for `video_quality` contains three scores: **[Score_2A, Score_2B, Score_2C]**.

### Score 2A: Visual Fidelity (Scale: 1-10)
* **What it is:** Rates the overall clarity and presence of *static visual artifacts* (e.g., blur, distortion, "melting" objects) in the edited frames.
* **1:** The video is extremely blurry, full of artifacts, or heavy distortions.
* **10:** The video is sharp, clear, and free of any unnatural visual artifacts.

### Score 2B: Temporal Stability (Pixel-level) (Scale: 1-10)
* **What it is:** Rates the low-level consistency of the video *over time*, focusing on flicker, boiling, or popping textures.
* **1:** The video is extremely unstable. Edits or styles flicker erratically, or textures "boil" constantly between frames.
* **10:** The video is perfectly stable over time. All edits and textures are consistent from one frame to the next.

### Score 2C: Edit Effect Persistence (Semantic-level) (Scale: 1-10)
* **What it is:** Rates if the *intended edit effect* (add, remove, replace, style) is stable and persists correctly from the beginning to the end of the video.
* **1:** The edit effect fails mid-video. The edit breaks, disappears, or reverts.
* **10:** The intended edit effect is perfectly stable and consistent throughout the entire video.
* **Bad Cases:** "A removed object 'pops back' into view." "An added object 'disappears' halfway through." "A replaced object 'reverts' to the original." "A stylization effect 'stops working' after a few seconds."

---

## Category 3: Naturalness

This category evaluates how *plausible and seamlessly integrated* the edit is within the scene's context, physics, and lighting.
The `scores` list for `naturalness` contains three scores: **[Score_3A, Score_3B, Score_3C]**.

### Score 3A: Appearance Naturalness (Integration) (Scale: 1-10)
* **What it is:** Rates how *naturally* the new or edited parts blend with the original scene's lighting, shadows, reflections, and texture.
* **1:** The edit looks completely fake and "pasted on". It clashes with the scene's lighting, casts no or incorrect shadows, and boundaries are harsh.
* **10:** The edit is perfectly integrated. It looks completely natural, matches the scene's lighting, and blends flawlessly.

### Score 3B: Scale & Proportion (Scale: 1-10)
* **What it is:** Rates if the edited object's size is reasonable and proportional to the scene.
* **1:** The object's scale is completely illogical and breaks the scene's realism.
* **10:** The edited object's size is perfectly proportional and natural within the scene.
* **Bad Cases:** "Added a cat in the living room that is 'as large as the sofa'." "After removing an object, the inpainted background texture (like floor tiles) is 'magnified' several times, appearing disproportionate." "Replaced a 'car' with a 'motorcycle', but the motorcycle is 'huge' and fills the entire lane." "Stylization (e.g., 'anime') causes the character's 'head' to become abnormally large, beyond the style's reasonable scope."

### Score 3C: Motion Naturalness (Physical Laws) (Scale: 1-10)
* **What it is:** Rates if the edit and its motion obey basic physical laws (e.g., gravity, rigidity) and interact logically with the scene.
* **1:** The edit blatantly violates physics (e.g., objects fall up, solids pass through each other) or interacts nonsensically.
* **10:** The edit's behavior is physically plausible and interacts naturally with its environment.
* **Bad Cases:** "Instructed to 'add a balloon', but the balloon 'falls straight' to the ground (instead of floating)." "Instructed to 'add a hat on the dog's head', but the hat floats in mid-air and doesn't track the dog's movement." "Removed a 'pillar', but the 'roof' it was supporting remains suspended in mid-air, defying gravity." "Replaced 'water' with 'lava', but the lava flows calmly 'like water'."

---

## Critical Rule: Failed Edits (Identical Videos)

If the edited video (right side) is identical to the original video (left side), this indicates a total failure (the edit did not apply).
You must set all nine scores to 0.

{
  "edit_accuracy": {
    "scores": [0, 0, 0],
    "reasoning": "Edit failed to apply. The edited video is identical to the original."
  },
  "video_quality": {
    "scores": [0, 0, 0],
    "reasoning": "Edit failed to apply. The edited video is identical to the original."
  },
  "naturalness": {
    "scores": [0, 0, 0],
    "reasoning": "Edit failed to apply. The edited video is identical to the original."
  }
}
"""


from typing import Optional, Tuple, Any

def extract_braced_json(s: str) -> Optional[str]:
    """从 s 中截取从第一个 { 到匹配的最后一个 } 的完整片段。
    处理双引号字符串与转义字符，避免引号内的大括号干扰。
    """
    start = s.find('{')
    if start == -1:
        return None

    depth = 0
    in_str = False   # 是否在双引号字符串中
    esc = False      # 是否刚遇到反斜杠转义
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
    return None  # 没闭合


class OpenAIVLMEngine:
    def __init__(self, model_name="gpt-4o", max_tokens=8192, custom_base_url=None):
        """
        Initialize the OpenAI Vision Language Model Engine.
        
        Args:
            model_name (str): Name of the OpenAI model to use (default: gpt-4o)
            max_tokens (int): Maximum number of tokens to generate
            custom_base_url (str, optional): Custom base URL for API
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        
        # Initialize OpenAI client
        client_params = {}
        if custom_base_url:
            client_params["base_url"] = custom_base_url
        self.client = OpenAI(**client_params)
        self.write_lock = threading.Lock()

    def encode_image_to_base64(self, image_path):
        """
        Reads an image file, resizes it ensuring min side is 512px,
        converts to JPEG, and returns the data URI.
        """
        try:
            # Open and resize the image
            img = Image.open(image_path).convert("RGB")
            
            # Calculate new dimensions ensuring min side is 512
            width, height = img.size
            scale = max(512 / min(width, height), 1.0)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to JPEG in memory to save bandwidth
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            buffer.seek(0)
            
            # Encode to base64
            encoded_string = base64.b64encode(buffer.read()).decode('utf-8')
            return f"data:image/jpeg;base64,{encoded_string}"
            
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            return None
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None

    def process_single_item(self, item):
        """
        Process a single item with OpenAI API.
        
        Args:
            item (dict): Dict containing "image_path" and "question"
            
        Returns:
            str: The model's response
        """
        question = item['question']
        image_paths = item['image_path']
        
        # Handle single or multiple images
        if not isinstance(image_paths, list):
            image_paths = [image_paths]
        
        # Encode images to base64
        image_contents = []
        for img_path in image_paths:
            try:
                base64_image = self.encode_image_to_base64(img_path)
                if base64_image:
                    image_contents.append(
                        {"type": "image_url", "image_url": {"url": base64_image}}
                    )
                else:
                    print(f"Warning: Could not encode image {img_path}")
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
        
        # Add the question as text content
        message_content = [
            *image_contents,  # Unpack the list of image dictionaries
            {"type": "text", "text": question}
        ]
        
        # Call OpenAI API with retries
        max_retries = 10
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": message_content}
                    ],
                    max_tokens=self.max_tokens,
                )
                response = completion.choices[0].message.content
                return response
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"API call failed (attempt {attempt+1}/{max_retries}): {e}. Retrying...")
                else:
                    print(f"API call failed after {max_retries} attempts: {e}")
                    return f"Error: API call failed - {e}"

    def process_image_query(self, data, num_workers=8):
        """
        Process images and questions through OpenAI API in parallel.
        
        Args:
            data (list): A list of dicts, each dict contains "image_path" and "question"
            num_workers (int): Number of concurrent API calls
            
        Returns:
            list: The model's responses
        """
        total_items = len(data)
        responses = [None] * total_items
        
        print(f"Starting OpenAI API calls with {num_workers} workers...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Create futures for each item
            future_to_idx = {
                executor.submit(self.process_single_item, item): idx
                for idx, item in enumerate(data)
            }
            
            # Process completed futures with a progress bar
            for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=total_items, desc="Generating responses"):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    with self.write_lock:
                        responses[idx] = result
                except Exception as e:
                    print(f"Error processing item {idx}: {e}")
                    responses[idx] = f"Error: {str(e)}"
                    
        return responses


def get_base_video_name(tar_video_path, all_src_videos):

    for base_name in all_src_videos:
        base_video_name = base_name.replace('.mp4', '')
        if base_video_name in tar_video_path:
            break

    assert base_video_name in tar_video_path, 'error, must can find a source video'

    return base_video_name+'.mp4'


def get_videos_from_path(video_src_path, video_tar_path, resolution_h=224, out_folder=None):
    """
    读取 tar 文件夹下的每个 .mp4，与匹配到的 src 视频逐帧按宽度拼接，
    将拼接后的帧按指定高度缩放（等比），以 frame_00.png 等命名保存到子文件夹中。

    返回：
      - all_concate_videos_folder: list[str]，所有输出帧所在的文件夹路径
    """
    # 1) 读取目标视频 (tar)
    vr_tar = decord.VideoReader(video_tar_path)
    tar_len = len(vr_tar)
    # 直接批量拉取所有帧到 numpy
    video_tar = vr_tar.get_batch(list(range(tar_len))).asnumpy()  # [T, H, W, C]

    vr_src = decord.VideoReader(video_src_path)
    src_len = len(vr_src)
    video_src = vr_src.get_batch(list(range(src_len))).asnumpy()  # [T, H, W, C]

    # 5) 沿宽度拼接
    # 形状 [T, H, W1+W2, C]
    video_concat = np.concatenate([video_src, video_tar], axis=2)

    # 6) 最终按参数 resolution_h 进行等比缩放
    H_final = resolution_h
    H_curr, W_curr = video_concat.shape[1], video_concat.shape[2]
    if H_curr != H_final:
        scale = H_final / float(H_curr)
        W_final = int(round(W_curr * scale))
        resized_frames = []
        for fr in video_concat:
            img = Image.fromarray(fr)
            img = img.resize((W_final, H_final), Image.BICUBIC)
            resized_frames.append(np.array(img))
        video_concat = np.stack(resized_frames, axis=0)

    # 7) 保存帧：输出文件夹形如 ".../xxx_reso_244"
    os.makedirs(out_folder, exist_ok=True)

    saved_paths = []
    for i, fr in enumerate(video_concat):
        out_path = os.path.join(out_folder, f'frame_{i:03d}.jpg')
        if os.path.exists(out_path):
            continue
        Image.fromarray(fr).save(out_path)
        saved_paths.append(out_path)

    concated_frame_folders = out_folder

    return concated_frame_folders

import os
import subprocess
from pathlib import Path
from typing import Literal

def hstack_and_dump_frames_ffmpeg(
    video_src_path: str,
    video_tar_path: str,
    out_height: int = 224,
    out_folder=None,
    fmt: Literal["jpg","png"] = "jpg",
    quality: int = 3,  # jpg: 2(高质)~31(低)；png: 0(快/大)~9(慢/小)
    use_cuda: bool = False
) -> str:
    """
    用 FFmpeg 把两段视频按高度 out_height 等比缩放到同高，然后左右拼接，导出逐帧图片。
    返回输出目录路径。
    """

    out_dir = Path(out_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 让两路视频都 scale 到统一高度，再 hstack
    # 注意：scale= -2:HEIGHT 代表宽度按比例取最接近的偶数
    fc = (
        f"[0:v]scale=-2:{out_height}:flags=lanczos[left];"
        f"[1:v]scale=-2:{out_height}:flags=lanczos[right];"
        f"[left][right]hstack=inputs=2[out]"
    )

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if use_cuda:
        cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]

    cmd += [
        "-i", video_src_path,
        "-i", video_tar_path,
        "-filter_complex", fc,
        "-map", "[out]",
        "-vsync", "0",
        "-frame_pts", "1",
    ]

    pattern = "%06d." + fmt
    out_pattern = str(out_dir / pattern)

    if fmt == "jpg":
        cmd += ["-q:v", str(quality)]
    elif fmt == "png":
        cmd += ["-compression_level", str(quality)]
    else:
        raise ValueError("fmt must be 'jpg' or 'png'")

    cmd += [out_pattern]

    subprocess.run(cmd, check=True)
    return str(out_dir)


def load_output_as_json(response):

    if 'error' in response.lower():
        json_return = {"score": "error"}

    else:
        json_prompt = extract_braced_json(response)
        json_return = json.loads(json_prompt)

    return json_return


from typing import List, Dict, Optional

def parse_instruction_file(
    file_path: str,
    encoding: str = "utf-8",
    base_dir_for_ip: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    读取形如：
        855029-hd_1920_1080_30fps.mp4: Add ... | asserts/ip_images/clean_ip/rabbit_2.png
    的文本文件并解析为字典列表：
        [{"src_video_path": ..., "instructed_prompt": ..., "ip_path": ...}, ...]
    
    规则：
    - 允许行首以 # 开头作为注释，或空行，均跳过
    - 仅使用第一处冒号分割出 video 与其余部分
    - 使用 ' | '（两侧可有可无多余空格）分割出 prompt 与 ip_path
    - 若缺少 ip_path，则置为 ""（空字符串）
    - 若提供 base_dir_for_ip，则把 ip_path 用该目录拼成绝对/规范路径
    """
    results: List[Dict[str, str]] = []
    with open(file_path, "r", encoding=encoding) as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            # 1) 拆出 video 与其余部分（只按第一个冒号切）
            if ":" not in line:
                raise ValueError(f"[line {lineno}] 格式错误：缺少冒号 ':' —— {raw!r}")
            video, rest = line.split(":", 1)
            video = video.strip()
            rest = rest.strip()

            if not video:
                raise ValueError(f"[line {lineno}] 格式错误：src_video_path 为空 —— {raw!r}")

            # 2) 拆出 prompt 与 ip（'|' 可选）
            ip_path = None
            if "|" in rest:
                prompt, ip = rest.split("|", 1)
                prompt = prompt.strip()
                ip_path = ip.strip()
            else:
                prompt = rest.strip()  # 允许没有 ip 的行

            if not prompt:
                raise ValueError(f"[line {lineno}] 格式错误：instructed_prompt 为空 —— {raw!r}")

            # 3) 规范化 ip_path（可选）
            if base_dir_for_ip and ip_path:
                import os
                ip_path = os.path.normpath(os.path.join(base_dir_for_ip, ip_path))

            results.append({
                "src_video_path": video,
                "instructed_prompt": prompt,
                "ip_path": ip_path,
            })

    return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run video editing script.")
    parser.add_argument("--edited_video_folder", type=str, default="all_results/replace")
    parser.add_argument("--src_video_folder", type=str, default="ori_videos")
    parser.add_argument("--base_txt_folder", type=str, default="configs")
    parser.add_argument("--task_name", type=str, default='replace')
    
    args = parser.parse_args()

    # 0. Define engine
    os.environ["OPENAI_API_KEY"] = "sk-YOUR_API_TOKENS"
    model_name = 'gemini-2.5-flash-thinking'
    custom_base_url = "https://api.nuwaapi.com/v1"       # cheap api ref: https://api.nuwaapi.com/
    engine = OpenAIVLMEngine(model_name=model_name, max_tokens=8192, custom_base_url=custom_base_url)

    # 1. define eval_name and get src_configs
    sys_prompt = benchmark_score_eval_sys_prompt
    base_txt_folder = args.base_txt_folder
    if args.task_name == "remove":
        file_name = os.path.join(base_txt_folder, 'remove.txt')
    elif args.task_name == "replace":
        file_name = os.path.join(base_txt_folder, 'replace.txt')
    elif args.task_name == "add":
        file_name = os.path.join(base_txt_folder, 'add.txt')
    elif args.task_name == "style":
        file_name = os.path.join(base_txt_folder, 'style.txt')
    else:
        NotImplementedError

    all_video_list = parse_instruction_file(file_name)

    # 2. define output json path
    json_path = os.path.join(os.path.dirname(args.edited_video_folder), 'gemini_results', f'{args.task_name}_vllm_gemini.json')
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    if os.path.exists(json_path):
        print(f"[abort] Output exists: {json_path}\n"
            f"        Use --force to overwrite.", file=sys.stderr)
        sys.exit(1)


    # ================== 3. START EVAL ============
    use_ffmpeg = False
    fps = 2
    stride = 81//(fps*5)        # 8fps
    counts = 0
    data_list = []
    fail_video_dict = {}
    all_dict_list = []
    for video_dict in tqdm(all_video_list, total=len(all_video_list)):
    
        # *** NOTE: The video saving format must follow the naming convention as below. ***
        prompt = video_dict['instructed_prompt']
        video_base_name = video_dict['src_video_path']
        prompt_name = prompt.replace(' ', '_').replace('.', '').replace(',','').replace(':',' ')
        vide_save_name = video_base_name.replace('.mp4', '')
        video_save_name = f'{vide_save_name}_{prompt_name[:80]}.mp4'       # ! NOTE: VIDEO SAVE NAME

        try:
            tar_video_path = os.path.join(args.edited_video_folder, video_save_name)
            src_video_path = os.path.join(args.src_video_folder, video_base_name)

            # get concated resized input videos.. FFMPEG can be faster..
            resolution_h = 224
            concated_frames_folders = os.path.join(args.edited_video_folder, 'video_frames', os.path.basename(tar_video_path).replace('.mp4', '')) + f'_reso_{resolution_h:03d}'
            if use_ffmpeg:
                # sudo apt-get update
                # sudo apt-get install ffmpeg
                print(f"Processing video pair using ffmpeg: {video_save_name}")
                concated_frames_folders = hstack_and_dump_frames_ffmpeg(src_video_path, video_tar_path=tar_video_path, out_height=resolution_h, out_folder=concated_frames_folders)
            else:
                print(f"Processing video pair using decord: {video_save_name}")
                concated_frames_folders = get_videos_from_path(
                    video_src_path=src_video_path, 
                    video_tar_path=tar_video_path, 
                    resolution_h=resolution_h, 
                    out_folder=concated_frames_folders
                )
            all_img_list = [os.path.join(concated_frames_folders, f) for f in sorted(os.listdir(concated_frames_folders))]
            all_img_list = all_img_list[::stride]

            # conduct vllm input pair...
            data_list.append({
                "image_path": all_img_list,
                "question": sys_prompt + f'instruction: {prompt}',
            })

            sub_dict = {}
            sub_dict['video_name'] = video_save_name
            all_dict_list.append(sub_dict)

        except Exception as e:
            print(e)
            fail_video_dict[video_save_name] = str(e)


    with open(json_path.replace('.json', '_failed.json'), "w", encoding="utf-8") as f:
        json.dump(fail_video_dict, f, ensure_ascii=False, indent=2)

    # run inference...
    responses = engine.process_image_query(data_list, num_workers=8)
    
    # save dict into json file..
    for i in range(len(all_dict_list)):
        response_fix = load_output_as_json(responses[i])
        all_dict_list[i]['response'] = response_fix

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_dict_list, f, ensure_ascii=False, indent=4)


