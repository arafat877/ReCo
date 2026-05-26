import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

import torch
import time
from tqdm import tqdm
import re
import json
from torch.utils.data import DataLoader
import imageio
import numpy as np


def get_dict_list_from_json_folder(json_folder, dict_key=None):

    all_json_files = [os.path.join(json_folder, f) for f in os.listdir(json_folder) if f.endswith('.json')]
    
    all_dict_list = []
    for json_path in all_json_files:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # for sub_data in data:
        #     sub_data['task_name'] = task_name

        all_dict_list.extend(data)
    
    if dict_key is not None:
        filtered = [
            d for d in all_dict_list
            if (d[dict_key] is not None and "none" not in d[dict_key].lower() and 'error' not in d[dict_key].lower() and len(d[dict_key])>len(str(dict_key))+10)
        ]
    else:
        filtered = all_dict_list
    # print(f'total {task_name} is: {len(filtered)}')
    return filtered


import boto3
from torch.utils.data import Dataset
import time
import contextlib
import uuid
import decord


@contextlib.contextmanager
def temp_file_context_manager(directory, suffix):
    filename = os.path.join(directory, f"{uuid.uuid4()}{suffix}")
    try:
        yield filename
    finally:
        if os.path.exists(filename):
            os.remove(filename)


def download_video_from_s3(video_clip_s3, video_clip, s3, retry=3):
    bucket_name = video_clip_s3.split('/')[2]
    file_name ='/'.join(video_clip_s3.split('/')[3:])
    for _ in range(retry):
        try:
            s3.download_file(bucket_name, file_name, video_clip)
            return
        except Exception as e:
            print(f'download {video_clip_s3} -> {video_clip} failed with {e}, sleep 5s')
            time.sleep(5)
            raise RuntimeError(f'download {video_clip_s3} -> {video_clip} failed')


def get_file(file_path, s3, prefix):

    with temp_file_context_manager('/dev/shm', prefix) as local_file_path:
        try:
            if 's3://' in file_path:
                download_video_from_s3(file_path, local_file_path, s3, 3)
                
                if prefix == '.mp4':
                    vr = decord.VideoReader(local_file_path)
                    return vr
                else:
                    mask_array = np.load(local_file_path)
                    return mask_array


        except Exception as e:
            print(f'Failed to read video: {file_path}')
            raise e


class ReCo_Dataset(Dataset):
    def __init__(
        self,
        all_data_list: list = None,
        height: int = 480,
        width: int = 832,
        max_num_frames: int = 81,
        rank: int = 0,
        world_size: int = 1,
        base_video_folder: str = '',
        read_video_from_local = False,

    ) -> None:
        super().__init__()
        self.all_data_list = all_data_list
        self.height = height
        self.width = width
        self.max_num_frames = max_num_frames
        self.rank = rank
        self.world_size = world_size
        self.instance_metas = self.all_data_list[self.rank::self.world_size]
        self.s3 = boto3.client('s3')
        self.base_video_folder = base_video_folder
        self.read_video_from_local = read_video_from_local


    def __len__(self):
        return len(self.instance_metas)

    def read_video(self, video_path):

        vr = get_file(video_path, self.s3, prefix='.mp4')

        return vr

    def calculate_frame_indices(self, total_frames: int, video_fps: float, num_samples: int = 0) -> np.ndarray:

        beg = 0  # 固定从第0帧开始
        end = total_frames

        frame_interval = video_fps / 16
        indices = np.arange(beg, end, frame_interval).astype(int)
        if len(indices) < num_samples:
            repeated_indices = np.array([indices[-1]] * (num_samples - len(indices)))
            indices = np.concatenate([indices, repeated_indices])
        
        if len(indices) > num_samples:
            frame_start = 0  
            frame_end = frame_start + num_samples
            indices = indices[frame_start:frame_end]       

        indices = np.clip(indices, 0, total_frames - 1)
        return indices


    def __getitem__(self, index):

        item = self.instance_metas[index]
        src_video_path = os.path.join(self.base_video_folder,item['src_video'])
        tar_video_path = os.path.join(self.base_video_folder, item['tar_video'])

        video_tar_name = os.path.basename(item['tar_video']).replace('.mp4','')
        video_src_name = os.path.basename(item['src_video']).replace('.mp4','')
        prompt = item['instruction_final_refine']

        try:
            # -------------- STEP.1 read src video and process src video
            if not self.read_video_from_local:
                # load from aoss
                vr_src = self.read_video(src_video_path)
                vr_tar = self.read_video(tar_video_path)
            else:
                # load from local
                vr_src = decord.VideoReader(src_video_path)
                vr_tar = decord.VideoReader(tar_video_path)

            src_video = torch.from_numpy(vr_src.get_batch(list(range(0, len(vr_src)))).asnumpy())
            src_video = (src_video/255) *2 -1

            tar_video = torch.from_numpy(vr_tar.get_batch(list(range(0, len(vr_tar)))).asnumpy())
            tar_video = (tar_video/255) *2 -1

            concated_video = torch.concat([src_video, tar_video], dim=2)

            return {
                'src_video': src_video.permute(3,0,1,2),                    # video input, [c,f,h,w], tensor [-1, 1]
                'tar_video': tar_video.permute(3,0,1,2),                            # tar_video, [c,f,h,w], tensor [-1, 1]
                'prompt': prompt, 
                'video_tar_name': video_tar_name,                   # contain task_name
                'video_src_name': video_src_name,
                'concated_video': concated_video.permute(3,0,1,2),
                'item': item,
            }
        
        except Exception as e:
            return {
                'src_video': torch.zeros((81,480,832,3)).permute(3,0,1,2),                    # video input, [c,f,h,w], tensor [-1, 1]
                'tar_video': torch.zeros((81,480,832,3)).permute(3,0,1,2),          # mask for tar_video_key, [c,f,h,w], 0-1                  # tar_video, [c,f,h,w], tensor [-1, 1]
                'prompt': None, 
                'video_tar_name': video_tar_name,                   # contain task_name
                'video_src_name': video_src_name,
                'concated_video': torch.zeros((81,480,832,3)).permute(3,0,1,2),
                'item': None,
            }




def save_video(frames, save_path, fps, quality=8, ffmpeg_params=None):
    writer = imageio.get_writer(save_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params)
    for frame in frames:
        frame = np.array(frame)
        writer.append_data(frame)
    writer.close()




def collate_fn(batch):

    src_video = torch.stack([item['src_video'] for item in batch], dim=0)
    tar_video = torch.stack([item['tar_video'] for item in batch], dim=0)
    concated_video = torch.stack([item['concated_video'] for item in batch], dim=0)
    prompt = [item['prompt'] for item in batch]
    video_tar_name = [item['video_tar_name'] for item in batch]
    video_src_name = [item['video_src_name'] for item in batch]
    # src_video_mask = torch.stack([item['src_video_mask'] for item in batch], dim=0)
    item = [item['item'] for item in batch]

    dict_data = {
        'src_video': src_video,
        'tar_video': tar_video,
        'prompt': prompt,
        'video_tar_name': video_tar_name,
        'video_src_name': video_src_name,
        # 'src_video_mask': src_video_mask,
        'concated_video': concated_video,
        'item': item
    }

    return dict_data



import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="./ReCo-Data/style/style_data_configs.json", help="config file path")
    parser.add_argument("--video_folder", type=str, default="./ReCo-Data", help="Video base folder path")
    parser.add_argument("--debug", action="store_true", help="Debug mode with fewer videos")

    args = parser.parse_args()

    # Define cache folder
    task_name = os.path.dirname(args.json_path)
    cache_folder = f"./cache_folder/{task_name}"
    os.makedirs(cache_folder, exist_ok=True)

    # ===================== step1: Read video json configs =====================
    with open(args.json_path, "r", encoding="utf-8") as f:
        all_dict_list = json.load(f)

    if args.debug:
        import random
        rng = random.Random(2026)
        rng.shuffle(all_dict_list)
    print(f'====== Process, total {len(all_dict_list)} videos ======')


    # ===================== step2: Create dataset and dataloader =====================
    video_dataset = ReCo_Dataset(all_data_list=all_dict_list, base_video_folder=args.video_folder, read_video_from_local=True)
    dataloader_debug = DataLoader(video_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    all_video_info_dict_list = []
    for i,data_iter in tqdm(enumerate(dataloader_debug), total=len(dataloader_debug)):
 
        if data_iter['item'][0] is None:
            tar_video_name = data_iter['video_tar_name'][0]
            print(f'Read erro {tar_video_name}')

        # get src_video, video_mask, tar_video
        src_video = (data_iter['src_video'][0].permute(1,2,3,0).numpy()/2 + 0.5)*255
        tar_video = (data_iter['tar_video'][0].permute(1,2,3,0).numpy()/2 + 0.5)*255      # 0-255

        concated_video = (data_iter['concated_video'][0].permute(1,2,3,0).numpy()/2 + 0.5)*255
        video_ori_np = [concated_video[i].astype(np.uint8) for i in range(concated_video.shape[0])]

        video_tar_name = data_iter['video_tar_name'][0]
        video_src_name = data_iter['video_src_name'][0]
        prompt = data_iter['prompt'][0]

        # visual...
        clean_name = f"{video_tar_name}.mp4"
        concat_video_path = os.path.join(cache_folder, clean_name)
        save_video(video_ori_np, concat_video_path, fps=16, quality=5)

        # record text prompt
        txt_path = os.path.join(cache_folder, f'{video_tar_name}.txt')
        with open(txt_path, 'w') as f:
            f.write(str(prompt))


        if args.debug and i >= 50:
            break



