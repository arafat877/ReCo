#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, argparse, sys, os
from pathlib import Path
from typing import Iterable, Any, List, Dict

METRICS = ["edit_accuracy", "video_quality", "naturalness"]
# 各子dict内部细分指标（按 scores 顺序）
INDICATORS: Dict[str, List[str]] = {
    "edit_accuracy": ["SA", "SP", "CP"],
    "video_quality": ["VF", "TS", "ES"],
    "naturalness":   ["AN", "SN", "MN"],
}

def geometric_mean(scores: Iterable[Any]) -> float:
    """几何平均：(∏ scores) ** (1/n)。若为空/非数值/含负数，返回 0.0。"""
    vals: List[float] = []
    for s in scores:
        try:
            v = float(s)
        except Exception:
            return 0.0
        if v < 0:
            return 0.0
        vals.append(v)
    n = len(vals)
    if n == 0:
        return 0.0
    prod = 1.0
    for v in vals:
        prod *= v
    return prod ** (1.0 / n)

def parse_weights(s: str) -> Dict[str, float]:
    if not s:
        return {}
    out: Dict[str, float] = {}
    for seg in s.split(","):
        seg = seg.strip()
        if not seg:
            continue
        if "=" not in seg:
            raise ValueError(f"Bad weight segment: {seg}")
        k, v = seg.split("=", 1)
        k = k.strip()
        v = float(v.strip())
        if v < 0:
            raise ValueError(f"Negative weight for {k}")
        out[k] = v
    return out

def round4(x: float) -> float:
    return round(float(x), 4)



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



def read_video_name_list_from_txt(txt_file):

    all_video_dict = parse_instruction_file(txt_file)

    all_video_name_list = []
    for video_dict in all_video_dict:

        # define video path name
        prompt = video_dict['instructed_prompt']

        video_base_name = video_dict['src_video_path']
        prompt_name = prompt.replace(' ', '_').replace('.', '').replace(',','').replace(':',' ')
        vide_save_name = video_base_name.replace('.mp4', '')
        video_save_name = f'{vide_save_name}_{prompt_name[:80]}.mp4'

        all_video_name_list.append(video_save_name)

    return all_video_name_list


def main():
    ap = argparse.ArgumentParser(
        description="Compute per-metric geometric mean for each item, metric subtotals, weighted overall, and per-indicator averages."
    )
    ap.add_argument("--json_folder", default="all_results/gemini_results",
                    help="Folder that contains *gemini.json files")
    ap.add_argument("--base_txt_folder", type=str, default="configs")
    ap.add_argument("-o", "--output", default=None,
                    help="Optional output path; default: write *_final.json next to each input")
    ap.add_argument("--weights", default="",
                    help="Metric weights, e.g. 'edit_accuracy=0.5,video_quality=0.3,naturalness=0.2', default all 1/3")
    args = ap.parse_args()

    all_json_list = [
        os.path.join(args.json_folder, f)
        for f in os.listdir(args.json_folder)
        if f.endswith("gemini.json")
    ]

    for json_path in all_json_list:
        inp = Path(json_path)
        if not inp.exists():
            print(f"[error] file not found: {inp}", file=sys.stderr); sys.exit(1)

        with open(inp, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            print("[error] input must be a list of dicts", file=sys.stderr); sys.exit(2)

        # --- A) 子dict gmean 的 subtotal 统计器 ---
        metric_sum: Dict[str, float] = {m: 0.0 for m in METRICS}
        metric_cnt: Dict[str, int]   = {m: 0   for m in METRICS}

        # --- B) 9 个细分指标的“全局均值”统计器（算术平均）---
        # 结构：indicator_sum["edit_accuracy"]["SA"] 累加；同样有 count
        indicator_sum: Dict[str, Dict[str, float]] = {
            m: {label: 0.0 for label in INDICATORS[m]} for m in METRICS
        }
        indicator_cnt: Dict[str, Dict[str, int]] = {
            m: {label: 0 for label in INDICATORS[m]} for m in METRICS
        }

        per_item_results: List[Dict] = []

        # --------------- New: Read config_file as dict -----------------
        base_txt_folder = args.base_txt_folder
        base_task_name = os.path.basename(json_path).split('_vllm')[0]
        txt_file_name = os.path.join(base_txt_folder, f'{base_task_name}.txt')
        all_video_name_list = read_video_name_list_from_txt(txt_file_name)


        # 逐条计算并写回 gmean；同时累加各 indicator
        for item in data: 
            if not isinstance(item, dict):
                continue

            # ---------New: read corespond txt file to detect-----
            if item['video_name'] not in all_video_name_list:
                continue
            
            new_item = dict(item)
            resp = dict(new_item.get("response") or {})

            for m in METRICS:
                block = dict(resp.get(m) or {})
                scores = block.get("scores", [])
                # 1) gmean
                gm = geometric_mean(scores)
                gm_r = round4(gm)
                block["gmean"] = gm_r
                resp[m] = block

                metric_sum[m] += gm
                metric_cnt[m] += 1

                # 2) 细分指标：逐个累加（算术平均）
                labels = INDICATORS[m]
                for idx, label in enumerate(labels):
                    if idx < len(scores):
                        try:
                            v = float(scores[idx])
                        except Exception:
                            continue
                        indicator_sum[m][label] += v
                        indicator_cnt[m][label] += 1

            new_item["response"] = resp
            per_item_results.append(new_item)

        # --- 1) 三个子dict的 subtotal（宏平均）---
        subtotals: Dict[str, float] = {}
        present_metrics = []
        for m in METRICS:
            if metric_cnt[m] > 0:
                subtotals[m] = round4(metric_sum[m] / metric_cnt[m])
                present_metrics.append(m)
            else:
                subtotals[m] = 0.0

        print("\nSub totals (macro avg across items):")
        for m in METRICS:
            print(f"- {m}: {subtotals[m]}")

        # --- 2) 最终加权平均 ---
        raw_w = parse_weights(args.weights)
        if present_metrics:
            if raw_w:
                w = {m: raw_w.get(m, 0.0) for m in present_metrics}
                ssum = sum(w.values())
                if ssum == 0:
                    w = {m: 1.0 / len(present_metrics) for m in present_metrics}
                else:
                    w = {m: v / ssum for m, v in w.items()}
            else:
                w = {m: 1.0 / len(present_metrics) for m in present_metrics}
            weighted_overall = round4(sum(subtotals[m] * w[m] for m in present_metrics))
        else:
            w = {}
            weighted_overall = 0.0

        print(f"- weighted overall: {weighted_overall}")
        print(f"  (weights used: { {m: round4(w[m]) for m in w} })")

        # --- 3) 9 个细分指标的全局平均（算术平均）---
        per_indicator_avg: Dict[str, Dict[str, float]] = {}
        for m in METRICS:
            per_indicator_avg[m] = {}
            for label in INDICATORS[m]:
                c = indicator_cnt[m][label]
                avg = (indicator_sum[m][label] / c) if c else 0.0
                per_indicator_avg[m][label] = round4(avg)

        # 组织输出
        out_payload = {
            "per_indicator_avg": per_indicator_avg,         # 9 个细分指标在所有样本上的平均（算术平均）
            "sub_totals": subtotals,                        # 三个子dict gmean 的宏平均
            "weights_used": {m: round4(w[m]) for m in w},
            "weighted_overall": weighted_overall,           # 三个 subtotal 的加权平均
            "count": len(per_item_results),
            "items": per_item_results                       # 每条样本：各子dict已写入 gmean
        }

        out_path = args.output or str(inp).rstrip(".json") + "_final_120.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_payload, f, ensure_ascii=False, indent=2)
        print(f"[saved] {out_path}")

if __name__ == "__main__":
    main()

