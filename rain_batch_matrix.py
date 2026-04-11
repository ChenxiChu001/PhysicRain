"""
============================================================================
PhysicRain 参数矩阵批量生成 (v2)
============================================================================

对 3440 组(降雨量, 风速, 风向)参数遍历,
每组用不同的 Cityscapes 原图, 生成掩码并调用 Gemini API 合成雨天图像.

用法:
  # 先导测试 20 张
  python rain_batch_matrix.py --test 20

  # 全量生成
  python rain_batch_matrix.py

============================================================================
"""

import os
import sys
import time
import json
import base64
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from typing import List, Dict, Tuple, Optional
from collections import Counter as StdCounter
import numpy as np
import cv2
import requests

from rain_renderer import render_rain_mask


# ===========================================================================
# 配置
# ===========================================================================

DEFAULT_API_URL = 'https://api.aipaibox.com/v1/chat/completions'
DEFAULT_MODEL = 'gemini-3.1-flash-image-preview'
DEFAULT_TIMEOUT = 300

# Cityscapes 数据集默认路径
DEFAULT_IMAGE_DIR = 'dataset/leftImg8bit_trainvaltest/leftImg8bit'
DEFAULT_DEPTH_DIR = 'dataset/depth_trainvaltest/gray16'
DEFAULT_CAMERA_DIR = 'dataset/camera_trainvaltest/camera'
DEFAULT_OUTPUT_DIR = 'dataset_create'
DEFAULT_MASK_DIR = 'dataset_create_masks'

# 风向映射
WIND_DIRS = {
    '北': 0, '东北': 45, '东': 90, '东南': 135,
    '南': 180, '西南': 225, '西': 270, '西北': 315,
}
WIND_DIR_NAMES = ['北', '东北', '东', '东南', '南', '西南', '西', '西北']

# 掩码生成默认参数 (与 GUI 调试值一致)
MASK_DEFAULTS = dict(
    exposure_time=0.012,
    turbulence_deg=0.0,
    harmonic_blend=0.92,
    brightness_min=45,
    brightness_max=90,
    focus_distance=0.5,
    dof_strength=0.10,
    radial_strength=0.3,
    depth_scale=100.0,
    streak_softness=0.0,
)


# ===========================================================================
# 7 套提示词模板
# ===========================================================================

PROMPT_TEMPLATES = {
    1: (
        '这是一张只添加了雨痕(灰白色线条)的微雨图像。\n'
        '核心约束 (Core Constraint)： 所有雨痕的确切空间分布、方向和密度不得改变。\n'
        '确保雨痕融入到了整个环境中，雨痕和周围的环境不要对比度太高，不要格格不入。\n'
        '请你保证真实物理特性。\n'
        '请你实现场景真实化 (Scene Realism)，细节如下：\n'
        '地面积水： 无明显积水。柏油路面整体变暗，颗粒纹理清晰可见。\n'
        '路面反光： 镜面反射尚未形成。仅有微弱的环境光漫反射，路面颗粒在光下呈现不均匀的湿润感。\n'
        '表面湿润： 车辆和暴露的自行车架表面形成极薄的、不连续的湿润感，颜色略微加深，但不如小雨均匀。\n'
        '大气效果： 远景对比度极轻微下降。空气湿度增加带来的轻度背景散射。\n'
        '近景雨雾： 无。\n'
        '环境光交互： 受到光线直射的雨滴更亮，与环境光（如车灯，红绿灯等）有极弱的交互。\n'
        '光照改变： 直射光变弱，场景整体亮度稍低。转为均匀的漫射光（阴天环境光），色彩饱和度轻微降低，略微偏灰。\n'
        '天空： 均匀的薄灰白云层，完全遮蔽太阳。\n'
        '细节保留： 保留所有原始细节，如汽车、牌照、人物，和建筑。'
    ),
    2: (
        '这是一张只添加了雨痕(灰白色线条)的小雨图像\n'
        '核心约束 (Core Constraint)：所有雨痕的确切空间分布、方向和密度不得改变\n'
        '确保雨痕融入到了整个环境中，雨痕和周围的环境不要对比度太高，不要格格不入。\n'
        '请你保证真实物理特性\n'
        '请你实现场景真实化 (Scene Realism),细节如下：\n'
        '地面积水： 柏油路面开始形成不连续的薄水膜，但路面颗粒纹理依然可见。\n'
        '路面反光： 镜面反射开始显现，能模糊倒映出车辆尾灯、路灯的彩色光晕。\n'
        '表面湿润： 车辆和暴露在外的自行车架被均匀的水膜覆盖，呈现出完整的湿润光泽，颜色明显变深。\n'
        '大气效果： 远景对比度轻微下降，空气湿度增加带来轻度的背景散射。\n'
        '近景雨雾： 几乎无，仅在雨滴撞击车顶和地面时激起小水花。\n'
        '环境光交互：受到光线直射的雨滴更亮，与环境光（如车灯，红绿灯等）有交互。\n'
        '光照改变： 直射光完全消失，转为均匀的漫射光（阴天环境光），场景整体色彩饱和度略有降低,偏灰。\n'
        '天空： 均匀的灰白色云层，完全遮蔽太阳。\n'
        '细节保留：保留所有原始细节，如汽车、牌照、人物，和建筑。'
    ),
    3: (
        '这是一张只添加了雨痕(灰白色线条)的中雨图像\n'
        '核心约束 (Core Constraint)： 所有雨痕的确切空间分布、方向和密度不得改变\n'
        '确保雨痕融入到了整个环境中，雨痕和周围的环境不要对比度太高，不要格格不入。\n'
        '请你保证真实物理特性\n'
        '请你实现场景真实化 (Scene Realism),细节如下：\n'
        '地面积水： 柏油路面形成连续的薄水膜，颗粒纹理被覆盖。开始出现细小、不连续的撞击水花。\n'
        '路面反光： 镜面反射增强，能清晰倒映出车辆尾灯、路灯的彩色光晕（比小雨更清晰）。\n'
        '表面湿润： 车辆和暴露在外的自行车架被均匀、深层的湿润光泽覆盖，颜色明显变深，有细小水滴在表面流动。\n'
        '大气效果： 远景对比度下降明显，轻度体积散射，场景景深开始压缩。\n'
        '近景雨雾： 有少量细微雨雾，仅在雨滴撞击车顶和地面时激起小水花。\n'
        '环境光交互： 车灯光晕变大，穿透力减弱。\n'
        '光照改变： 场景整体亮度下降，环境光灰调加重，场景色彩饱和度降低，偏灰。\n'
        '天空： 均匀的灰色积层云。\n'
        '细节保留： 保留所有原始细节，如汽车、牌照、人物，和建筑。'
    ),
    4: (
        '这是一张只添加了雨痕(灰白色线条)的大雨图像。\n'
        '核心约束 (Core Constraint)： 所有雨痕的确切空间分布、方向和密度不得改变。\n'
        '确保雨痕融入到了整个环境中，雨痕和周围的环境不要对比度太高，不要格格不入。\n'
        '请你保证真实物理特性\n'
        '请你实现场景真实化 (Scene Realism),细节如下：\n'
        '地面积水： 路面积水深度可达2厘米，刚好淹没车轮边缘。撞击水花密集且明显。\n'
        '路面反光： 镜面反射被流动的积水和水花严重干扰，变得破碎，呈现出混沌的混沌发光状态。\n'
        '表面湿润： 车辆和暴露的自行车架被厚重的水膜覆盖，细节开始模糊。\n'
        '大气效果： 能见度显著降低，体积散射强烈，中远景景深被压缩。\n'
        '近景雨雾： 产生明显的细微水雾，空气中充满微小的悬浮水滴，遮挡了物体细节。\n'
        '环境光交互： 车灯穿透力极差，由于雨雾的原因仅在近处形成模糊的光晕晕染。\n'
        '光照改变： 场景整体亮度低，环境光灰（灰调），车灯光晕变大，穿透力减弱。\n'
        '天空： 深灰色的层积云。\n'
        '结构保留： 保留所有原始结构，如汽车、建筑等。'
    ),
    5: (
        '这是一张只添加了雨痕(灰白色线条)的暴雨图像。\n'
        '核心约束 (Core Constraint)： 所有雨痕的确切空间分布、方向和密度不得改变。\n'
        '确保雨痕融入到了整个环境中，雨痕和周围的环境不要对比度太高，不要格格不入。\n'
        '请你保证真实物理特性。\n'
        '请你实现场景真实化 (Scene Realism),细节如下：\n'
        '地面积水： 路面积水深度可达5厘米，局部内涝。撞击产生大量、密集的白色水花。\n'
        '路面反光： 镜面反射几乎完全被飞溅的水花和厚重的流动水层打碎，呈现混沌状态。\n'
        '表面湿润： 车辆和暴露的自行车架细节被厚重流动的水层和水花严重掩盖。\n'
        '大气效果： 能见度非常低，体积散射（Volumetric scattering）非常强，中远景被水幕完全遮蔽。\n'
        '近景雨雾： 雨滴砸向车辆金属外壳和硬质路面时，产生非常强烈的水雾。空气中充满微小的悬浮水滴，遮挡了物体细节。\n'
        '光照改变： 场景整体亮度非常低，环境光灰（灰调），车灯形成模糊的晕染。\n'
        '天空： 灰色的滚滚积雨云，云层细节丰富且暗。\n'
        '结构保留： 保留所有原始结构，如汽车、建筑等。'
    ),
    6: (
        '这是一张只添加了雨痕(灰白色线条)的大暴雨图像。\n'
        '核心约束 (Core Constraint)： 所有雨痕的确切空间分布、方向和密度不得改变\n'
        '确保雨痕融入到了整个环境中，雨痕和周围的环境不要对比度太高，不要格格不入。\n'
        '请你保证真实物理特性\n'
        '请你实现场景真实化 (Scene Realism),细节如下：\n'
        '地面积水： 积水淹没车轮五分之一。有局部内涝现象，雨滴撞击产生大量白色水花，形成一层跳动的水膜。\n'
        '路面反光： 由飞溅水花形成的高频破碎闪烁，光线在流动的深层积水中产生浑浊的漫散效果。\n'
        '表面湿润： 车辆和建筑物表面厚重的水帘顺着金属外壳快速流淌，物体的轮廓边缘因水流的厚度而产生视觉上的物理变形。\n'
        '大气效果： 能见度受限在极短距离内，体积散射（Volumetric scattering）呈现出一种压抑的厚重感，中景远景完全消散在白灰色水幕中。\n'
        '近景雨雾： 强力降雨撞击硬质表面激起的雾化效果开始向上升腾，形成一层浓重近地水雾，显著遮蔽近景细节。\n'
        '光照改变： 环境亮度进一步压低，呈现灰色调。车灯光晕范围扩大但核心亮度被大幅稀释，形成明显的光路散射路径。\n'
        '天空： 浓重的乌云，呈现出深铅灰色。\n'
        '结构保留： 保留所有原始结构，如汽车、建筑等。'
    ),
    7: (
        '这是一张只添加了雨痕(灰白色线条)的大暴雨图像\n'
        '核心约束 (Core Constraint)：所有雨痕的确切空间分布、方向和密度不得改变\n'
        '确保雨痕融入到了整个环境中，雨痕和周围的环境不要对比度太高，不要格格不入。\n'
        '请你保证真实物理特性\n'
        '请你实现场景真实化 (Scene Realism),细节如下：\n'
        '地面积水： 路面积水淹没车轮三分之一,严重内涝,雨滴密集撞击产生大量白色水花。\n'
        '路面反光： 反光被密集的飞溅水花和厚重的流动水层严重打碎，呈现出漫反射与镜面反射交织的混沌发光状态。\n'
        '表面湿润： 车辆如同被高压水柱冲刷，表面细节被厚重流动的水层、水花和反光完全掩盖。\n'
        '大气效果： 能见度极低,发生强烈的体积散射（Volumetric scattering），中远景被浓重的白灰色水幕完全遮蔽，场景景深被严重压缩。\n'
        '近景雨雾： 雨滴砸向车辆金属外壳和硬质路面时，产生非常强烈并且厚重的水雾笼罩在画面中。空气中充满微小的悬浮水滴，遮挡了大量物体细节。\n'
        '光照改变： 场景整体亮度低，环境光灰（暗灰调），车灯穿透力极差,由于雨雾的原因仅在近处形成模糊的光晕晕染。\n'
        '天空：深灰色的滚滚积雨云，云层有多处暗部细节。\n'
        '结构保留：保留所有原始结构，如汽车、建筑等。'
    ),
}


# ===========================================================================
# 参数矩阵生成
# ===========================================================================

def _frange(start: float, stop: float, step: float) -> List[float]:
    vals = []
    v = start
    while v <= stop + step * 0.01:
        vals.append(round(v, 2))
        v += step
    return vals


PARAM_MATRIX_SPEC = [
    (1, _frange(0.5, 2.5, 0.5),   _frange(0, 4, 0.5)),
    (2, _frange(3.0, 6.0, 0.5),   _frange(0, 4, 0.5)),
    (3, _frange(6.5, 10.0, 0.5),  _frange(0, 4, 0.5)),
    (3, _frange(11.0, 23.0, 1.0), _frange(0, 4, 0.5)),
    (4, _frange(24.0, 40.0, 1.0), _frange(1, 5, 1.0)),
    (5, _frange(41.0, 45.0, 2.0), _frange(1, 11, 2.0)),
    (6, _frange(46.0, 50.0, 2.0), _frange(1, 11, 2.0)),
    (7, _frange(55.0, 60.0, 5.0), _frange(1, 11, 2.0)),
]


def build_param_matrix() -> List[Dict]:
    combos = []
    for template_id, rain_rates, wind_speeds in PARAM_MATRIX_SPEC:
        for rr in rain_rates:
            for ws in wind_speeds:
                for wd_name in WIND_DIR_NAMES:
                    combos.append({
                        'template': template_id,
                        'rain_rate': rr,
                        'wind_speed': ws,
                        'wind_dir_name': wd_name,
                        'wind_direction': WIND_DIRS[wd_name],
                    })
    return combos


def sample_pilot_combos(combos: List[Dict], n: int = 20) -> List[Dict]:
    by_template = {}
    for c in combos:
        by_template.setdefault(c['template'], []).append(c)
    templates = sorted(by_template.keys())
    total_combos = len(combos)
    allocation = {}
    remaining = n
    for t in templates:
        allocation[t] = 2
        remaining -= 2
    for t in templates:
        if remaining <= 0:
            break
        extra = max(0, round(len(by_template[t]) / total_combos * remaining))
        allocation[t] += extra
    sampled = []
    for t in templates:
        pool = by_template[t]
        count = min(allocation[t], len(pool))
        step = max(1, len(pool) // count)
        picks = pool[::step][:count]
        sampled.extend(picks)
    return sampled[:n]


def make_filename(combo: Dict, img_stem: str) -> str:
    """生成文件名: {原图stem}_T{模板}_R{降雨量}_W{风速}_D{风向}"""
    rr = combo['rain_rate']
    ws = combo['wind_speed']
    rr_str = f'{rr:.0f}' if rr == int(rr) else f'{rr:.1f}'
    ws_str = f'{ws:.0f}' if ws == int(ws) else f'{ws:.1f}'
    return f"{img_stem}_T{combo['template']}_R{rr_str}_W{ws_str}_D{combo['wind_dir_name']}"


# ===========================================================================
# Cityscapes 文件匹配
# ===========================================================================

def _strip_suffix(stem: str) -> str:
    for sfx in ('_leftImg8bit', '_rightImg8bit', '_disparity', '_depth', '_camera'):
        if stem.endswith(sfx):
            return stem[:-len(sfx)]
    return stem


def collect_cityscapes_images(image_dir: str) -> List[str]:
    """递归收集所有 png/jpg 图像, 按路径排序"""
    images = []
    for root, _, files in os.walk(image_dir):
        for f in sorted(files):
            if f.lower().endswith(('.png', '.jpg')):
                images.append(os.path.join(root, f))
    return sorted(images)


def find_matching_depth(img_path: str, image_dir: str, depth_dir: str) -> Optional[str]:
    """根据原图路径找到对应的深度图"""
    basename = os.path.splitext(os.path.basename(img_path))[0]
    stem = _strip_suffix(basename)
    # 相对子目录: test/berlin/ 等
    rel_dir = os.path.relpath(os.path.dirname(img_path), image_dir)
    if rel_dir == '.':
        rel_dir = ''

    candidates = [
        f'{basename}_depth_u16.png',
        f'{basename}_depth.png',
        f'{stem}_depth_u16.png',
        f'{stem}_leftImg8bit_depth_u16.png',
    ]
    for name in candidates:
        p = os.path.join(depth_dir, rel_dir, name)
        if os.path.isfile(p):
            return p
        p = os.path.join(depth_dir, name)
        if os.path.isfile(p):
            return p
    return None


def find_matching_camera(img_path: str, image_dir: str, camera_dir: str) -> Optional[str]:
    """根据原图路径找到对应的相机参数 JSON"""
    basename = os.path.splitext(os.path.basename(img_path))[0]
    stem = _strip_suffix(basename)
    rel_dir = os.path.relpath(os.path.dirname(img_path), image_dir)
    if rel_dir == '.':
        rel_dir = ''

    candidates = [
        f'{stem}_camera.json',
        f'{basename}_camera.json',
        f'{basename}.json',
    ]
    for name in candidates:
        p = os.path.join(camera_dir, rel_dir, name)
        if os.path.isfile(p):
            return p
        p = os.path.join(camera_dir, name)
        if os.path.isfile(p):
            return p
    return None


# ===========================================================================
# 图像编解码工具
# ===========================================================================

def image_to_base64(image_path: str) -> str:
    raw = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f'无法读取图像: {image_path}')
    _, buf = cv2.imencode('.png', img)
    return base64.b64encode(buf).decode('utf-8')


def numpy_to_base64(img: np.ndarray) -> str:
    _, buf = cv2.imencode('.png', img)
    return base64.b64encode(buf).decode('utf-8')


def base64_to_image(b64_str: str) -> Optional[np.ndarray]:
    if ',' in b64_str:
        b64_str = b64_str.split(',', 1)[1]
    img_bytes = base64.b64decode(b64_str)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ===========================================================================
# Gemini API 调用 (SSL 错误可重试)
# ===========================================================================

def call_gemini(
    original_b64: str,
    mask_image: np.ndarray,
    prompt: str,
    api_key: str,
    api_url: str = DEFAULT_API_URL,
    model: str = DEFAULT_MODEL,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = 2,
) -> np.ndarray:
    b64_mask = numpy_to_base64(mask_image)

    payload = {
        'model': model,
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{original_b64}'}},
                {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64_mask}'}},
                {'type': 'text', 'text': prompt},
            ]
        }],
        'max_tokens': 4096,
    }
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }

    last_error = None
    for attempt in range(1, max_retries + 2):
        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code != 200:
                last_error = f'HTTP {resp.status_code}: {resp.text[:200]}'
                if attempt <= max_retries:
                    time.sleep(5)
                continue

            data = resp.json()
            if 'choices' not in data or not data['choices']:
                last_error = f'无 choices: {json.dumps(data, ensure_ascii=False)[:200]}'
                if attempt <= max_retries:
                    time.sleep(5)
                continue

            content = data['choices'][0].get('message', {}).get('content', '')

            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get('type') == 'image_url':
                        url = part.get('image_url', {}).get('url', '')
                        if 'base64,' in url:
                            img = base64_to_image(url)
                            if img is not None:
                                return img

            if isinstance(content, str) and 'data:image' in content and 'base64,' in content:
                start = content.index('base64,') + 7
                end = content.index('"', start) if '"' in content[start:] else len(content)
                img = base64_to_image(content[start:end])
                if img is not None:
                    return img
                last_error = 'base64 解码失败'
            else:
                last_error = f'模型返回文字: {str(content)[:150]}'

            if attempt <= max_retries:
                time.sleep(5)

        except requests.exceptions.Timeout:
            last_error = f'超时 ({timeout}s)'
            if attempt <= max_retries:
                time.sleep(5)
        except (requests.exceptions.ProxyError,
                requests.exceptions.ConnectionError,
                requests.exceptions.SSLError) as e:
            # SSL/连接错误: 允许重试
            last_error = f'连接错误: {e}'
            if attempt <= max_retries:
                time.sleep(5)
        except Exception as e:
            last_error = str(e)
            if attempt <= max_retries:
                time.sleep(5)

    raise RuntimeError(f'API 失败 ({max_retries + 1}次): {last_error}')


# ===========================================================================
# Key 池 + 连续失败追踪 + 备用令牌自动替换
# ===========================================================================

class KeyPool:
    def __init__(self, keys: List[str], spare_keys: List[str] = None):
        self._keys = list(keys)
        self._spare_keys = list(spare_keys) if spare_keys else []
        self._queue = Queue()
        self._consecutive_fails = {}
        self._disabled = set()
        self._replaced = {}  # old_key -> new_key 替换记录
        self._lock = threading.Lock()
        for k in self._keys:
            self._queue.put(k)
            self._consecutive_fails[k] = 0

    def acquire(self) -> Optional[str]:
        while True:
            key = self._queue.get()
            with self._lock:
                if key in self._disabled:
                    continue
                return key

    def release(self, key: str):
        with self._lock:
            if key not in self._disabled:
                self._queue.put(key)

    def report_success(self, key: str):
        with self._lock:
            self._consecutive_fails[key] = 0

    def report_fail(self, key: str) -> str:
        """返回状态: 'ok'(继续用), 'replaced'(已替换), 'disabled'(无备用)"""
        with self._lock:
            self._consecutive_fails[key] += 1
            if self._consecutive_fails[key] >= 3:
                self._disabled.add(key)
                # 尝试用备用令牌替换
                if self._spare_keys:
                    new_key = self._spare_keys.pop(0)
                    self._keys.append(new_key)
                    self._consecutive_fails[new_key] = 0
                    self._queue.put(new_key)
                    self._replaced[key] = new_key
                    return 'replaced'
                return 'disabled'
            return 'ok'

    def key_index(self, key: str) -> int:
        try:
            return self._keys.index(key) + 1
        except ValueError:
            return -1

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._keys) - len(self._disabled)

    @property
    def spare_count(self) -> int:
        with self._lock:
            return len(self._spare_keys)

    def status_line(self) -> str:
        with self._lock:
            parts = []
            for i, k in enumerate(self._keys):
                idx = i + 1
                if k in self._disabled:
                    if k in self._replaced:
                        new_idx = self._keys.index(self._replaced[k]) + 1
                        parts.append(f'K{idx}->K{new_idx}')
                    else:
                        parts.append(f'K{idx}:STOP')
                else:
                    fails = self._consecutive_fails[k]
                    if fails > 0:
                        parts.append(f'K{idx}:F{fails}')
                    else:
                        parts.append(f'K{idx}:OK')
            spare = len(self._spare_keys)
            if spare > 0:
                parts.append(f'备用:{spare}')
            return ' | '.join(parts)

    def __len__(self):
        return len(self._keys)

    def revive_all(self):
        """复活所有已停用的 key, 重置失败计数, 放回队列"""
        with self._lock:
            revived = 0
            for k in list(self._disabled):
                self._disabled.discard(k)
                self._consecutive_fails[k] = 0
                self._queue.put(k)
                revived += 1
            return revived


# ===========================================================================
# 进度追踪
# ===========================================================================

class Progress:
    def __init__(self, total: int):
        self.total = total
        self._lock = threading.Lock()
        self.success = 0
        self.fail = 0
        self.skip = 0
        self.start_time = time.time()

    def inc(self, kind: str):
        with self._lock:
            setattr(self, kind, getattr(self, kind) + 1)

    @property
    def done(self) -> int:
        with self._lock:
            return self.success + self.fail + self.skip

    def summary(self, key_pool: KeyPool) -> str:
        with self._lock:
            done = self.success + self.fail + self.skip
            pct = done / self.total * 100 if self.total else 0
            elapsed = time.time() - self.start_time
            eta_s = (elapsed / done * (self.total - done)) if done > 0 else 0
            eta_m = eta_s / 60
            return (
                f'[{done}/{self.total} ({pct:.1f}%)] '
                f'OK:{self.success} FAIL:{self.fail} SKIP:{self.skip} | '
                f'ETA:{eta_m:.0f}m | {key_pool.status_line()}'
            )


# ===========================================================================
# 单任务处理 (每个 combo 自带原图信息)
# ===========================================================================

def _process_one(
    combo: Dict,
    output_dir: str,
    mask_dir: str,
    key_pool: KeyPool,
    progress: Progress,
    api_url: str, model: str, timeout: int,
    seed: int,
    combo_index: int,
    print_lock: threading.Lock,
    failed_queue: list,
    failed_lock: threading.Lock,
) -> bool:
    img_path = combo['image_path']
    depth_path = combo['depth_path']
    camera_json = combo.get('camera_json')
    img_stem = combo['image_stem']

    fname = make_filename(combo, img_stem)
    out_path = os.path.join(output_dir, f'{fname}.png')
    mask_path = os.path.join(mask_dir, f'{fname}_mask.png')

    def log(msg):
        with print_lock:
            print(msg, flush=True)

    # 跳过已存在
    if os.path.isfile(out_path):
        progress.inc('skip')
        return True

    # 1. 读取原图尺寸 + 编码
    try:
        raw = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f'无法读取: {img_path}')
        img_h, img_w = img.shape[:2]
        _, buf = cv2.imencode('.png', img)
        original_b64 = base64.b64encode(buf).decode('utf-8')
    except Exception as e:
        log(f'  [IMG FAIL] {fname}: {e}')
        progress.inc('fail')
        with failed_lock:
            failed_queue.append(combo)
        return False

    # 2. 生成掩码
    try:
        mask = render_rain_mask(
            depth_path=depth_path,
            camera_json_path=camera_json,
            rain_rate=combo['rain_rate'],
            wind_speed=combo['wind_speed'],
            wind_direction=combo['wind_direction'],
            image_width=img_w,
            image_height=img_h,
            seed=seed + combo_index,
            **MASK_DEFAULTS,
        )
        cv2.imencode('.png', mask)[1].tofile(mask_path)
    except Exception as e:
        log(f'  [MASK FAIL] {fname}: {e}')
        progress.inc('fail')
        with failed_lock:
            failed_queue.append(combo)
        return False

    # 3. 调用 Gemini
    prompt = PROMPT_TEMPLATES[combo['template']]

    if key_pool.active_count == 0:
        log(f'  [ALL KEYS DOWN] {fname}')
        progress.inc('fail')
        with failed_lock:
            failed_queue.append(combo)
        return False

    key = key_pool.acquire()
    if key is None:
        progress.inc('fail')
        with failed_lock:
            failed_queue.append(combo)
        return False

    try:
        t0 = time.time()
        result = call_gemini(
            original_b64=original_b64,
            mask_image=mask,
            prompt=prompt,
            api_key=key,
            api_url=api_url,
            model=model,
            timeout=timeout,
        )
        cv2.imencode('.png', result)[1].tofile(out_path)
        elapsed = time.time() - t0
        key_pool.report_success(key)
        progress.inc('success')
        log(f'  [{progress.done}/{progress.total}] {fname} OK ({elapsed:.0f}s) | {progress.summary(key_pool)}')
        return True

    except RuntimeError as e:
        status = key_pool.report_fail(key)
        progress.inc('fail')
        extra = ''
        if status == 'replaced':
            extra = ' [KEY REPLACED]'
        elif status == 'disabled':
            extra = ' [KEY DISABLED]'
        log(f'  [{progress.done}/{progress.total}] {fname} FAIL: {e}{extra}')
        with failed_lock:
            failed_queue.append(combo)
        return False
    finally:
        key_pool.release(key)


# ===========================================================================
# 主流水线
# ===========================================================================

def load_env_keys() -> Tuple[List[str], str]:
    """从 .env 文件加载 API keys 和 URL"""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    keys = []
    url = DEFAULT_API_URL
    if os.path.isfile(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('API_KEYS='):
                    raw = line[len('API_KEYS='):]
                    keys = [k.strip() for k in raw.split(',') if k.strip()]
                elif line.startswith('API_URL='):
                    url = line[len('API_URL='):].strip()
    return keys, url


# 30 个备用令牌
SPARE_KEYS = [
    'sk-GTCYkx5FmCFlNvtIK5CIvxLfFhvyZfuqUqwI2ukp2t4vJn54',
    'sk-KfC80ixRiIfyeRFLHvrS1177VX77qrmCtwI6At8rtnpn8uru',
    'sk-W9qmGRwHWDOofZmcBX3UjSyrximrBuaOKLQSa6VxV6Dwtxw5',
    'sk-347XUItHA4XqwvJW96LnUwOCdxLBkT3WFhAG5AxEv9ch6Ara',
    'sk-J7nRQz95rXSWsubiu5QbSx7IiEGba2qdadbNdoTT7ClX8XFd',
    'sk-MsOCtq3fxBwoPutye6XHDLkcfvR7ZAOULwsXpm6wUBFWbPxZ',
    'sk-ro47HphfRaDhs1PJYGX96RNNxwEmC3R71LAdoDVhNb0thqq5',
    'sk-ZGT7GBnxkXPeK7cT6IfkhhjZ6KfFIlHdSpKuLwQ02x2IQ8Ei',
    'sk-2WT1dV05fGFcYgulSm4pvf5aUvpmSnr2ZGuNnYhk7Sf2F91Y',
    'sk-gxUS5B1mJTcYpZnpTDT3eeBa2Tu7ny5Y96GfCB9unZH6rqB2',
    'sk-ZWGMdmTLQpAgJOcn2Av3bEQT3RDmDgkL9DmEoAoQJyNEz6Fx',
    'sk-8bRbPX59yBacTPG2QcoBOlihUBGb5eyEmE2Y1o5XHI23rwR3',
    'sk-uncYFUyrWQAw6RdRWSG9QZSe3tEBviWYnQRjEZFL1avQT44k',
    'sk-Jw038jhrw78A1nqh7mUMSvxMgzu495f5pBQvqIsgNZO5RNB6',
    'sk-8qtImZiSF23hIbLdraJcW9JAC5febXkL5LVvNg73Htv48sXc',
    'sk-ZsMoRlKzcdqmeJui3nFro1vgddvdlabw1EPyhb05ZGmY1QAs',
    'sk-ukbFj3DKAsqGnLS5ejDfAlOkAGxSVcTrq6rpYR8wlGxsl5fy',
    'sk-lfNoTSi6DYtFgiXHFzTev7FsDkRuvNaR6U8znu2km0EgjlQg',
    'sk-ngQZXsYR5B822uUfqrA0rzoIej5cmUKRiz7jp5Uv1MKhn3m9',
    'sk-VBm7jhuTd014p6TWwsbkBzSYjAxwn9ehRyPhaVlGsjcQ46fk',
    'sk-BxfxZ5LrGx5elJjjNqhgs1vjz1qAsqKIAYLgTjpt6trdyKDz',
    'sk-JkPSrz7QRh09hCPoHkuH1UvHVyCDKGYzOPwFwAtCgXQReGzZ',
    'sk-BYFYC6JEiGgRFT7DZWofRL0R09SUCFXz56kezAgYpYwnvghE',
    'sk-h3WWJBEfrMCkKYOd8GsO6PKi9wDtV1FJcPRwmUz2ePXNT04K',
    'sk-7iPawFPDYYQfDnindZjGKCoK8ECZEkrvhSbcamflnE3F9j0I',
    'sk-hfUhDociKUM6HR3z7aZHFOHqDDLCxJvOXoGP3C2otP2Vguv5',
    'sk-7PopcdB0xLlEsmxsaI4HnFKrqf3cKPFhVX2e4GSR7gQcCq6R',
    'sk-27G1mtGgMfojqjBBL8OhmHIPBLPm1l5xFjJvX2ey7n9fmCIN',
    'sk-J7q9tX5ZiJgQCN5QUFVN0f1SZGknWGt32Asrx5m18AwdOWix',
    'sk-iPlb8RF5B0AWx3ZueH7yz0TuqASKocjQHprk3FNKO1SEn3Al',
]


def run_pipeline(
    image_dir: str = DEFAULT_IMAGE_DIR,
    depth_dir: str = DEFAULT_DEPTH_DIR,
    camera_dir: str = DEFAULT_CAMERA_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    mask_dir: str = DEFAULT_MASK_DIR,
    test_n: int = 0,
    seed: int = 42,
    model: str = DEFAULT_MODEL,
    timeout: int = DEFAULT_TIMEOUT,
):
    # 加载配置
    api_keys, api_url = load_env_keys()
    if not api_keys:
        print('[错误] .env 中未找到 API_KEYS')
        return

    # 收集所有原图
    print('扫描 Cityscapes 图像...', flush=True)
    all_images = collect_cityscapes_images(image_dir)
    print(f'  找到 {len(all_images)} 张原图')

    # 构建参数矩阵
    all_combos = build_param_matrix()

    if test_n > 0:
        combos = sample_pilot_combos(all_combos, test_n)
        mode_str = f'先导测试 ({len(combos)} 张)'
    else:
        combos = all_combos
        mode_str = f'全量生成 ({len(combos)} 张)'

    if len(all_images) < len(combos):
        print(f'[警告] 原图数 ({len(all_images)}) < 参数组合数 ({len(combos)}), 将循环使用')

    # 为每个 combo 分配原图 (按顺序, 不够则循环)
    no_depth_count = 0
    for i, combo in enumerate(combos):
        img_path = all_images[i % len(all_images)]
        basename = os.path.splitext(os.path.basename(img_path))[0]
        stem = _strip_suffix(basename)

        depth_path = find_matching_depth(img_path, image_dir, depth_dir)
        camera_json = find_matching_camera(img_path, image_dir, camera_dir) if camera_dir else None

        if depth_path is None:
            no_depth_count += 1

        combo['image_path'] = img_path
        combo['depth_path'] = depth_path
        combo['camera_json'] = camera_json
        combo['image_stem'] = stem

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    print('=' * 70)
    print('PhysicRain 参数矩阵批量生成 v2')
    print('=' * 70)
    print(f'模式:         {mode_str}')
    print(f'原图目录:     {image_dir} ({len(all_images)} 张)')
    print(f'深度图目录:   {depth_dir}')
    print(f'相机目录:     {camera_dir}')
    print(f'输出目录:     {output_dir}')
    print(f'掩码目录:     {mask_dir}')
    print(f'API Keys:     {len(api_keys)} 主 + {len(SPARE_KEYS)} 备用')
    print(f'模型:         {model}')
    print(f'参数总数:     {len(all_combos)} 组')
    print(f'本次生成:     {len(combos)} 组')
    if no_depth_count > 0:
        print(f'[警告] {no_depth_count} 个组合未找到深度图')
    print('=' * 70)

    template_counts = StdCounter(c['template'] for c in combos)
    for t in sorted(template_counts):
        print(f'  模板 T{t}: {template_counts[t]} 组')
    print('=' * 70)

    # 初始化
    key_pool = KeyPool(api_keys, spare_keys=list(SPARE_KEYS))
    max_workers = len(api_keys)
    progress = Progress(len(combos))
    print_lock = threading.Lock()
    failed_queue = []
    failed_lock = threading.Lock()

    round_num = 1
    current_combos = combos
    max_revive_cycles = 10  # 最多循环复活 10 次, 防止无限循环
    revive_count = 0

    while current_combos:
        print(f'\n--- 第 {round_num} 轮 ({len(current_combos)} 个任务) ---')

        if key_pool.active_count == 0:
            if revive_count >= max_revive_cycles:
                print(f'[终止] 已循环复活 {max_revive_cycles} 次, 停止')
                break
            # 循环复活: 等待 60s 后重新启用所有 key
            revive_count += 1
            print(f'[循环复活 {revive_count}/{max_revive_cycles}] 所有 Key 已停用, 等待 60s 后复活全部 Key...')
            time.sleep(60)
            n_revived = key_pool.revive_all()
            print(f'  已复活 {n_revived} 个 Key, 继续运行')

        workers = min(max_workers, key_pool.active_count)
        if workers <= 0:
            print('[终止] 无可用 Key')
            break
        failed_queue = []

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for idx, combo in enumerate(current_combos):
                f = executor.submit(
                    _process_one,
                    combo,
                    output_dir, mask_dir,
                    key_pool, progress,
                    api_url, model, timeout,
                    seed, idx,
                    print_lock,
                    failed_queue, failed_lock,
                )
                futures.append(f)

            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    with print_lock:
                        print(f'[线程异常] {e}')

        if not failed_queue:
            break

        round_num += 1
        current_combos = failed_queue
        print(f'\n重试 {len(current_combos)} 个失败任务...')

    # 最终报告
    remaining_fails = len(failed_queue)
    elapsed = time.time() - progress.start_time
    print('\n' + '=' * 70)
    print('生成完成!')
    print(f'  成功: {progress.success}')
    print(f'  失败: {remaining_fails}')
    print(f'  跳过: {progress.skip}')
    print(f'  总耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)')
    if progress.success > 0:
        print(f'  平均: {elapsed/progress.success:.1f}s/张')
    print(f'  输出: {output_dir}')
    print(f'  掩码: {mask_dir}')
    print('=' * 70)

    if failed_queue:
        fail_log = os.path.join(output_dir, '_failed.json')
        with open(fail_log, 'w', encoding='utf-8') as f:
            json.dump([{k: v for k, v in c.items() if k != 'image_path'} for c in failed_queue],
                      f, ensure_ascii=False, indent=2)
        print(f'  失败任务已保存到: {fail_log}')


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PhysicRain 参数矩阵批量生成 v2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 先导测试 20 张
  python rain_batch_matrix.py --test 20

  # 全量生成 3440 张 (每张用不同原图)
  python rain_batch_matrix.py

  # 自定义目录
  python rain_batch_matrix.py --image_dir path/to/images --depth_dir path/to/depths
        """
    )

    parser.add_argument('--image_dir', default=DEFAULT_IMAGE_DIR, help='Cityscapes 原图目录')
    parser.add_argument('--depth_dir', default=DEFAULT_DEPTH_DIR, help='深度图目录')
    parser.add_argument('--camera_dir', default=DEFAULT_CAMERA_DIR, help='相机参数目录')
    parser.add_argument('--output', default=DEFAULT_OUTPUT_DIR, help='合成图输出目录')
    parser.add_argument('--masks', default=DEFAULT_MASK_DIR, help='掩码输出目录')
    parser.add_argument('--test', type=int, default=0, help='先导测试张数 (0=全量)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--model', default=DEFAULT_MODEL, help='Gemini 模型名称')
    parser.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT, help='API 超时 (秒)')

    args = parser.parse_args()

    run_pipeline(
        image_dir=args.image_dir,
        depth_dir=args.depth_dir,
        camera_dir=args.camera_dir,
        output_dir=args.output,
        mask_dir=args.masks,
        test_n=args.test,
        seed=args.seed,
        model=args.model,
        timeout=args.timeout,
    )
