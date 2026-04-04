"""
============================================================================
PhysicRain 批量雨天场景合成流水线
============================================================================

流程:
  深度图 ──→ rain_renderer ──→ 雨滴掩码 ──→┐
  原始图像 ────────────────────────────────→├──→ Gemini API ──→ 雨天合成图
  提示词模板 ──────────────────────────────→┘

依赖:
  - rain_renderer.py (雨滴掩码生成)
  - requests (HTTP 请求)
  - numpy, cv2 (图像处理)

日期: 2026-03-27
============================================================================
"""

import os
import sys
import glob
import time
import base64
import json
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import numpy as np
import cv2
import requests

from rain_renderer import render_rain_mask


# ===========================================================================
# 配置
# ===========================================================================

DEFAULT_API_URL = 'https://api.aipaibox.com/v1/chat/completions'
DEFAULT_MODEL = 'gemini-3.1-flash-image-preview'
DEFAULT_TIMEOUT = 300  # 秒，图像生成较慢

DEFAULT_PROMPT_TEMPLATE = (
    '一张具有逼真摄影质感的图像，场景与第一张图片一致。'
    '该场景如今设定在一场{rain_desc}中，降雨量约为 {rain_rate} 毫米/小时。'
    '降雨的具体分布、角度情况，均参照所提供的第二张雨水蒙版图像中的样式。'
    '整体光照昏暗阴沉，路面有积水，能见度明显降低，尤其是在远处。'
    '仅仅是分布如蒙版所示而非简单叠加，让雨滴更自然一些，融入到下雨的环境之中。'
    '请直接输出修改后的图像。'
)


def get_rain_description(rain_rate: float) -> str:
    """根据降雨量返回中文描述"""
    if rain_rate < 5:
        return '小雨'
    elif rain_rate < 15:
        return '中雨'
    elif rain_rate < 30:
        return '大雨'
    elif rain_rate < 60:
        return '暴雨'
    elif rain_rate < 100:
        return '大暴雨'
    else:
        return '特大暴雨'


# ===========================================================================
# 图像编码工具
# ===========================================================================

def image_to_base64(image_path: str) -> str:
    """读取图像文件并编码为 base64（支持中文路径）"""
    raw = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f'无法读取图像: {image_path}')
    _, buf = cv2.imencode('.png', img)
    return base64.b64encode(buf).decode('utf-8')


def numpy_to_base64(img: np.ndarray) -> str:
    """将 numpy 图像数组编码为 base64"""
    _, buf = cv2.imencode('.png', img)
    return base64.b64encode(buf).decode('utf-8')


def base64_to_image(b64_str: str) -> np.ndarray:
    """将 base64 字符串解码为 numpy 图像数组"""
    # 去掉可能的前缀 data:image/...;base64,
    if ',' in b64_str:
        b64_str = b64_str.split(',', 1)[1]
    img_bytes = base64.b64decode(b64_str)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def _strip_known_stem_suffix(stem: str) -> str:
    for suffix in ('_leftImg8bit', '_rightImg8bit', '_disparity', '_depth', '_camera'):
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def _resolve_cityscapes_relative_dir(image_dir: str, img_path: str) -> str:
    rel_dir = os.path.relpath(os.path.dirname(img_path), image_dir)
    return '' if rel_dir == '.' else rel_dir


def _find_depth_and_camera(
    img_path: str,
    image_dir: str,
    depth_dir: str,
    camera_dir: str = None,
):
    basename = os.path.splitext(os.path.basename(img_path))[0]
    stem = _strip_known_stem_suffix(basename)
    rel_dir = _resolve_cityscapes_relative_dir(image_dir, img_path)

    depth_candidates = []
    for ext in ('.png', '.jpg'):
        depth_candidates.extend([
            f'{basename}_depth_u16{ext}',
            f'{basename}_depth{ext}',
            f'{basename}{ext}',
            f'{stem}_depth_u16{ext}',
            f'{stem}_depth{ext}',
            f'{stem}_disparity_depth{ext}',
        ])

    depth_path = None
    for name in depth_candidates:
        candidate = os.path.join(depth_dir, rel_dir, name)
        if os.path.isfile(candidate):
            depth_path = candidate
            break
        candidate = os.path.join(depth_dir, name)
        if os.path.isfile(candidate):
            depth_path = candidate
            break

    camera_json_path = None
    if camera_dir:
        camera_candidates = [
            f'{stem}_camera.json',
            f'{basename}_camera.json',
            f'{basename}.json',
        ]
        for name in camera_candidates:
            candidate = os.path.join(camera_dir, rel_dir, name)
            if os.path.isfile(candidate):
                camera_json_path = candidate
                break
            candidate = os.path.join(camera_dir, name)
            if os.path.isfile(candidate):
                camera_json_path = candidate
                break

    return depth_path, camera_json_path


# ===========================================================================
# Gemini API 调用
# ===========================================================================

def call_gemini_image_edit(
    original_image_path: str,
    mask_image: np.ndarray,
    prompt: str,
    api_key: str,
    api_url: str = DEFAULT_API_URL,
    model: str = DEFAULT_MODEL,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = 2,
) -> np.ndarray:
    """
    调用 Gemini API 合成雨天图像

    Args:
        original_image_path: 原始晴天图像路径
        mask_image:          雨滴掩码 (numpy array)
        prompt:              提示词
        api_key:             API 密钥
        api_url:             API 端点 URL
        model:               模型名称
        timeout:             超时时间 (秒)
        max_retries:         最大重试次数

    Returns:
        合成的雨天图像 (numpy array, BGR)

    Raises:
        RuntimeError: API 调用失败或未返回图像
    """
    b64_original = image_to_base64(original_image_path)
    b64_mask = numpy_to_base64(mask_image)

    payload = {
        'model': model,
        'messages': [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:image/png;base64,{b64_original}'
                        }
                    },
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:image/png;base64,{b64_mask}'
                        }
                    },
                    {
                        'type': 'text',
                        'text': prompt
                    }
                ]
            }
        ],
        'max_tokens': 4096,
    }

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }

    last_error = None
    for attempt in range(1, max_retries + 2):
        try:
            print(f'    API 调用 (尝试 {attempt}/{max_retries + 1})...', end='', flush=True)
            t0 = time.time()

            resp = requests.post(
                api_url, headers=headers, json=payload, timeout=timeout
            )

            elapsed = time.time() - t0
            print(f' {elapsed:.1f}s', end='', flush=True)

            if resp.status_code != 200:
                last_error = f'HTTP {resp.status_code}: {resp.text[:200]}'
                print(f' 失败: {last_error}')
                if attempt <= max_retries:
                    time.sleep(5)
                continue

            data = resp.json()

            # 解析返回内容
            if 'choices' not in data or not data['choices']:
                last_error = f'返回无 choices: {json.dumps(data, ensure_ascii=False)[:200]}'
                print(f' 失败: {last_error}')
                continue

            content = data['choices'][0].get('message', {}).get('content', '')

            # 情况 1: content 是 list（包含图像）
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        if part.get('type') == 'image_url':
                            url = part.get('image_url', {}).get('url', '')
                            if 'base64,' in url:
                                img = base64_to_image(url)
                                if img is not None:
                                    print(' 成功')
                                    return img

            # 情况 2: content 是字符串（可能包含 base64 图像数据）
            if isinstance(content, str):
                # 检查是否有 base64 图像嵌入
                if 'data:image' in content and 'base64,' in content:
                    start = content.index('base64,') + 7
                    # 找到 base64 数据的结束位置
                    end = content.index('"', start) if '"' in content[start:] else len(content)
                    b64_data = content[start:end]
                    img = base64_to_image(b64_data)
                    if img is not None:
                        print(' 成功')
                        return img

                # 纯文字回复，模型没有生成图像
                last_error = f'模型返回文字而非图像: {content[:150]}'
                print(f' 无图像')

            if attempt <= max_retries:
                print(f'    等待 5 秒后重试...')
                time.sleep(5)

        except requests.exceptions.Timeout:
            last_error = f'超时 ({timeout}s)'
            print(f' 超时')
            if attempt <= max_retries:
                time.sleep(5)

        except (requests.exceptions.ProxyError,
                requests.exceptions.ConnectionError) as e:
            # 代理/连接错误：本地网络问题，重试无意义，直接放弃
            last_error = f'代理/连接错误: {e}'
            print(f' 代理断开')
            raise RuntimeError(f'代理连接失败 (不重试): {e}')

        except Exception as e:
            last_error = str(e)
            print(f' 异常: {last_error}')
            if attempt <= max_retries:
                time.sleep(5)

    raise RuntimeError(f'API 调用失败 ({max_retries + 1} 次尝试): {last_error}')


# ===========================================================================
# Key 轮询池（线程安全）
# ===========================================================================

class KeyPool:
    """
    线程安全的 API Key 轮询池

    多个线程并发获取 key，每个 key 同一时间只被一个线程使用。
    用完归还后可被下一个线程获取。
    """
    def __init__(self, keys):
        if isinstance(keys, str):
            keys = [keys]
        self._keys = list(keys)
        self._queue = Queue()
        for k in self._keys:
            self._queue.put(k)

    def acquire(self):
        """获取一个可用的 key（若全忙则阻塞等待）"""
        return self._queue.get()

    def release(self, key):
        """归还 key"""
        self._queue.put(key)

    def __len__(self):
        return len(self._keys)


# ===========================================================================
# 线程安全计数器
# ===========================================================================

class Counter:
    def __init__(self):
        self._lock = threading.Lock()
        self.success = 0
        self.fail = 0
        self.skip = 0

    def inc_success(self):
        with self._lock:
            self.success += 1

    def inc_fail(self):
        with self._lock:
            self.fail += 1

    def inc_skip(self):
        with self._lock:
            self.skip += 1


# ===========================================================================
# 单张处理任务
# ===========================================================================

def _process_single_image(
    idx, total, img_path, image_dir, depth_dir, camera_dir, output_dir,
    key_pool, rain_params, prompt, api_url, model,
    timeout, seed, skip_existing, mask_output_dir, counter,
    print_lock,
):
    """处理单张图像：掩码生成 + API 合成 + 保存"""
    basename = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(output_dir, f'{basename}_rain.png')

    def log(msg):
        with print_lock:
            print(msg, flush=True)

    # 跳过已存在
    if skip_existing and os.path.isfile(out_path):
        log(f'[{idx+1}/{total}] {basename} - 已存在，跳过')
        counter.inc_skip()
        return

    # 1. 查找深度图 / 相机参数
    depth_path, camera_json_path = _find_depth_and_camera(
        img_path=img_path,
        image_dir=image_dir,
        depth_dir=depth_dir,
        camera_dir=camera_dir,
    )

    # 2. 生成掩码
    img_seed = seed + idx if seed is not None else None
    try:
        raw = np.fromfile(img_path, dtype=np.uint8)
        orig = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        h, w = orig.shape[:2]

        mask = render_rain_mask(
            depth_path=depth_path,
            camera_json_path=camera_json_path,
            rain_rate=rain_params['rain_rate'],
            wind_speed=rain_params['wind_speed'],
            wind_angle=rain_params.get('wind_angle', 30.0),
            wind_direction=rain_params.get('wind_direction'),
            exposure_time=rain_params['exposure_time'],
            turbulence_deg=rain_params['turbulence_deg'],
            depth_scale=rain_params['depth_scale'],
            image_width=w, image_height=h,
            seed=img_seed,
        )
        coverage = np.sum(mask > 0) / mask.size * 100

        if mask_output_dir:
            mask_path = os.path.join(mask_output_dir, f'{basename}_mask.png')
            cv2.imwrite(mask_path, mask)

    except Exception as e:
        log(f'[{idx+1}/{total}] {basename} - 掩码失败: {e}')
        counter.inc_fail()
        return

    # 3. 获取 key，调用 API
    key = key_pool.acquire()
    key_id = f'key{list(range(len(key_pool)))[-1]}'  # 简化标识
    try:
        log(f'[{idx+1}/{total}] {basename} - 掩码 {coverage:.0f}%, 调用 API...')
        t0 = time.time()

        result = call_gemini_image_edit(
            original_image_path=img_path,
            mask_image=mask,
            prompt=prompt,
            api_key=key,
            api_url=api_url,
            model=model,
            timeout=timeout,
        )

        cv2.imwrite(out_path, result)
        elapsed = time.time() - t0
        log(f'[{idx+1}/{total}] {basename} - 成功 ({elapsed:.0f}s)')
        counter.inc_success()

    except RuntimeError as e:
        log(f'[{idx+1}/{total}] {basename} - 失败: {e}')
        counter.inc_fail()
    finally:
        key_pool.release(key)


# ===========================================================================
# 批量流水线（支持多 key 并发）
# ===========================================================================

def batch_synthesize(
    image_dir: str,
    depth_dir: str,
    output_dir: str,
    api_keys,
    camera_dir: str = None,
    rain_rate: float = 70.0,
    wind_speed: float = 8.0,
    wind_angle: float = 30.0,
    wind_direction=None,
    exposure_time: float = 1/25,
    turbulence_deg: float = 6.0,
    depth_scale: float = 100.0,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    api_url: str = DEFAULT_API_URL,
    model: str = DEFAULT_MODEL,
    timeout: int = DEFAULT_TIMEOUT,
    seed: int = 42,
    skip_existing: bool = True,
    mask_output_dir: str = None,
    max_workers: int = None,
):
    """
    批量合成雨天场景图像（支持多 key 并发）

    Args:
        api_keys:        API 密钥，str 或 list[str]
                         多个 key 时自动并发调用
        max_workers:     最大并发数（默认 = key 数量）
        其他参数同前
    """
    # 规范化 keys
    if isinstance(api_keys, str):
        api_keys = [api_keys]
    key_pool = KeyPool(api_keys)

    if max_workers is None:
        max_workers = len(api_keys)
    max_workers = min(max_workers, len(api_keys))

    os.makedirs(output_dir, exist_ok=True)
    if mask_output_dir:
        os.makedirs(mask_output_dir, exist_ok=True)

    image_files = sorted(
        glob.glob(os.path.join(image_dir, '**', '*.png'), recursive=True) +
        glob.glob(os.path.join(image_dir, '**', '*.jpg'), recursive=True)
    )
    if not image_files:
        print(f'[错误] 在 {image_dir} 中未找到图像文件')
        return

    rain_desc = get_rain_description(rain_rate)
    prompt = prompt_template.format(rain_rate=rain_rate, rain_desc=rain_desc)

    rain_params = {
        'rain_rate': rain_rate, 'wind_speed': wind_speed,
        'wind_angle': wind_angle, 'wind_direction': wind_direction,
        'exposure_time': exposure_time, 'turbulence_deg': turbulence_deg,
        'depth_scale': depth_scale,
    }

    print('=' * 60)
    print('PhysicRain 批量雨天合成流水线')
    print('=' * 60)
    print(f'图像目录:      {image_dir} ({len(image_files)} 张)')
    print(f'深度图目录:    {depth_dir}')
    if camera_dir:
        print(f'相机目录:      {camera_dir}')
    print(f'输出目录:      {output_dir}')
    print(f'降雨量:        {rain_rate} mm/h ({rain_desc})')
    print(f'API Keys:      {len(api_keys)} 个')
    print(f'并发数:        {max_workers}')
    print(f'模型:          {model}')
    print('=' * 60)

    counter = Counter()
    print_lock = threading.Lock()
    total = len(image_files)

    t_start = time.time()

    if max_workers <= 1:
        # 单线程模式
        for idx, img_path in enumerate(image_files):
            _process_single_image(
                idx, total, img_path,
                image_dir, depth_dir, camera_dir, output_dir,
                key_pool, rain_params, prompt, api_url, model,
                timeout, seed, skip_existing, mask_output_dir,
                counter, print_lock,
            )
    else:
        # 多线程并发模式
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for idx, img_path in enumerate(image_files):
                f = executor.submit(
                    _process_single_image,
                    idx, total, img_path, image_dir, depth_dir, camera_dir, output_dir,
                    key_pool, rain_params, prompt, api_url, model,
                    timeout, seed, skip_existing, mask_output_dir,
                    counter, print_lock,
                )
                futures.append(f)

            # 等待所有完成
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    with print_lock:
                        print(f'[异常] 线程异常: {e}')

    elapsed_total = time.time() - t_start
    print('\n' + '=' * 60)
    print(f'完成! 成功: {counter.success}  失败: {counter.fail}  跳过: {counter.skip}')
    print(f'总耗时: {elapsed_total:.1f}s  ({elapsed_total/60:.1f}min)')
    if counter.success > 0:
        print(f'平均: {elapsed_total/counter.success:.1f}s/张')
    print(f'输出目录: {output_dir}')
    print('=' * 60)


# ===========================================================================
# 命令行入口
# ===========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PhysicRain 批量雨天场景合成流水线',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单 key
  python rain_pipeline.py --image_dir test_dataset --depth_dir test_depth --api_keys YOUR_KEY

  # 多 key 并发 (3x 加速)
  python rain_pipeline.py --image_dir test_dataset --depth_dir test_depth \\
    --api_keys key1 key2 key3

  # 指定并发数
  python rain_pipeline.py --image_dir imgs --depth_dir depths \\
    --api_keys key1 key2 key3 key4 --workers 3
        """
    )

    parser.add_argument('--image_dir', required=True, help='原始晴天图像目录')
    parser.add_argument('--depth_dir', required=True, help='深度图目录')
    parser.add_argument('--camera_dir', default=None, help='相机标定目录 (Cityscapes camera)')
    parser.add_argument('--output_dir', default='rain_output', help='合成结果输出目录')
    parser.add_argument('--api_keys', nargs='+', required=True,
                        help='API 密钥 (支持多个，空格分隔)')
    parser.add_argument('--workers', type=int, default=None,
                        help='最大并发数 (默认 = key 数量)')
    parser.add_argument('--api_url', default=DEFAULT_API_URL, help='API 端点 URL')
    parser.add_argument('--model', default=DEFAULT_MODEL, help='模型名称')

    parser.add_argument('--rain_rate', type=float, default=70.0, help='降雨量 (mm/h)')
    parser.add_argument('--wind_speed', type=float, default=8.0, help='风速 (m/s)')
    parser.add_argument('--wind_angle', type=float, default=30.0, help='风向角 (度，旧接口)')
    parser.add_argument('--wind_direction', type=float, default=None,
                        help='罗盘风向 (度，0=北风迎面，90=东风)')
    parser.add_argument('--exposure', type=float, default=1/25, help='快门时间 (秒)')
    parser.add_argument('--turbulence', type=float, default=6.0, help='湍流强度 (度)')
    parser.add_argument('--depth_scale', type=float, default=100.0, help='深度缩放 (米)')

    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT, help='API 超时 (秒)')
    parser.add_argument('--no_skip', action='store_true', help='不跳过已存在的输出文件')
    parser.add_argument('--save_masks', default=None, help='掩码保存目录')
    parser.add_argument('--prompt', default=None, help='自定义提示词 (覆盖默认模板)')

    args = parser.parse_args()

    prompt_template = args.prompt if args.prompt else DEFAULT_PROMPT_TEMPLATE

    batch_synthesize(
        image_dir=args.image_dir,
        depth_dir=args.depth_dir,
        camera_dir=args.camera_dir,
        output_dir=args.output_dir,
        api_keys=args.api_keys,
        rain_rate=args.rain_rate,
        wind_speed=args.wind_speed,
        wind_angle=args.wind_angle,
        wind_direction=args.wind_direction,
        exposure_time=args.exposure,
        turbulence_deg=args.turbulence,
        depth_scale=args.depth_scale,
        prompt_template=prompt_template,
        api_url=args.api_url,
        model=args.model,
        timeout=args.timeout,
        seed=args.seed,
        skip_existing=not args.no_skip,
        mask_output_dir=args.save_masks,
        max_workers=args.workers,
    )
