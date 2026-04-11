"""
测试高斯横截面渲染 + streak_softness 对雨丝透明感的影响
"""
import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from rain_renderer import render_rain_mask

DEPTH_PATH = os.path.join(
    os.path.dirname(__file__),
    'dataset', 'depth_trainvaltest', 'gray16', 'test', 'berlin',
    'berlin_000000_000019_leftImg8bit_depth_u16.png'
)
OUT_DIR = os.path.join(os.path.dirname(__file__), 'test_output')
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
# 新默认参数
COMMON = dict(
    depth_path=DEPTH_PATH,
    rain_rate=5.0,
    wind_speed=0.5,
    exposure_time=0.02,
    turbulence_deg=1.0,
    image_width=2048,
    image_height=1024,
    seed=SEED,
)

# ============================================================
# 测试 1: 不同 softness (高斯横截面宽度)
# ============================================================
print('=' * 60)
print('测试 1: streak_softness 对比')
print('=' * 60)

softness_values = [0.0, 0.3, 0.5, 0.7, 1.0]

for s in softness_values:
    print(f'  softness={s:.1f} ...', end=' ', flush=True)
    mask = render_rain_mask(**COMMON, streak_softness=s)
    fname = f'softness_{s:.1f}.png'
    cv2.imencode('.png', mask)[1].tofile(os.path.join(OUT_DIR, fname))
    coverage = np.sum(mask > 0) / mask.size * 100
    mean_bright = mask[mask > 0].mean() if np.any(mask > 0) else 0
    print(f'覆盖率={coverage:.1f}%, 均值亮度={mean_bright:.0f}, 保存: {fname}')

# ============================================================
# 测试 2: 局部放大 512x512 (细节对比)
# ============================================================
print()
print('=' * 60)
print('测试 2: 局部放大 512x512')
print('=' * 60)

crop_x, crop_y = 768, 256
crop_w, crop_h = 512, 512

for s in [0.0, 0.5, 1.0]:
    print(f'  softness={s:.1f} 裁切 ...', end=' ', flush=True)
    mask = render_rain_mask(**COMMON, streak_softness=s)
    crop = mask[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
    fname = f'crop_softness_{s:.1f}.png'
    cv2.imencode('.png', crop)[1].tofile(os.path.join(OUT_DIR, fname))
    print(f'保存: {fname}')

# ============================================================
# 测试 3: 不同降雨强度 + softness=0.5
# ============================================================
print()
print('=' * 60)
print('测试 3: 不同降雨强度 (softness=0.5)')
print('=' * 60)

for rate in [2.0, 5.0, 15.0, 30.0]:
    print(f'  rain_rate={rate:.0f}mm/h ...', end=' ', flush=True)
    mask = render_rain_mask(
        **{**COMMON, 'rain_rate': rate},
        streak_softness=0.5,
    )
    fname = f'rain_{rate:.0f}mmh_soft05.png'
    cv2.imencode('.png', mask)[1].tofile(os.path.join(OUT_DIR, fname))
    coverage = np.sum(mask > 0) / mask.size * 100
    print(f'覆盖率={coverage:.1f}%, 保存: {fname}')

print()
print(f'所有测试图已保存到: {OUT_DIR}')
print('完成!')
