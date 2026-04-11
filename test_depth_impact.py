"""
测试增强遮挡效果: 有深度图 vs 无深度图对比
"""
import os, sys, numpy as np, cv2
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
COMMON = dict(
    rain_rate=5.0, wind_speed=0.5, exposure_time=0.02,
    turbulence_deg=1.0, image_width=2048, image_height=1024,
    streak_softness=0.5, seed=SEED,
)

print('渲染: 有深度图 ...')
mask_depth = render_rain_mask(depth_path=DEPTH_PATH, **COMMON)
cv2.imencode('.png', mask_depth)[1].tofile(os.path.join(OUT_DIR, 'occlusion_with_depth.png'))

print('渲染: 无深度图 ...')
mask_no = render_rain_mask(depth_path=None, depth_mode='gradient', **COMMON)
cv2.imencode('.png', mask_no)[1].tofile(os.path.join(OUT_DIR, 'occlusion_no_depth.png'))

H = mask_depth.shape[0]
top = slice(0, H // 4)
bot = slice(3 * H // 4, H)

print()
print('=' * 60)
for name, m in [('有深度图', mask_depth), ('无深度图', mask_no)]:
    t_cov = np.sum(m[top] > 0) / m[top].size * 100
    b_cov = np.sum(m[bot] > 0) / m[bot].size * 100
    t_mean = m[top][m[top] > 0].mean() if np.any(m[top] > 0) else 0
    b_mean = m[bot][m[bot] > 0].mean() if np.any(m[bot] > 0) else 0
    print(f'[{name}] 上1/4: 覆盖={t_cov:.1f}% 亮={t_mean:.0f} | 下1/4: 覆盖={b_cov:.1f}% 亮={b_mean:.0f}')

diff = cv2.absdiff(mask_depth, mask_no)
print(f'差异率: {np.sum(diff > 10) / diff.size * 100:.1f}%')
cv2.imencode('.png', diff)[1].tofile(os.path.join(OUT_DIR, 'occlusion_diff.png'))

# 局部放大 - 下半部分 (近处遮挡最明显的区域)
crop = mask_depth[512:1024, 768:1280]
crop_no = mask_no[512:1024, 768:1280]
cv2.imencode('.png', crop)[1].tofile(os.path.join(OUT_DIR, 'occlusion_crop_depth.png'))
cv2.imencode('.png', crop_no)[1].tofile(os.path.join(OUT_DIR, 'occlusion_crop_no.png'))

print(f'\n保存到: {OUT_DIR}/occlusion_*.png')
