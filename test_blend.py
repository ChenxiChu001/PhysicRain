"""
测试 harmonic_blend 在 1.0 ~ 0.75 之间的 10 个细分梯度
"""
import os, sys, numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__))
from rain_renderer import render_rain_mask, load_and_process_depth

DEPTH_PATH = os.path.join(
    os.path.dirname(__file__),
    'dataset', 'depth_trainvaltest', 'gray16', 'test', 'berlin',
    'berlin_000000_000019_leftImg8bit_depth_u16.png'
)
OUT_DIR = os.path.join(os.path.dirname(__file__), 'test_output')
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
COMMON = dict(
    depth_path=DEPTH_PATH,
    rain_rate=5.0, wind_speed=0.5, exposure_time=0.02,
    turbulence_deg=1.0, image_width=2048, image_height=1024,
    streak_softness=0.5, seed=SEED,
)

blends = [round(1.0 - i * 0.025, 3) for i in range(11)]
# [1.0, 0.975, 0.95, 0.925, 0.9, 0.875, 0.85, 0.825, 0.8, 0.775, 0.75]

H = 1024
top = slice(0, H // 4)
bot = slice(3 * H // 4, H)

print('=' * 75)
print(f'{"blend":>6}  {"远处p90":>8} {"远处mean":>9} {"近处mean":>9} {"近处p10":>8} {"覆盖率":>7}')
print('=' * 75)

for blend in blends:
    depth = load_and_process_depth(DEPTH_PATH, (2048, 1024), harmonic_blend=blend)
    top_d = depth[top]
    bot_d = depth[bot]

    mask = render_rain_mask(**COMMON, harmonic_blend=blend)
    coverage = np.sum(mask > 0) / mask.size * 100

    tag = f'{int(blend*100)}'
    fname = f'fine_blend_{tag}.png'
    cv2.imencode('.png', mask)[1].tofile(os.path.join(OUT_DIR, fname))

    crop = mask[256:768, 768:1280]
    cv2.imencode('.png', crop)[1].tofile(os.path.join(OUT_DIR, f'fine_crop_{tag}.png'))

    print(f'  {blend:.3f}  {np.percentile(top_d,90):>7.1f}m {top_d.mean():>8.1f}m '
          f'{bot_d.mean():>8.1f}m {np.percentile(bot_d,10):>7.1f}m {coverage:>6.1f}%')

print(f'\n保存到: {OUT_DIR}/fine_blend_*.png + fine_crop_*.png')
