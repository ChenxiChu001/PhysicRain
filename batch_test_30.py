"""
批量测试脚本：30种参数组合 × 30个API Key 并发
"""
import sys, os, time, json, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from concurrent.futures import ThreadPoolExecutor, as_completed
from rain_renderer import render_rain_mask
from rain_pipeline import call_gemini_image_edit, image_to_base64, numpy_to_base64
import numpy as np
import cv2

# ===========================================================================
# 30 个 API Keys
# ===========================================================================
API_KEYS = [
    "sk-xIBjDGuoRLxjwoVCYAAQHELASPRhLTAC44NUne0R0eCoN6Xm",
    "sk-1dVBhjKkQJ5Uu2FhTDMahiQpo4x3aO0dCTQN7P1WuDRsmLaO",
    "sk-Rgs6oOd5XyQxN0jInrn2MvSheKgJim6oRc1r4MJrLy4LyAe9",
    "sk-A6iYCGRoa5JJm5Cu3z51DjK2D3oluIJpwRBQOyF7BQc2FqpE",
    "sk-yOX39v273D0qpIQfBRkrkAAYx26DgK6OHUSrj0jAELbPtQbW",
    "sk-G3Blu4nEyXQ33TAQGOI92gMetzayYM522kVB3W0EEcRNOyXZ",
    "sk-JChnmydravqJU5QUiKzKrlYn6mePG8uHHYr72B5hxR2uXPCp",
    "sk-PuimBeMYCXvYJfMe41gkwklQ9LmbonjfTL3OH3dDcrcNwASj",
    "sk-DWAknUKRIchcQ5pdocY3Nna6XtWDq3fzhEWgtEqLTcjKcNPI",
    "sk-GFsY6DYqpbPP6PcFiGzWFonImB77459opCLCC9gUJLWmWBBH",
    "sk-AVgqJRmFew99xQtbZiH3DqX9MDWyRargIWpojWQCiSqxcA7V",
    "sk-1eaEIEnD47hDcckFC0DFT0Hkwum991UlmqB7wqQZdyK5afda",
    "sk-EsksfEx2iwwqyciXXSKug1SO8fmTPmpXqqwFEVB47m5PdNFS",
    "sk-xC4NaLGfTcapwDmLDODBj0eehM8bLoZRrj4Jga5aUUULIlcj",
    "sk-WgJto4BLbTbgHWgOCB2mtBYYLVHd5R5XkInxrBHEyoi7XE8H",
    "sk-Hmi7pudGkUSF0Fv85iOxpkXrnok1Fg97ljh9bvfXpPo3R9cX",
    "sk-UkeZ2WDQwCmAAv0OP1k09mLXECz3CH7KHiQL8v4AGQryyHP5",
    "sk-poYzFbx18hM6Sondn9XrBDEvAavkyTBtWyuvYdk5APErP1SP",
    "sk-f8dz8EbVSUgilNIpDjAgX0TKULGactgcHxXYitu68CYnRlzR",
    "sk-0dP4xKxWwqH4KmIJiboH53wqwr1xqkdwOqMT0CqOumFjk9Qc",
    "sk-5epw6V7GY357puWcgOK6yNu4zSC2YZO0aB3WRLn0Wc6qAQvb",
    "sk-a21QtijPXKjZQRRz5GfHXbBziI3XTHISTHFDPDCp7Tj9Bmf8",
    "sk-nudICB7biu2mpVtaWlosEQhd0nrwDmWnQaOxjDkjAPkm2RZj",
    "sk-eCLuFssvTst0nox42W55KFlazJBsmreddYScZ6rnhn1aWgR2",
    "sk-xXKma0B1taeJHpKTiWj34hUk6KGOg3RzYbYj4xsfloYd4snT",
    "sk-MwR2ti3VrqZir5vunn3Mpq5SqSWgSldWcOz27ZZejWMYIhQR",
    "sk-vA90NYQ4uvizrDzYeBfswsyvQySBbvGZMfYoPJC0VTZcLBkR",
    "sk-gBkvO4bv0AEKn9KAaN3OlnV90Exz7iBBAqSo0zZcZxpIss4g",
    "sk-cLv48HOo23hssuCL0PLB0ptEN5RIsAwNNgIerVvyuZTtSGxk",
    "sk-hrGDIFfXe3oWrFnSCLlhbIYvvHnfIu40QWE171Ukuz92l5of",
]

# ===========================================================================
# 30 种参数组合 (5降雨量 × 2风速 × 3风向)
# ===========================================================================

rain_rates = [3, 5, 10, 15, 20]
wind_speeds = [2, 3]
wind_configs = [
    ('N', 0, '北风'),       # 北风 迎面
    ('SW', 225, '西南风'),   # 西南风
    ('calm', None, '无风'),  # 无风
]

# 生成 30 种组合
tasks = []
task_id = 0
for rain_rate in rain_rates:
    for wind_speed in wind_speeds:
        for wind_abbr, wind_dir, wind_cn in wind_configs:
            tasks.append({
                'id': task_id,
                'rain_rate': rain_rate,
                'wind_speed': wind_speed if wind_dir is not None else 0,
                'wind_direction': wind_dir,
                'wind_abbr': wind_abbr,
                'wind_cn': wind_cn,
                'img_idx': task_id,  # 每个组合用不同的原图
            })
            task_id += 1

# 降雨量描述
def rain_desc(r):
    if r <= 3: return '毛毛细雨'
    if r <= 5: return '小雨'
    if r <= 10: return '中雨'
    if r <= 15: return '大雨'
    return '暴雨'

# 风描述
def wind_desc(ws, wn):
    if wn == '无风': return '无风'
    return f'{wn}，风速约{ws}米每秒'

# 生成提示词
def make_prompt(task):
    r = task['rain_rate']
    ws = task['wind_speed']
    wn = task['wind_cn']

    rd = rain_desc(r)
    wd = wind_desc(ws, wn)

    if wn == '无风':
        wind_part = '雨滴近乎垂直下落，没有明显的风力影响。'
    elif wn == '北风':
        wind_part = f'刮着{wd}的{wn}，雨滴被风从正前方吹来，呈现迎面扑来的态势。'
    else:
        wind_part = f'刮着{wd}的{wn}，雨滴受风力影响呈现斜向飘落。'

    prompt = (
        f'一张具有逼真摄影质感的图像，场景与第一张图片完全一致。'
        f'该场景如今设定在一场{rd}中，降雨量约为 {r} 毫米每小时。'
        f'{wind_part}'
        f'降雨的具体分布、角度和密度，均参照所提供的第二张雨水蒙版图像中的样式。'
    )

    # 根据降雨量调整光照和环境描述
    if r <= 5:
        prompt += '天色略显阴沉，地面微微湿润，远处能见度略有下降但整体通透。'
    elif r <= 10:
        prompt += '天色阴沉，路面有薄层积水反射环境光，能见度有所下降。'
    else:
        prompt += '整体光照昏暗阴沉，路面有明显积水，能见度明显降低，尤其是在远处。'

    prompt += '仅仅是分布如蒙版所示而非简单叠加，让雨滴更自然地融入到下雨的环境之中。请直接输出修改后的图像。'

    return prompt


# ===========================================================================
# 单任务处理
# ===========================================================================

print_lock = threading.Lock()
counter = {'success': 0, 'fail': 0}
counter_lock = threading.Lock()

OUTPUT_DIR = 'batch_test_output'
MASK_DIR = 'batch_test_masks'

def process_task(task, api_key):
    tid = task['id']
    img_idx = task['img_idx']
    r = task['rain_rate']
    ws = task['wind_speed']
    wd = task['wind_direction']
    wa = task['wind_abbr']

    img_path = f'test_dataset/{img_idx:05d}.png'
    depth_path = f'test_depth/{img_idx:05d}_depth.png'
    out_name = f'R{r}_W{ws}_{wa}_{img_idx:05d}_rain.png'
    out_path = os.path.join(OUTPUT_DIR, out_name)
    mask_name = f'R{r}_W{ws}_{wa}_{img_idx:05d}_mask.png'
    mask_path = os.path.join(MASK_DIR, mask_name)

    # 跳过已存在
    if os.path.isfile(out_path):
        with print_lock:
            print(f'[{tid+1}/30] {out_name} - 已存在，跳过')
        with counter_lock:
            counter['success'] += 1
        return

    try:
        # 1. 生成掩码
        raw = np.fromfile(img_path, dtype=np.uint8)
        orig = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        h, w = orig.shape[:2]

        mask = render_rain_mask(
            depth_path=depth_path,
            rain_rate=r,
            wind_speed=ws,
            wind_direction=wd,
            exposure_time=0.02,
            turbulence_deg=1.5,
            focal_length=5.0,
            depth_scale=100.0,
            image_width=w,
            image_height=h,
            seed=42 + tid,
        )

        # 保存掩码
        cv2.imencode('.png', mask)[1].tofile(mask_path)

        cov = np.sum(mask > 0) / mask.size * 100

        # 2. 生成提示词
        prompt = make_prompt(task)

        with print_lock:
            print(f'[{tid+1}/30] {out_name} - 掩码{cov:.0f}%, API调用中...', flush=True)

        # 3. 调用 API
        t0 = time.time()
        result = call_gemini_image_edit(
            original_image_path=img_path,
            mask_image=mask,
            prompt=prompt,
            api_key=api_key,
            timeout=300,
        )

        # 4. 保存结果
        cv2.imencode('.png', result)[1].tofile(out_path)
        elapsed = time.time() - t0

        with print_lock:
            print(f'[{tid+1}/30] {out_name} - 成功 ({elapsed:.0f}s)')
        with counter_lock:
            counter['success'] += 1

    except Exception as e:
        with print_lock:
            print(f'[{tid+1}/30] {out_name} - 失败: {e}')
        with counter_lock:
            counter['fail'] += 1


# ===========================================================================
# 主入口
# ===========================================================================

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MASK_DIR, exist_ok=True)

    # 限制并发数为 6，避免压垮本地代理
    MAX_CONCURRENT = 6

    print('=' * 60)
    print(f'PhysicRain 批量测试 - 30种组合 (并发={MAX_CONCURRENT})')
    print('=' * 60)
    print(f'降雨量:  {rain_rates}')
    print(f'风速:    {wind_speeds}')
    print(f'风向:    {[c[2] for c in wind_configs]}')
    print(f'API Keys: {len(API_KEYS)} 个')
    print(f'总任务:  {len(tasks)} 个')
    print('=' * 60)

    # 生成提示词文档
    with open('batch_test_prompts.md', 'w', encoding='utf-8') as f:
        f.write('# PhysicRain 批量测试 - 提示词文档\n\n')
        f.write('| # | 文件名 | 降雨量 | 风速 | 风向 | 原图 |\n')
        f.write('|---|--------|--------|------|------|------|\n')
        for t in tasks:
            fname = f"R{t['rain_rate']}_W{t['wind_speed']}_{t['wind_abbr']}_{t['img_idx']:05d}_rain.png"
            f.write(f"| {t['id']+1} | {fname} | {t['rain_rate']} mm/h | {t['wind_speed']} m/s | {t['wind_cn']} | {t['img_idx']:05d}.png |\n")
        f.write('\n---\n\n')
        for t in tasks:
            fname = f"R{t['rain_rate']}_W{t['wind_speed']}_{t['wind_abbr']}_{t['img_idx']:05d}_rain.png"
            prompt = make_prompt(t)
            f.write(f"## {t['id']+1}. {fname}\n\n")
            f.write(f"- **降雨量**: {t['rain_rate']} mm/h ({rain_desc(t['rain_rate'])})\n")
            f.write(f"- **风速**: {t['wind_speed']} m/s\n")
            f.write(f"- **风向**: {t['wind_cn']}\n")
            f.write(f"- **原图**: {t['img_idx']:05d}.png\n\n")
            f.write(f"**提示词:**\n\n> {prompt}\n\n---\n\n")
    print('提示词文档已生成: batch_test_prompts.md')

    t_start = time.time()

    # 限流并发，每个线程用不同的 key
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        futures = []
        for i, task in enumerate(tasks):
            key = API_KEYS[i % len(API_KEYS)]
            f = executor.submit(process_task, task, key)
            futures.append(f)

        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                with print_lock:
                    print(f'[异常] {e}')

    elapsed = time.time() - t_start
    print('\n' + '=' * 60)
    print(f'完成! 成功: {counter["success"]}  失败: {counter["fail"]}')
    print(f'总耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)')
    print(f'输出: {OUTPUT_DIR}/')
    print(f'掩码: {MASK_DIR}/')
    print('=' * 60)
