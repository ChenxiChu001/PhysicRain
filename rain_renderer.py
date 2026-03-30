"""
============================================================================
物理建模雨滴掩码渲染器 (Physically-Based Rain Streak Mask Renderer)
============================================================================

基于真实物理模型渲染雨滴条纹掩码图，核心物理模型包括：

1. Marshall-Palmer (M-P) 雨滴谱分布：
   N(D) = N_0 * exp(-Λ * D)
   其中 N_0 = 8000 m^{-3} mm^{-1}，Λ = 4.1 * R^{-0.21} mm^{-1}
   R 为降雨强度 (mm/h)，D 为雨滴直径 (mm)

2. 雨滴终端速度（Atlas & Ulbrich 经验公式）：
   v_t(D) = 9.65 - 10.3 * exp(-0.6 * D) [m/s]
   D 为雨滴直径 (mm)

3. 运动模糊（快门时间积分）：
   雨滴在快门曝光时间 t_exp 内移动的距离决定了条纹长度：
   L_streak = v_total * t_exp
   其中 v_total 包含垂直终端速度分量 + 水平风速分量

4. 深度感知的柱体积采样 (Column Sampling)：
   将图像每一列（或分块区域）视为一个穿过场景的视锥柱体，
   利用该柱体的深度信息计算其中的雨滴数。
   近处雨滴投影大、条纹长；远处雨滴投影小、更密集。

输入：
  - 降雨强度 R (mm/h)
  - 风向角度 wind_angle (度) 和风速 wind_speed (m/s)
  - 深度图 (512x512 灰度图像)

输出：
  - 512x512 黑色背景、白色雨滴条纹的二值掩码图

参考文献：
  - Marshall, J.S. and Palmer, W.M. (1948),
    "The distribution of raindrops with size"
  - Atlas, D. and Ulbrich, C.W. (1977),
    "Path- and area-integrated rainfall measurement"
  - Garg, K. and Nayar, S.K. (2006),
    "Photorealistic rendering of rain streaks"

日期: 2026-03-23
============================================================================
"""

import numpy as np
import cv2
import os
import glob
from dataclasses import dataclass
from typing import Tuple, Optional


# ===========================================================================
# 物理常数与相机参数配置
# ===========================================================================

@dataclass
class CameraParams:
    """
    相机内参与曝光参数

    这些参数模拟一个典型的监控/街景相机设置。
    焦距和传感器尺寸共同决定了视场角 (FoV)，
    曝光时间决定了雨滴运动模糊条纹的长度。
    """
    focal_length: float = 35.0       # 焦距 (mm)，典型街景/监控镜头
    sensor_width: float = 36.0       # 传感器宽度 (mm)，全画幅 35mm
    sensor_height: float = 24.0      # 传感器高度 (mm)，全画幅 35mm (3:2)
    image_width: int = 512           # 输出图像宽度 (像素)
    image_height: int = 512          # 输出图像高度 (像素)
    exposure_time: float = 1.0/30.0  # 曝光时间 (秒)，1/30s 模拟普通摄像头
    # 相机外参 (位姿)
    cam_height: float = 1.6          # 相机离地高度 (米)
    cam_pitch: float = 0.0           # 俯仰角 (度)，正值向下看


@dataclass
class RainParams:
    """
    降雨物理参数

    Attributes:
        rain_rate: 降雨强度 R (mm/h)
            - 小雨:   1-5   mm/h     - 中雨:   5-15  mm/h
            - 大雨:   15-30 mm/h     - 暴雨:   30-60 mm/h
            - 大暴雨: >60   mm/h
        wind_speed: 水平风速 (m/s)
            - 微风: 0-3 m/s  - 轻风: 3-6 m/s  - 中风: 6-10 m/s
        wind_direction: 罗盘风向 (度)，表示风吹来的方向
            0° = 北风 (从正前方/迎面吹来，向相机方向)
            90° = 东风 (从右侧吹来)
            180° = 南风 (从身后吹来，顺风)
            270° = 西风 (从左侧吹来)
            支持任意角度，如 345° = 北偏西15度
        wind_angle: [兼容旧接口] 旧版风向角 (度)，0°=左向右
                    如果同时指定 wind_direction，以 wind_direction 为准
        depth_scale: 深度图最大映射距离 (米)
        d_min: 最小可见雨滴直径 (mm)，小于此为雾滴
        d_max: 最大雨滴直径 (mm)，超过则因气动不稳定而破碎
    """
    rain_rate: float = 25.0
    wind_speed: float = 2.0
    wind_direction: Optional[float] = None   # 新：罗盘风向
    wind_angle: float = 10.0                 # 旧：兼容保留
    depth_scale: float = 100.0
    d_min: float = 0.5
    d_max: float = 5.0


# ===========================================================================
# 罗盘风向工具
# ===========================================================================

# 预设罗盘方向名称 -> 角度映射
WIND_DIRECTIONS = {
    'N': 0, '北': 0, '北风': 0,
    'NNE': 22.5, '北偏东': 22.5,
    'NE': 45, '东北': 45, '东北风': 45,
    'ENE': 67.5, '东偏北': 67.5,
    'E': 90, '东': 90, '东风': 90,
    'ESE': 112.5, '东偏南': 112.5,
    'SE': 135, '东南': 135, '东南风': 135,
    'SSE': 157.5, '南偏东': 157.5,
    'S': 180, '南': 180, '南风': 180,
    'SSW': 202.5, '南偏西': 202.5,
    'SW': 225, '西南': 225, '西南风': 225,
    'WSW': 247.5, '西偏南': 247.5,
    'W': 270, '西': 270, '西风': 270,
    'WNW': 292.5, '西偏北': 292.5,
    'NW': 315, '西北': 315, '西北风': 315,
    'NNW': 337.5, '北偏西': 337.5,
}


def parse_wind_direction(direction) -> float:
    """
    解析风向输入，返回罗盘角度 (0-360)

    支持格式：
      - 数字: 0, 90, 180, 345 等
      - 英文缩写: 'N', 'NE', 'SSW', 'NNW' 等
      - 中文: '北', '东北', '北偏西', '西南风' 等
      - 复合中文: '北偏西15' → 360 - 15 = 345°

    Args:
        direction: 风向输入 (float, int, 或 str)
    Returns:
        罗盘角度 (度), 0-360
    """
    if isinstance(direction, (int, float)):
        return float(direction) % 360

    if isinstance(direction, str):
        direction = direction.strip()

        # 直接匹配预设
        if direction in WIND_DIRECTIONS:
            return float(WIND_DIRECTIONS[direction])

        # 尝试解析数字字符串
        try:
            return float(direction) % 360
        except ValueError:
            pass

        # 复合格式：'北偏西15度' -> 提取基础方向和偏移
        import re
        # 匹配 "X偏Y数字" 格式
        m = re.match(r'([东南西北]+)偏([东南西北])(\d+\.?\d*)度?', direction)
        if m:
            base_name = m.group(1)
            offset_dir = m.group(2)
            offset_deg = float(m.group(3))

            base_dirs = {'北': 0, '东': 90, '南': 180, '西': 270}
            if base_name in base_dirs:
                base = base_dirs[base_name]
                # 偏移方向：顺时针为正
                if base_name == '北':
                    sign = 1 if offset_dir == '东' else -1
                elif base_name == '南':
                    sign = 1 if offset_dir == '西' else -1
                elif base_name == '东':
                    sign = 1 if offset_dir == '南' else -1
                elif base_name == '西':
                    sign = 1 if offset_dir == '北' else -1
                else:
                    sign = 1
                return (base + sign * offset_deg) % 360

    raise ValueError(f"无法解析风向: {direction}")


def decompose_wind_3d(wind_speed: float, compass_deg: float,
                      cam_pitch_deg: float = 0.0):
    """
    将罗盘风向分解为相机坐标系下的 3D 风速分量

    相机坐标系约定（右手系）：
      X = 向右 (画面右方)
      Y = 向下 (画面下方 / 重力方向)
      Z = 向前 (画面内 / 远离相机)

    风吹来的方向 compass_deg：
      0°   = 北风 = 从 Z+ 方向吹来 → 风速沿 -Z
      90°  = 东风 = 从 X+ 方向吹来 → 风速沿 -X
      180° = 南风 = 从 Z- 方向吹来 → 风速沿 +Z
      270° = 西风 = 从 X- 方向吹来 → 风速沿 +X

    物理含义：
      - v_x: 横向风速（正=向右推雨滴）→ 雨滴条纹向右倾斜
      - v_z: 纵深风速（正=向远处推）→ 迎面风为负，使雨滴看起来
             向画面中心/消失点汇聚；顺风为正，使雨滴从中心发散

    Args:
        wind_speed:   风速 (m/s)
        compass_deg:  罗盘风向 (度)
        cam_pitch_deg: 相机俯仰角 (度)

    Returns:
        (v_x, v_z): 横向风速, 纵深风速 (m/s)
    """
    rad = np.radians(compass_deg)

    # 风吹来的方向 → 风推动的方向是反向
    # compass 0° (北) = 从前方吹来 → 推向后方 (v_z < 0)
    # 所以 v_z = -cos(rad) * speed, v_x = -sin(rad) * speed
    # 但对于画面上的效果，我们关心的是风把雨滴推向哪里：
    #   北风把雨滴向画面"推近" (向相机) → v_z = -speed (朝相机)
    #   东风把雨滴向左推 → v_x = -speed
    v_x = -wind_speed * np.sin(rad)  # 向右为正
    v_z = -wind_speed * np.cos(rad)  # 向远处为正

    return v_x, v_z


# ===========================================================================
# 核心物理模型
# ===========================================================================

def marshall_palmer_nD(D: np.ndarray, R: float) -> np.ndarray:
    """
    Marshall-Palmer 雨滴谱分布 (1948)

    N(D) = N_0 * exp(-Λ * D)

    参数：
      N_0 = 8000 m^{-3} mm^{-1}  截距参数
      Λ = 4.1 * R^{-0.21} mm^{-1} 斜率参数

    物理含义：
      - 小雨滴远多于大雨滴（指数衰减）
      - 降雨越强，Λ越小，大雨滴比例越高

    Args:
        D: 雨滴直径 (mm)，标量或数组
        R: 降雨强度 (mm/h)
    Returns:
        每立方米每毫米直径间隔的雨滴数 (m^{-3} mm^{-1})
    """
    N_0 = 8000.0                        # m^{-3} mm^{-1}
    Lambda = 4.1 * R ** (-0.21)         # mm^{-1}
    return N_0 * np.exp(-Lambda * D)


def terminal_velocity(D: np.ndarray) -> np.ndarray:
    """
    雨滴终端速度 (Atlas & Ulbrich, 1977)

    v_t(D) = 9.65 - 10.3 * exp(-0.6 * D)  [m/s]

    重力和空气阻力平衡后的稳态下落速度。
    典型值: D=0.5mm→2.0m/s, D=2mm→6.5m/s, D=5mm→9.1m/s

    Args:
        D: 雨滴直径 (mm)
    Returns:
        终端速度 (m/s)
    """
    return 9.65 - 10.3 * np.exp(-0.6 * D)


def integrated_mp_count(R: float, d_min: float, d_max: float) -> float:
    """
    对 M-P 分布在 [d_min, d_max] 上积分，得到每立方米的总雨滴数

    ∫_{d_min}^{d_max} N_0 * exp(-Λ*D) dD
      = N_0/Λ * [exp(-Λ*d_min) - exp(-Λ*d_max)]

    Args:
        R:     降雨强度 (mm/h)
        d_min: 最小直径 (mm)
        d_max: 最大直径 (mm)
    Returns:
        每立方米空气中 [d_min, d_max] 范围内的雨滴总数
    """
    N_0 = 8000.0
    Lambda = 4.1 * R ** (-0.21)
    return (N_0 / Lambda) * (np.exp(-Lambda * d_min) -
                              np.exp(-Lambda * d_max))


def sample_diameters_mp(R: float, d_min: float, d_max: float,
                        n: int) -> np.ndarray:
    """
    从 M-P 分布中采样 n 个雨滴直径

    使用逆变换采样 (Inverse CDF Sampling)：
    CDF(D) = [exp(-Λ*d_min) - exp(-Λ*D)] / [exp(-Λ*d_min) - exp(-Λ*d_max)]
    D = -ln(exp(-Λ*d_min) - u * [exp(-Λ*d_min) - exp(-Λ*d_max)]) / Λ

    Args:
        R:     降雨强度 (mm/h)
        d_min: 最小直径 (mm)
        d_max: 最大直径 (mm)
        n:     采样个数
    Returns:
        直径数组 (mm)，shape = (n,)
    """
    Lambda = 4.1 * R ** (-0.21)
    # 两个边界处的 CDF 值
    exp_min = np.exp(-Lambda * d_min)
    exp_max = np.exp(-Lambda * d_max)
    # 均匀随机数 u ∈ [0, 1)
    u = np.random.uniform(0.0, 1.0, n)
    # 逆 CDF: D = -ln(exp_min - u*(exp_min - exp_max)) / Lambda
    D = -np.log(exp_min - u * (exp_min - exp_max)) / Lambda
    return D


# ===========================================================================
# 深度图处理
# ===========================================================================

def generate_default_depth_map(width: int, height: int,
                               depth_scale: float = 100.0,
                               min_depth: float = 1.0,
                               mode: str = 'gradient') -> np.ndarray:
    """
    生成默认深度图（不提供真实深度图时使用）

    Args:
        width:       图像宽度 (像素)
        height:      图像高度 (像素)
        depth_scale: 最大深度 (米)
        min_depth:   最小深度 (米)
        mode:        深度图生成模式
                     'gradient'  - 上远下近的线性渐变（模拟平地向前看）
                     'uniform'   - 均匀深度（所有区域同一距离）
                     'radial'    - 中心近边缘远的径向渐变
    Returns:
        depth_meters: (H, W) 深度图，单位米
    """
    if mode == 'gradient':
        # 上方远、下方近，模拟站在平地向前看
        gradient = np.linspace(0.0, 1.0, height).reshape(-1, 1)
        gradient = np.broadcast_to(gradient, (height, width)).copy()
        # 反转：上方 = 远处, 下方 = 近处
        gradient = 1.0 - gradient
        depth_meters = gradient * (depth_scale - min_depth) + min_depth
    elif mode == 'uniform':
        mid_depth = (depth_scale + min_depth) / 2.0
        depth_meters = np.full((height, width), mid_depth, dtype=np.float64)
    elif mode == 'radial':
        cy, cx = height / 2.0, width / 2.0
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        dist = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
        max_dist = np.sqrt(cx**2 + cy**2)
        normalized = dist / max_dist
        depth_meters = normalized * (depth_scale - min_depth) + min_depth
    else:
        raise ValueError(f"未知深度图模式: {mode}")

    return depth_meters


def load_and_process_depth(depth_path: str,
                           target_size: Tuple[int, int] = (512, 512),
                           depth_scale: float = 100.0,
                           min_depth: float = 1.0) -> np.ndarray:
    """
    加载深度图并转换为实际距离 (米)

    深度图像素值 [0, 255] 线性映射到 [min_depth, depth_scale] 米。
    值越大 = 距离越远。

    Args:
        depth_path:  深度图文件路径
        target_size: (width, height) 目标尺寸
        depth_scale: 最大深度距离 (米)
        min_depth:   最小截断深度 (米)
    Returns:
        depth_meters: (H, W) 深度图，单位米
    """
    # 使用 numpy 读取文件字节再解码，以支持中文路径
    raw = np.fromfile(depth_path, dtype=np.uint8)
    depth_img = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    if depth_img is None:
        raise FileNotFoundError(f"无法读取深度图: {depth_path}")

    if depth_img.shape[:2] != (target_size[1], target_size[0]):
        depth_img = cv2.resize(depth_img, target_size,
                               interpolation=cv2.INTER_LINEAR)

    depth_normalized = depth_img.astype(np.float64) / 255.0
    depth_meters = depth_normalized * (depth_scale - min_depth) + min_depth
    return depth_meters


# ===========================================================================
# 高效雨滴采样与向量化渲染
# ===========================================================================

def sample_raindrops_efficient(rain_params: RainParams,
                               depth_map: np.ndarray,
                               camera: CameraParams) -> dict:
    """
    高效的深度感知雨滴采样（物理正确 + 可分辨性裁剪）

    ======================== 采样策略详解 ========================

    真实 3D 体积中的雨滴数量极其庞大（R=25mm/h 时约 1350 滴/m³），
    整个视锥体内可达数千万甚至上亿。然而绝大部分远处雨滴在 512×512
    图像上的投影小于 1 像素，人眼完全不可见。

    本函数的核心思路是 **保留物理体密度模型，但只生成"图像可分辨"的雨滴**：

    1. 将场景按深度对数分层 (20 层)
    2. 对每层计算其物理体积 × M-P 积分 → 真实雨滴总数
    3. 计算该层中一个"平均雨滴"的条纹投影长度，
       若 < 1 像素则该层整层跳过（不可见）
    4. 对于可见层，用投影尺寸自适应地确定采样上限：
       max_n = 图像面积 / 单条纹平均占据面积
       若物理数 > max_n，则按 max_n 降采样
    5. 深度遮挡：丢弃被场景物体遮挡的雨滴

    这保证了：
    - 近处层完全保留物理数量（通常只有几百到几千）
    - 远处层的不可见雨滴被物理合理地裁剪
    - 降雨强度越高 → 可见雨滴越多（正确的物理趋势）

    Args:
        rain_params: 降雨参数
        depth_map:   (H, W) 深度图，单位米
        camera:      相机参数
    Returns:
        dict: { 'x', 'y', 'D', 'depth' } 所有可见雨滴的参数
    """
    H, W = depth_map.shape
    R = rain_params.rain_rate

    # ---- 深度分层：对数间距 ----
    depth_min = max(depth_map.min(), 1.0)
    depth_max = depth_map.max()
    num_layers = 20
    depth_edges = np.logspace(np.log10(depth_min), np.log10(depth_max),
                              num_layers + 1)

    # ---- M-P 积分：每 m³ 雨滴数 ----
    n_per_m3 = integrated_mp_count(R, rain_params.d_min, rain_params.d_max)

    # 平均直径 (用于估算该层典型条纹长度)
    Lambda = 4.1 * R ** (-0.21)
    D_mean = 1.0 / Lambda + rain_params.d_min  # M-P 分布的均值 ≈ 1/Λ + d_min

    all_x, all_y, all_D, all_depth = [], [], [], []

    for i in range(num_layers):
        z_near = depth_edges[i]
        z_far = depth_edges[i + 1]
        z_mid = 0.5 * (z_near + z_far)
        dz = z_far - z_near

        # ---- 该层视锥体体积 ----
        fov_w = camera.sensor_width / camera.focal_length * z_mid
        fov_h = camera.sensor_height / camera.focal_length * z_mid
        layer_volume = fov_w * fov_h * dz

        # ---- 物理雨滴总数 ----
        expected_n_physical = n_per_m3 * layer_volume

        # ---- 计算该层的平均条纹投影长度 ----
        # proj_scale = f * W_img / (z * W_sensor) [px/m]
        proj_scale = (camera.focal_length * camera.image_width) / \
                     (z_mid * camera.sensor_width)
        vt_mean = terminal_velocity(D_mean)
        streak_len_mean = vt_mean * camera.exposure_time * proj_scale

        # 如果该层平均条纹 < 2 像素，人眼几乎不可分辨，跳过
        # 真实照片中，远处的极短条纹会融入背景噪声
        if streak_len_mean < 2.0:
            continue

        # ---- 自适应采样上限 ----
        # 关键物理约束：在一张照片上，可分辨的独立雨滴条纹数量
        # 受限于图像分辨率和条纹尺寸。
        #
        # Garg & Nayar (2006) 的研究表明，在典型相机设置下，
        # 一张 640x480 图像中可见的独立雨滴条纹约为 500-3000 条。
        # 对于 512x512，我们使用类似的数量级。
        #
        # 密度因子 density_factor 控制覆盖率：
        # - 0.01 → 稀疏（~5-10%覆盖率，适合小雨/中雨）
        # - 0.03 → 中等（~15-25%覆盖率，适合大雨）
        # - 0.05 → 密集（~25-40%覆盖率，适合暴雨）
        #
        # 覆盖率随降雨强度对数增长（与 M-P 分布的物理行为一致）
        density_factor = 0.008 * np.log10(max(R, 1.0))
        density_factor = np.clip(density_factor, 0.002, 0.025)

        streak_wid_mean = (D_mean / 1000.0) * proj_scale
        streak_wid_mean = max(streak_wid_mean, 1.0)
        streak_area = streak_len_mean * streak_wid_mean
        img_area = H * W
        max_resolvable = int(img_area / streak_area * density_factor)
        max_resolvable = max(max_resolvable, 10)  # 至少 10 条

        # 取物理数和可分辨上限的较小值
        expected_n = min(expected_n_physical, max_resolvable)
        actual_n = np.random.poisson(min(expected_n, 50000))

        if actual_n <= 0:
            continue

        # ---- 采样直径 (M-P 逆 CDF) ----
        diameters = sample_diameters_mp(R, rain_params.d_min,
                                         rain_params.d_max, actual_n)

        # ---- 图像平面均匀采样位置 ----
        xs = np.random.uniform(0, W, actual_n)
        ys = np.random.uniform(0, H, actual_n)

        # ---- 层内均匀采样深度 ----
        depths = np.random.uniform(z_near, z_far, actual_n)

        # ---- 深度遮挡剔除 ----
        ix = np.clip(xs.astype(int), 0, W - 1)
        iy = np.clip(ys.astype(int), 0, H - 1)
        scene_depths = depth_map[iy, ix]
        visible = depths < scene_depths

        # ---- 过滤不可分辨的条纹 (<1px) ----
        proj_scales = (camera.focal_length * camera.image_width) / \
                      (depths * camera.sensor_width)
        vts = terminal_velocity(diameters)
        streak_lens = vts * camera.exposure_time * proj_scales
        resolvable = streak_lens >= 2.0

        keep = visible & resolvable
        if np.any(keep):
            all_x.append(xs[keep])
            all_y.append(ys[keep])
            all_D.append(diameters[keep])
            all_depth.append(depths[keep])

    if all_x:
        return {
            'x':     np.concatenate(all_x),
            'y':     np.concatenate(all_y),
            'D':     np.concatenate(all_D),
            'depth': np.concatenate(all_depth),
        }
    else:
        return {
            'x': np.array([]), 'y': np.array([]),
            'D': np.array([]), 'depth': np.array([]),
        }


def render_rain_mask(depth_path: Optional[str] = None,
                     rain_rate: float = 25.0,
                     wind_speed: float = 2.0,
                     wind_angle: float = 10.0,
                     wind_direction=None,
                     exposure_time: float = 1.0/30.0,
                     depth_scale: float = 100.0,
                     focal_length: float = 35.0,
                     sensor_width: float = 36.0,
                     sensor_height: float = 24.0,
                     image_width: int = 512,
                     image_height: int = 512,
                     turbulence_deg: float = 2.5,
                     focus_distance: float = 12.0,
                     depth_mode: str = 'gradient',
                     cam_height: float = 1.6,
                     cam_pitch: float = 0.0,
                     seed: Optional[int] = None) -> np.ndarray:
    """
    主渲染函数：生成物理真实的雨滴掩码图

    Args:
        depth_path:      深度图文件路径，为 None 时使用默认生成的深度图
        rain_rate:       降雨强度 R (mm/h)
        wind_speed:      水平风速 (m/s)
        wind_angle:      [旧接口] 风向角度 (度)，0° = 从左向右
        wind_direction:  [新接口] 罗盘风向 (度或字符串)
                         0°=北风(迎面) 90°=东风 180°=南风 270°=西风
                         指定后 wind_angle 被忽略
        exposure_time:   曝光时间 (秒)
        depth_scale:     深度缩放因子 (米)
        focal_length:    焦距 (mm)
        sensor_width:    传感器宽度 (mm)
        sensor_height:   传感器高度 (mm)
        image_width:     输出图像宽度 (像素)
        image_height:    输出图像高度 (像素)
        turbulence_deg:  湍流角度标准差 (度)
        focus_distance:  对焦距离 (米)，影响景深离焦效果
        depth_mode:      默认深度图模式 ('gradient'/'uniform'/'radial')
        cam_height:      相机离地高度 (米)
        cam_pitch:       相机俯仰角 (度)
        seed:            随机种子
    Returns:
        mask: (image_height, image_width) uint8 掩码，背景=0，雨滴=白色
    """
    if seed is not None:
        np.random.seed(seed)

    camera = CameraParams(
        exposure_time=exposure_time,
        focal_length=focal_length,
        sensor_width=sensor_width,
        sensor_height=sensor_height,
        image_width=image_width,
        image_height=image_height,
        cam_height=cam_height,
        cam_pitch=cam_pitch,
    )
    rain_params = RainParams(
        rain_rate=rain_rate,
        wind_speed=wind_speed,
        wind_angle=wind_angle,
        depth_scale=depth_scale
    )

    # ---- 1. 加载或生成深度图 ----
    if depth_path is not None and os.path.isfile(depth_path):
        depth_map = load_and_process_depth(
            depth_path,
            target_size=(camera.image_width, camera.image_height),
            depth_scale=depth_scale
        )
    else:
        depth_map = generate_default_depth_map(
            camera.image_width, camera.image_height,
            depth_scale=depth_scale,
            mode=depth_mode
        )

    # ---- 2. 采样雨滴 ----
    drops = sample_raindrops_efficient(rain_params, depth_map, camera)

    n_drops = len(drops['x'])
    if n_drops == 0:
        return np.zeros((camera.image_height, camera.image_width),
                        dtype=np.uint8)

    # ---- 3. 向量化计算所有条纹参数 ----

    # 3a. 终端速度 (m/s)
    vt = terminal_velocity(drops['D'])

    # 3b. 3D 风速分解
    #
    # 新罗盘风向系统：将风分解为横向 (v_x) 和纵深 (v_z) 两个分量
    # 纵深风 (v_z) 通过透视投影产生径向位移效果：
    #   - 迎面风 (北风, v_z < 0): 雨滴条纹向画面中心/消失点汇聚
    #   - 顺风 (南风, v_z > 0): 雨滴条纹从消失点向外发散
    #   - 侧风 (东/西): 纯横向位移，与旧行为一致
    #
    if wind_direction is not None:
        compass_deg = parse_wind_direction(wind_direction)
        v_x, v_z = decompose_wind_3d(wind_speed, compass_deg, cam_pitch)
    else:
        # 兼容旧接口: wind_angle 0° = 从左向右 → 纯横向
        old_rad = np.radians(wind_angle)
        v_x = wind_speed * np.cos(old_rad)
        v_z = 0.0  # 旧接口无纵深风

    # 3c. 曝光期间的 3D 位移 (米)
    disp_vert = vt * exposure_time              # 垂直位移 (重力)
    disp_x = v_x * exposure_time                # 横向风位移
    disp_z = v_z * exposure_time                # 纵深风位移

    # 3d. 透视投影比例因子
    # proj_x = f * W_img / (z * W_sensor)   横向 px/m
    # proj_y = f * H_img / (z * H_sensor)   纵向 px/m
    proj_x = (camera.focal_length * camera.image_width) / \
             (drops['depth'] * camera.sensor_width)
    proj_y = (camera.focal_length * camera.image_height) / \
             (drops['depth'] * camera.sensor_height)

    # 3e. 图像平面上的条纹位移 (像素)
    #
    # 横向位移: 直接投影
    streak_dx = disp_x * proj_x
    #
    # 垂直位移: 重力下落 + 纵深风的透视效应
    # 纵深风使雨滴沿 Z 轴移动，在透视投影下表现为：
    #   - 从画面中心 (消失点) 径向辐射/汇聚的位移
    #   - 数学上: dz 导致投影位置变化 ΔY ≈ (y - cy) * dz / z
    #             和 ΔX ≈ (x - cx) * dz / z
    # 对于垂直条纹外观，纵深风主要影响垂直方向和条纹缩短
    #
    # 消失点坐标 (画面中心)
    cx = camera.image_width / 2.0
    cy = camera.image_height / 2.0
    #
    # 纵深风的径向投影位移
    if abs(v_z) > 0.01:
        # 纵深位移比: 曝光期间沿 Z 轴移动的比例
        dz_ratio = disp_z / drops['depth']
        # 径向位移: 透视投影下 Δx = -(x-cx) * dZ/Z
        #   北风 (迎面, dZ<0): 雨滴朝相机来 → 投影从消失点向外发散
        #   南风 (顺风, dZ>0): 雨滴远离相机 → 投影向消失点汇聚
        radial_dx = -(drops['x'] - cx) * dz_ratio
        radial_dy = -(drops['y'] - cy) * dz_ratio
        streak_dx = streak_dx + radial_dx
        streak_dy = disp_vert * proj_y + radial_dy
    else:
        streak_dy = disp_vert * proj_y

    # 3f. 条纹长度 (像素)
    streak_len = np.sqrt(streak_dx**2 + streak_dy**2)

    # 3g. 条纹宽度 = 雨滴直径的投影
    streak_wid = (drops['D'] / 1000.0) * proj_x
    streak_wid = np.maximum(streak_wid, 1.0)

    # 3h. 条纹角度 (弧度)，atan2(水平, 垂直)
    streak_ang = np.arctan2(streak_dx, streak_dy)

    # ---- 3i. 深度感知亮度模型 ----
    #
    # (a) 距离衰减: 辐照度 ∝ 1/z
    # (b) 大气消光 (Beer-Lambert): I = I_0 * exp(-β*z)
    # (c) 雨滴截面积贡献: 大雨滴散射截面 ∝ D²，亮度更高
    # (d) 随机扰动 ±20%: 个体差异

    depth_ref = 3.0     # 参考近距离 (m)
    atm_beta = 0.008    # 降低大气消光，让远处雨滴不至于消失

    # 距离衰减：使用 sqrt(ref/z) 代替 ref/z
    # 线性反比衰减过于激进：z=20m 时只剩 10%
    # 平方根衰减更温和：z=20m 时仍有 39%，远处雨滴清晰可见
    dist_atten = np.clip(np.sqrt(depth_ref / drops['depth']), 0.0, 1.0)
    atm_atten = np.exp(-atm_beta * drops['depth'])

    # 大雨滴散射截面更大 → 亮度略高
    Lambda = 4.1 * rain_rate ** (-0.21)
    D_mean_val = 1.0 / Lambda + 0.5
    size_factor = np.clip((drops['D'] / D_mean_val) ** 0.5, 0.7, 1.4)

    brightness = dist_atten * atm_atten * size_factor

    # 随机扰动 ±15%
    jitter = 1.0 + np.random.uniform(-0.15, 0.15, n_drops)
    brightness = np.clip(brightness * jitter, 0.0, 1.0)

    # 映射到 [40, 255]：提高最暗下限，远处小雨滴至少有 40/255 ≈ 16% 亮度
    intensity = (40 + 215 * brightness).astype(np.float32)

    # ---- 3j. 风湍流角度扰动 ----
    # 真实大气中存在小尺度湍流，使每个雨滴的运动方向
    # 相对于平均风向有 ±2°~±5° 的随机偏差
    turb_std = np.radians(turbulence_deg)  # 湍流标准差
    angle_jitter = np.random.normal(0, turb_std, n_drops)
    streak_ang = streak_ang + angle_jitter

    # ---- 3k. 景深离焦 (Depth-of-Field) ----
    #
    # 户外场景中 f=35mm 镜头的超焦距约 10-20m，
    # 景深范围很大，但近处 (<3m) 和极远处 (>50m) 的雨滴
    # 仍会有轻微离焦，表现为条纹略微变宽、变柔和。
    #
    # 使用简化的离焦模型：
    #   defocus_ratio = |z - z_focus| / z_focus
    #   额外宽度 = base_defocus * defocus_ratio
    # 这比精确的薄透镜 CoC 公式更适合掩码渲染的目的。
    z_focus = focus_distance  # 对焦距离 (m)，由参数传入
    base_defocus_px = 0.8   # 最大离焦额外宽度 (px)，户外场景景深大，效果微弱

    defocus_ratio = np.abs(drops['depth'] - z_focus) / z_focus
    defocus_ratio = np.clip(defocus_ratio, 0.0, 1.0)
    # 离焦额外宽度 (像素)
    defocus_wid = base_defocus_px * defocus_ratio
    # 离焦条纹总宽度
    streak_wid_dof = streak_wid + defocus_wid

    # 离焦导致亮度轻微降低（能量守恒：宽度增加→单位面积亮度下降）
    dof_atten = streak_wid / np.maximum(streak_wid_dof, 0.5)
    dof_atten = np.clip(dof_atten, 0.5, 1.0)
    intensity = intensity * dof_atten

    # ---- 4. 锥形渐变条纹渲染 (Tapered Streak) ----
    #
    # 真实的运动模糊条纹不是均匀亮度的硬边线段。
    # 由于快门开合和雨滴球形形状，条纹呈现：
    #   - 中间最亮，两端渐暗（类似高斯沿轴分布）
    #   - 边缘有柔和的散射光晕
    #
    # 实现方法：将每条条纹沿长度方向分成 N 个子段，
    # 每个子段的亮度按照高斯权重衰减。
    # 子段数量根据条纹长度自适应：短条纹用少量子段，长条纹用更多。

    mask = np.zeros((camera.image_height, camera.image_width),
                    dtype=np.float32)

    sort_idx = np.argsort(-drops['depth'])

    for i in sort_idx:
        L = streak_len[i]
        W_dof = streak_wid_dof[i]
        if L < 0.5:
            continue

        x0 = drops['x'][i]
        y0 = drops['y'][i]
        ang = streak_ang[i]
        bright = intensity[i]

        # 条纹方向单位向量
        dx_unit = np.sin(ang)
        dy_unit = np.cos(ang)

        lw = max(1, int(round(W_dof)))

        # ---- 锥形渐变：沿条纹轴线分段渲染 ----
        # 子段数 = 条纹长度 / 3，但至少 2 段，最多 12 段
        n_segs = max(2, min(int(L / 3.0), 12))
        seg_len = L / n_segs

        for s in range(n_segs):
            # 该子段在条纹上的归一化位置 t ∈ [0, 1]
            t_start = s / n_segs
            t_end = (s + 1) / n_segs
            t_mid = 0.5 * (t_start + t_end)

            # 高斯权重：中间 (t=0.5) 最亮，两端渐暗
            # sigma=0.45 让中间 80% 区域保持高亮度，仅两端 ~10% 快速衰减
            # 这符合运动模糊的物理特性：球形雨滴在快门时间内
            # 中间段停留时间最长（相对于像素），两端快速经过
            gauss_weight = np.exp(-0.5 * ((t_mid - 0.5) / 0.45) ** 2)
            seg_bright = bright * gauss_weight

            if seg_bright < 1.0:
                continue

            # 子段两端的像素坐标
            sx0 = x0 + t_start * L * dx_unit
            sy0 = y0 + t_start * L * dy_unit
            sx1 = x0 + t_end * L * dx_unit
            sy1 = y0 + t_end * L * dy_unit

            pt1 = (int(round(sx0)), int(round(sy0)))
            pt2 = (int(round(sx1)), int(round(sy1)))

            cv2.line(mask, pt1, pt2, float(seg_bright),
                     thickness=lw, lineType=cv2.LINE_AA)

    # ---- 5. 不做全局模糊 ----
    # 锥形渐变本身已提供了条纹两端的自然柔化，
    # 加上 cv2.LINE_AA 的抗锯齿，无需额外模糊。
    # 全局高斯模糊会让整个画面发虚（"近视眼"效果），去掉。

    mask = np.clip(mask, 0, 255)
    return mask.astype(np.uint8)


# ===========================================================================
# 批量处理
# ===========================================================================

def batch_render(depth_dir: str, output_dir: str,
                 rain_rate: float = 25.0,
                 wind_speed: float = 2.0,
                 wind_angle: float = 10.0,
                 exposure_time: float = 1.0/30.0,
                 depth_scale: float = 100.0,
                 seed: Optional[int] = 42) -> None:
    """
    批量处理：为每张深度图生成对应的雨滴掩码

    Args:
        depth_dir:     深度图目录
        output_dir:    输出目录
        rain_rate:     降雨强度 (mm/h)
        wind_speed:    风速 (m/s)
        wind_angle:    风向 (度)
        exposure_time: 曝光时间 (秒)
        depth_scale:   深度缩放 (米)
        seed:          随机种子
    """
    os.makedirs(output_dir, exist_ok=True)

    depth_files = sorted(glob.glob(os.path.join(depth_dir, '*_depth.png')))
    if not depth_files:
        print(f"[错误] 在 {depth_dir} 中未找到 *_depth.png 文件")
        return

    print(f"[信息] 找到 {len(depth_files)} 张深度图")
    print(f"[参数] R={rain_rate} mm/h, 风速={wind_speed} m/s, "
          f"风向={wind_angle}°, 曝光={exposure_time:.4f}s")
    print("-" * 60)

    for idx, depth_path in enumerate(depth_files):
        filename = os.path.basename(depth_path)
        out_name = filename.replace('_depth.png', '_rain_mask.png')
        out_path = os.path.join(output_dir, out_name)

        img_seed = seed + idx if seed is not None else None

        mask = render_rain_mask(
            depth_path,
            rain_rate=rain_rate,
            wind_speed=wind_speed,
            wind_angle=wind_angle,
            exposure_time=exposure_time,
            depth_scale=depth_scale,
            seed=img_seed
        )

        cv2.imwrite(out_path, mask)

        rain_ratio = np.sum(mask > 0) / mask.size * 100
        print(f"  [{idx+1}/{len(depth_files)}] {filename} -> {out_name} "
              f"(覆盖率: {rain_ratio:.2f}%)")

    print("-" * 60)
    print(f"[完成] 掩码已保存到 {output_dir}")


# ===========================================================================
# 命令行入口
# ===========================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='物理建模雨滴掩码渲染器 (Marshall-Palmer + 终端速度 + 快门积分)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python rain_renderer.py --rain_rate 15                    # 中雨（无深度图）
  python rain_renderer.py --rain_rate 50 --wind_speed 5     # 暴雨+中风
  python rain_renderer.py --depth_dir depth --rain_rate 25  # 使用深度图批量处理
  python rain_renderer.py --single --width 1920 --height 1080  # 单张自定义尺寸

降雨强度参考:
  小雨:   1-5 mm/h    中雨: 5-15 mm/h    大雨: 15-30 mm/h
  暴雨:   30-60 mm/h  大暴雨: >60 mm/h
        """
    )
    parser.add_argument('--depth_dir',   default='depth',      help='深度图目录')
    parser.add_argument('--output_dir',  default='rain_masks',  help='输出目录')
    parser.add_argument('--rain_rate',   type=float, default=25.0,
                        help='降雨强度 (mm/h)')
    parser.add_argument('--wind_speed',  type=float, default=2.0,
                        help='风速 (m/s)')
    parser.add_argument('--wind_angle',  type=float, default=10.0,
                        help='风向角 (度)')
    parser.add_argument('--exposure',    type=float, default=1.0/30.0,
                        help='曝光时间 (秒)')
    parser.add_argument('--depth_scale', type=float, default=100.0,
                        help='深度缩放 (米)')
    parser.add_argument('--seed',        type=int,   default=42,
                        help='随机种子')
    parser.add_argument('--single',      action='store_true',
                        help='单张渲染模式（不使用深度图）')
    parser.add_argument('--width',       type=int, default=512,
                        help='输出图像宽度')
    parser.add_argument('--height',      type=int, default=512,
                        help='输出图像高度')
    parser.add_argument('--depth_mode',  default='gradient',
                        choices=['gradient', 'uniform', 'radial'],
                        help='默认深度图模式')
    parser.add_argument('--output',      default=None,
                        help='单张模式输出文件名')

    args = parser.parse_args()

    if args.single:
        mask = render_rain_mask(
            depth_path=None,
            rain_rate=args.rain_rate,
            wind_speed=args.wind_speed,
            wind_angle=args.wind_angle,
            exposure_time=args.exposure,
            depth_scale=args.depth_scale,
            image_width=args.width,
            image_height=args.height,
            depth_mode=args.depth_mode,
            seed=args.seed
        )
        out_path = args.output or f'rain_mask_{args.width}x{args.height}.png'
        cv2.imwrite(out_path, mask)
        rain_ratio = np.sum(mask > 0) / mask.size * 100
        print(f"[完成] 已保存到 {out_path} (覆盖率: {rain_ratio:.2f}%)")
    else:
        batch_render(
            depth_dir=args.depth_dir,
            output_dir=args.output_dir,
            rain_rate=args.rain_rate,
            wind_speed=args.wind_speed,
            wind_angle=args.wind_angle,
            exposure_time=args.exposure,
            depth_scale=args.depth_scale,
            seed=args.seed
        )
