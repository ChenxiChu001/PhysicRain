import numpy as np
from PIL import Image
import math
import time
from tqdm import tqdm

# 确保 scikit-image 和 scipy 已安装
# pip install scikit-image scipy tqdm
try:
    from skimage.draw import disk
    from scipy.stats import gamma as gamma_dist
except ImportError:
    print("scikit-image or scipy not found. Please install them:")
    print("pip install scikit-image scipy tqdm")
    exit()


# --- 1. 配置参数 ---
class Config:
    # 图像与相机参数
    IMG_WIDTH = 1242
    IMG_HEIGHT = 375
    FX = 721.5
    FY = 721.5
    CX = 609.5
    CY = 172.8

    # 降雨物理参数
    RAIN_RATE = 5.0
    N0 = 8000.0
    MIN_DIAMETER = 0.1
    MAX_DIAMETER = 6.0

    # 仿真空间参数
    Z_NEAR = 0.5
    Z_FAR = 50.0

    # --- REFINED: 调整核心参数以提升美感 ---
    # 计算优化参数 - 维持一个合理的粒子密度
    PARTICLE_BUDGET = 3500

    # 渲染效果参数
    EXPOSURE_TIME = 1 / 50.0
    FOCAL_LENGTH = 0.050
    F_NUMBER = 2.0
    FOCUS_DISTANCE = 5.0

    # 光照参数 - 稍微增强光照以突出水滴质感
    LIGHT_DIRECTION = np.array([0.0, 0.0, 1.0])
    LIGHT_INTENSITY = 60.0
    HG_G = 0.85

    # 图像输出参数
    EXPOSURE = 0.035

    # --- 真实感提升参数 ---
    SHARPNESS_THRESHOLD = 0.75
    SHARPNESS_BOOST = 1.5
    GLINT_PROBABILITY = 0.0005
    GLINT_MULTIPLIER = 200.0

    # 风力参数 (m/s) - 保持一个温和的风速
    WIND_VECTOR = np.array([1.0, 0.0, 0.0])
# --- 2. 物理与数学模型函数 ---
def get_terminal_velocity(D):
    return 9.65 - 10.3 * math.exp(-0.6 * D)


def henyey_greenstein_phase_function(cos_theta, g):
    g2 = g * g
    denominator = (1 + g2 - 2 * g * cos_theta) ** 1.5
    return (1 - g2) / denominator if denominator > 1e-6 else 0.0


def calculate_coc_diameter_pixels(Z, config):
    f = config.FOCAL_LENGTH
    N = config.F_NUMBER
    S1 = config.FOCUS_DISTANCE
    if abs(S1 - f) < 1e-6 or abs(Z - f) < 1e-6: return 0.0
    coc_physical = abs((f ** 2 / N) * (S1 - Z) / (Z * (S1 - f)))
    coc_pixels = (coc_physical / Z) * config.FY
    return coc_pixels


# --- 3. 核心模拟与渲染流程 ---
class RainSimulator:
    def __init__(self, config):
        self.config = config
        self.lambda_val = 4.1 * (config.RAIN_RATE ** -0.21)
        self.light_dir_norm = self.config.LIGHT_DIRECTION / np.linalg.norm(self.config.LIGHT_DIRECTION)
        self._setup_samplers()

    def _setup_samplers(self):
        self.d_sampler_shape = 3.0
        self.d_sampler_scale = 1.0 / self.lambda_val
        self.z_sampler_c = 1.0 / (1.0 / self.config.Z_NEAR - 1.0 / self.config.Z_FAR)

    def _sample_d_and_get_pdf(self):
        D = np.random.gamma(self.d_sampler_shape, self.d_sampler_scale)
        pdf_val = gamma_dist.pdf(D, a=self.d_sampler_shape, scale=self.d_sampler_scale)
        return D, pdf_val

    def _sample_z_and_get_pdf(self):
        u = np.random.random()
        Z = 1.0 / (1.0 / self.config.Z_NEAR - u / self.z_sampler_c)
        pdf_val = self.z_sampler_c / (Z ** 2)
        return Z, pdf_val

    def generate_raindrops_with_importance_sampling(self):
        print("Step 1: Generating raindrops with importance sampling...")
        particles = {'pos': np.zeros((self.config.PARTICLE_BUDGET, 3)),
                     'diameter': np.zeros(self.config.PARTICLE_BUDGET), 'weight': np.zeros(self.config.PARTICLE_BUDGET)}
        A_near = (self.config.IMG_WIDTH / self.config.FX) * self.config.Z_NEAR * (
                self.config.IMG_HEIGHT / self.config.FY) * self.config.Z_NEAR
        A_far = (self.config.IMG_WIDTH / self.config.FX) * self.config.Z_FAR * (
                self.config.IMG_HEIGHT / self.config.FY) * self.config.Z_FAR
        frustum_volume = (self.config.Z_FAR - self.config.Z_NEAR) / 3.0 * (A_near + A_far + math.sqrt(A_near * A_far))
        n_total_density = self.config.N0 / self.lambda_val
        total_physical_particles = n_total_density * frustum_volume
        for i in tqdm(range(self.config.PARTICLE_BUDGET), desc="Generating drops"):
            D, p_D = self._sample_d_and_get_pdf()
            if not (self.config.MIN_DIAMETER < D < self.config.MAX_DIAMETER): D, p_D = self._sample_d_and_get_pdf()
            Z, p_Z = self._sample_z_and_get_pdf()
            x_max_at_z = (self.config.IMG_WIDTH - self.config.CX) * Z / self.config.FX
            x_min_at_z = -self.config.CX * Z / self.config.FX
            y_max_at_z = (self.config.IMG_HEIGHT - self.config.CY) * Z / self.config.FY
            y_min_at_z = -self.config.CY * Z / self.config.FY
            X = np.random.uniform(x_min_at_z, x_max_at_z)
            Y = np.random.uniform(y_min_at_z, y_max_at_z)
            target_pdf_d = self.config.N0 * math.exp(-self.lambda_val * D)
            sampling_pdf_d = p_D
            weight = (target_pdf_d / (sampling_pdf_d + 1e-9))
            particles['pos'][i] = [X, Y, Z]
            particles['diameter'][i] = D
            particles['weight'][i] = weight
        if np.sum(particles['weight']) > 0: particles['weight'] *= (
                                                                           total_physical_particles / self.config.PARTICLE_BUDGET) / np.mean(
            particles['weight'])
        print(f"Generated {self.config.PARTICLE_BUDGET} particles with importance sampling.")
        return particles

    # --- REFINED: 彻底重构渲染逻辑 ---
    def render_final(self, particles):
        """
        最终渲染函数：通过绘制连续的光斑来构建雨丝，以获得更真实的“水滴感”。
        """
        print("\nStep 2: Rendering with Bokeh-based streak method...")
        image = np.zeros((self.config.IMG_HEIGHT, self.config.IMG_WIDTH), dtype=np.float32)

        for i in tqdm(range(self.config.PARTICLE_BUDGET), desc="Rendering drops"):
            X, Y, Z = particles['pos'][i]
            D = particles['diameter'][i]
            weight = particles['weight'][i]

            # --- 物理亮度计算 ---
            view_vec = -particles['pos'][i] / Z
            cos_theta = np.dot(view_vec, -self.light_dir_norm)
            phase_value = henyey_greenstein_phase_function(cos_theta, self.config.HG_G)
            scattering_cross_section = (D / 1000.0) ** 2
            distance_attenuation = 1.0 / (Z ** 2)
            hdr_brightness = (self.config.LIGHT_INTENSITY * phase_value *
                              scattering_cross_section * distance_attenuation * weight)

            # --- 模拟随机闪光 ---
            is_glint = False
            if np.random.random() < self.config.GLINT_PROBABILITY:
                hdr_brightness *= self.config.GLINT_MULTIPLIER
                is_glint = True

            if hdr_brightness <= 0: continue

            # --- 运动与光学模糊计算 ---
            v_terminal = get_terminal_velocity(D)
            p_start_2d = (self.config.FX * X / Z + self.config.CX, self.config.FY * Y / Z + self.config.CY)

            vertical_displacement = v_terminal * self.config.EXPOSURE_TIME
            Y_end = Y + vertical_displacement

            horizontal_displacement = self.config.WIND_VECTOR[0] * self.config.EXPOSURE_TIME
            X_end = X + horizontal_displacement

            p_end_2d = (self.config.FX * X_end / Z + self.config.CX, self.config.FY * Y_end / Z + self.config.CY)

            coc_rad = calculate_coc_diameter_pixels(Z, self.config) / 2.0

            p_start_x, p_start_y = p_start_2d
            p_end_x, p_end_y = p_end_2d
            length = max(1, int(np.hypot(p_end_x - p_start_x, p_end_y - p_start_y)))

            # --- 核心渲染逻辑：用光斑构建雨丝 ---

            # 根据是否是闪光点，决定光斑的基础半径
            base_radius = 0.5 if not is_glint else 1.5

            # 焦内雨滴应该更亮
            if coc_rad < self.config.SHARPNESS_THRESHOLD or is_glint:
                final_brightness = hdr_brightness * self.config.SHARPNESS_BOOST
            else:
                final_brightness = hdr_brightness

            # 预计算轨迹上亮度的总和，用于归一化
            sin_modulation_sum = sum(math.sin((j / float(length)) * math.pi) for j in range(length))
            if sin_modulation_sum < 1e-6: continue

            # 遍历轨迹的每一步，绘制一个光斑
            for j in range(length):
                t = j / float(length)
                center_x = (1 - t) * p_start_x + t * p_end_x
                center_y = (1 - t) * p_start_y + t * p_end_y

                # 模拟雨滴翻滚带来的亮度变化
                modulation = math.sin(t * math.pi)
                step_brightness = final_brightness * (modulation / sin_modulation_sum)

                # 光斑的最终半径是基础半径和光学模糊半径的和
                current_coc_rad = coc_rad if not is_glint else 0
                radius = base_radius + current_coc_rad

                if radius < 0.5:  # 对于非常细的雨丝，直接画一个像素点
                    if 0 <= center_x < self.config.IMG_WIDTH and 0 <= center_y < self.config.IMG_HEIGHT:
                        image[int(center_y), int(center_x)] += step_brightness
                else:  # 绘制一个圆盘光斑
                    rr, cc = disk((center_y, center_x), radius, shape=image.shape)
                    image[rr, cc] += step_brightness

        return image

    def post_process_and_save(self, hdr_image):
        print("\nStep 3: Applying tone mapping and saving the image...")
        hdr_image *= self.config.EXPOSURE
        ldr_image = hdr_image / (1.0 + hdr_image)
        final_image_8bit = np.clip(ldr_image * 255, 0, 255).astype(np.uint8)
        filename = f"rain_sim_R{self.config.RAIN_RATE}_final_droplet.png"
        Image.fromarray(final_image_8bit).save(filename)
        print(f"Image saved as {filename}")


if __name__ == '__main__':
    start_time = time.time()

    config = Config()
    simulator = RainSimulator(config)

    # 1. 生成雨滴
    raindrops_data = simulator.generate_raindrops_with_importance_sampling()

    # 2. 使用最终的高质量渲染函数
    hdr_render = simulator.render_final(raindrops_data)

    # 3. 后处理并保存
    simulator.post_process_and_save(hdr_render)

    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds.")
