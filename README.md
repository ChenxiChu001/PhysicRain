# PhysicRain
PhysicRain: 一个基于物理的雨天先验生成器与空间条件引擎，用于引导视觉大模型进行可控的天气编辑A physics-based rain prior generator and spatial conditioning engine for guiding vision foundation models in controllable weather editing.
# PhysicRain 🌧️

**A Physics-Based Rain Prior Generator for Vision Foundation Models**

PhysicRain 是一个轻量级、基于纯物理定律的 3D 雨景仿真与渲染引擎。本项目旨在生成高保真、结构物理受控的雨滴分布图（Spatial Conditioning Map），为下游的扩散模型（Diffusion Models, 如 DiT、Stable Diffusion）在处理复杂天气场景生成与编辑任务时，提供精确的**空间物理先验（Physics-based Prior）**。

## ✨ 核心特性 (Key Features)

* **🔬 严谨的微物理建模 (Microphysics Modeling):**
    * 采用 **Gamma 分布**精准模拟真实气象学中的雨滴尺寸分布（Drop Size Distribution）。
    * 内建空气动力学终端速度模型与风速矢量计算，真实还原 3D 空间内的雨滴运动学轨迹。
* **🎥 高级光学前向渲染 (Advanced Optical Rendering):**
    * 集成 **Henyey-Greenstein 相位函数**，精确计算光线在雨滴介质内部的次表面散射。
    * 通过薄透镜物理模型计算弥散圆 (Circle of Confusion)，结合运动位移，利用光斑（Bokeh）沿轨迹积分实现逼真的动态模糊（Motion Blur）与景深（DoF）效果。
* **🤖 完美契合 AIGC 架构 (AIGC-Ready Guidance):**
    * 输出的高对比度物理特征图，可无缝作为 ControlNet 的条件输入，或转化为**本征图感知注意力机制（IMAA）**中的空间掩码（Spatial Mask）。
    * 有效引导 Vision Foundation Models 生成物理合理的折射、反光及地面湿润等复杂光影交互。

## 🚀 快速开始 (Quick Start)

### 依赖安装
本项目基于轻量级的 Python 科学计算栈，无重型深度学习框架依赖：
```bash
pip install numpy scikit-image scipy pillow tqdm
