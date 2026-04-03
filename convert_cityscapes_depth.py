"""
============================================================================
Cityscapes disparity -> depth 转换工具
============================================================================

依据 Cityscapes 官方 README：
  disparity 有效像素值 p > 0 时
    d = (p - 1) / 256
    depth = baseline * fx / d

输入:
  - disparity_trainvaltest/disparity/**/_disparity.png
  - camera_trainvaltest/camera/**/_camera.json

输出:
  - depth_trainvaltest/depth/**/_depth.png

输出深度图编码:
  - uint16 PNG
  - 单位: 厘米 (cm)，最大表示 655.35m
  - 0 表示无效深度（渲染时会被填充为 depth_scale）
============================================================================
"""

import os
import glob
import argparse
import numpy as np
import cv2

from rain_renderer import convert_cityscapes_disparity_to_depth


def save_depth_png_cm(depth_m: np.ndarray, output_path: str) -> None:
    """保存深度图为 uint16 PNG，单位厘米，最大 655.35m，0 表示无效"""
    depth_cm = np.round(np.clip(depth_m, 0.0, 655.35) * 100.0).astype(np.uint16)
    ok, buf = cv2.imencode('.png', depth_cm)
    if not ok:
        raise RuntimeError(f'编码失败: {output_path}')
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    buf.tofile(output_path)


def convert_cityscapes_depth_dataset(
    disparity_root: str,
    camera_root: str,
    output_root: str,
    max_depth_m: float = 250.0,
    skip_existing: bool = True,
) -> None:
    disparity_files = sorted(
        glob.glob(os.path.join(disparity_root, '**', '*_disparity.png'), recursive=True)
    )
    if not disparity_files:
        raise FileNotFoundError(f'未找到 disparity 文件: {disparity_root}')

    print('=' * 72)
    print('Cityscapes disparity -> depth')
    print('=' * 72)
    print(f'disparity_root: {disparity_root}')
    print(f'camera_root:    {camera_root}')
    print(f'output_root:    {output_root}')
    print(f'files:          {len(disparity_files)}')
    print(f'max_depth_m:    {max_depth_m}')
    print('=' * 72)

    converted = 0
    skipped = 0

    for idx, disparity_path in enumerate(disparity_files, start=1):
        rel_path = os.path.relpath(disparity_path, disparity_root)
        rel_dir = os.path.dirname(rel_path)
        disparity_name = os.path.basename(disparity_path)

        stem = disparity_name[:-len('_disparity.png')]
        camera_path = os.path.join(camera_root, rel_dir, f'{stem}_camera.json')
        output_path = os.path.join(output_root, rel_dir, f'{stem}_depth.png')

        if skip_existing and os.path.isfile(output_path):
            skipped += 1
            print(f'[{idx}/{len(disparity_files)}] 跳过 {disparity_name}')
            continue

        if not os.path.isfile(camera_path):
            raise FileNotFoundError(f'缺少相机标定: {camera_path}')

        depth_m = convert_cityscapes_disparity_to_depth(
            disparity_path=disparity_path,
            camera_json_path=camera_path,
            max_depth_m=max_depth_m,
        )
        save_depth_png_cm(depth_m, output_path)
        converted += 1

        valid = depth_m > 0.0
        if np.any(valid):
            min_d = float(depth_m[valid].min())
            max_d = float(depth_m[valid].max())
            msg = f'{min_d:.2f}-{max_d:.2f}m'
        else:
            msg = 'no-valid-depth'

        print(f'[{idx}/{len(disparity_files)}] 完成 {disparity_name} -> {msg}')

    print('-' * 72)
    print(f'转换完成: converted={converted}, skipped={skipped}, total={len(disparity_files)}')
    print('=' * 72)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='将 Cityscapes disparity 按官方方法转换为深度图'
    )
    parser.add_argument(
        '--disparity_root',
        default=os.path.join('dataset', 'disparity_trainvaltest', 'disparity'),
        help='Cityscapes disparity 根目录'
    )
    parser.add_argument(
        '--camera_root',
        default=os.path.join('dataset', 'camera_trainvaltest', 'camera'),
        help='Cityscapes camera 根目录'
    )
    parser.add_argument(
        '--output_root',
        default=os.path.join('dataset', 'depth_trainvaltest', 'depth'),
        help='输出深度图根目录'
    )
    parser.add_argument(
        '--max_depth_m',
        type=float,
        default=250.0,
        help='最大深度裁剪（米）'
    )
    parser.add_argument(
        '--no_skip',
        action='store_true',
        help='不跳过已存在文件'
    )
    args = parser.parse_args()

    convert_cityscapes_depth_dataset(
        disparity_root=args.disparity_root,
        camera_root=args.camera_root,
        output_root=args.output_root,
        max_depth_m=args.max_depth_m,
        skip_existing=not args.no_skip,
    )
