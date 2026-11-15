# 3d_reconstruction

An end-to-end 3D Structure-from-Motion (SfM), textured surface meshing, WebGPU compute-accelerated, neural 3D Transformer hole-completion, and interactive OpenGL camera-viewer pipeline written in Zig.

## Overview

Starting from a calibrated multi-view photo collection (`K.txt` + images), **`3d_reconstruction`** reconstructs camera poses, a sparse 3D point cloud, and a completed, photo-textured triangle mesh exported to ASCII `.ply`:

1. **Multi-Scale Feature Extraction & Matching ([`src/features.zig`](src/features.zig), [`src/gpu.zig`](src/gpu.zig))**
   - Tiled Harris corner detection (`detectAndCompute`) and binary descriptor matching.
   - Optional `--gpu` acceleration offloading Harris scoring and batched descriptor distance matrices to WebGPU compute shaders (`wgpu-native`).
2. **Incremental Structure-from-Motion ([`src/pipeline.zig`](src/pipeline.zig), [`src/epipolar.zig`](src/epipolar.zig), [`src/triangulation.zig`](src/triangulation.zig), [`src/pnp.zig`](src/pnp.zig), [`src/bundle_adjust.zig`](src/bundle_adjust.zig))**
   - Two-view essential matrix initialization, iterative PnP RANSAC registration, DLT triangulation, and SE(3) Levenberg-Marquardt / Gauss-Newton bundle adjustment.
3. **Surface Meshing & Photometric Completion ([`src/mesh.zig`](src/mesh.zig))**
   - View-consistent Delaunay meshing (`buildColoredMesh`), multi-view photo-verified edge bridging (`completeMeshWithPhotoVerify`), interior boundary-loop triangulation (`fillMeshInteriorHoles`), and Z-buffered multi-view texture subdivision (`subdivideAndProjectTexture`).
4. **Optional 3D Transformer Hole Completion ([`src/neural_depth.zig`](src/neural_depth.zig))**
   - Optional `--nn-model <onnx>` pass using native Zig bindings to **ONNX Runtime** (`c_ort.h`) and models such as **MoGe-2 ViT-B** (`moge-2-vitb-normal.onnx`).
   - Schedules viewpoints widest-hole-first, aligns predicted neural depth to SfM scale across near and far ranges using 16 log-depth octave bins (`fitAffineDepthWeightedIrlS` / `fitMonotoneQuadDepthWeightedIrlS`), and stitches hole surfaces on a regular grid with silhouette depth-jump guards.
5. **Photometric Verification & Interactive Viewer ([`src/render.zig`](src/render.zig), [`src/viewer.zig`](src/viewer.zig), [`src/viewer_math.zig`](src/viewer_math.zig))**
   - `--verify-views <dir>` renders each registered camera, held-out view, and interpolated novel viewpoint with Coverage, MAE, RMSE, PSNR, and SSIM reports.
   - Standalone OpenGL + GLFW viewer (`sfm_viewer`) with Free Orbit mode and click-to-enter **First-Person Camera mode** aligned to any reconstructed camera's pose, intrinsics FOV, look rotation, local displacement, photo PiP reference, and aligned photo overlay.

## Requirements

- **Zig** `>= 0.17.1`
- **System C libraries** (linked in [`build.zig`](build.zig)):
  - `libpng`, `libjpeg`
  - `glfw`, `GL`, `X11` (for `sfm_viewer`)
- **In-project native runtimes** (binary `.so`/`.onnx` artifacts are excluded from Git via [`.gitignore`](.gitignore)):
  - [`third_party/wgpu-native/`](third_party/wgpu-native) (`v27.0.4.0` headers + shared library under `third_party/wgpu-native/lib/`)
  - [`third_party/onnxruntime/`](third_party/onnxruntime) (`v1.22.0` headers tracked + `libonnxruntime.so` under `third_party/onnxruntime/lib/`)
  - Optional ONNX model in `models/moge-2-vitb-normal.onnx` for `--nn-model`

## Building & Testing

```bash
# Run all unit and integration tests
zig build test --summary all

# Build release binaries (zig-out/bin/3d_reconstruction and zig-out/bin/sfm_viewer)
zig build -Doptimize=ReleaseFast
```

## Usage

### Reconstruction CLI (`3d_reconstruction`)

```bash
./zig-out/bin/3d_reconstruction \
  --data data/fountain-p11 \
  --threads 8 \
  --gpu \
  --nn-model models/moge-2-vitb-normal.onnx \
  --out output/fountain_nn.ply \
  --verify-views output/views_nn
```

| Flag | Default | Description |
| --- | --- | --- |
| `--data <dir>` | `data/fountain-p11` | Input directory containing `K.txt` and images (`.png`, `.jpg`, `.jpeg`) |
| `--out <ply>` | `output/reconstruction.ply` | Output ASCII `.ply` mesh with camera metadata |
| `--max-dim <px>` | `1200` | Downscale longest image edge to `<px>` before SfM |
| `--threads <n>` | `1` | Worker threads for SfM and ONNX Runtime intra-op execution |
| `--gpu` | `false` | Enable WebGPU compute acceleration for feature detection and matching |
| `--nn-model <onnx>` | *none* | Path to local ONNX 3D Transformer depth model for hole completion |
| `--verify-views <dir>` | *none* | Render real/virtual/holdout/novel `.ppm` views and print PSNR/SSIM/coverage |

### Interactive 3D & Camera Viewer (`sfm_viewer`)

```bash
./zig-out/bin/sfm_viewer output/fountain_nn.ply
```

- **Click any green camera frustum or top bar number (`00`..`N`)**: Smoothly transition into that camera's **First-Person View Model** (`exact pose`, intrinsics FOV, and orientation).
- **Left-drag**: Orbit around scene target (Free Orbit) or look around in place from the camera center (Camera View).
- **Right-drag / `Shift` + Left-drag**: Pan in the active camera plane.
- **`W`/`A`/`S`/`D`/`Q`/`E` & Scroll Wheel**: Displace forward/backward, strafe, and move up/down along the active camera's rotated axes (or translate target / zoom in Free Orbit).
- **`Left`/`Right` Arrows**: Step between reconstructed cameras.
- **`R`**: Cycle aligned real photo overlay (`OFF` / `50%` / `100%`).
- **`P`**: Toggle Picture-in-Picture reference photo thumbnail.
- **`F` / `Esc` / `FREE ORBIT (F)`**: Return smoothly to Free Orbit mode.

## Benchmark Snapshot (`fountain-p11`, 11 images, `--threads 8`, `-Doptimize=ReleaseFast`)

| Configuration | Total Time (SfM) | Vertices / Triangles | Mean Coverage (11 cams) | PSNR | SSIM |
| --- | --- | --- | --- | --- | --- |
| CPU | `4702.1 ms` (`4243.5 ms`) | `255,073` / `491,760` | `61.3%` | `26.00 dB` | `0.7304` |
| WebGPU (`--gpu`) | `3343.5 ms` (`2884.0 ms`) | `255,073` / `491,760` | `61.3%` | `26.00 dB` | `0.7304` |
| WebGPU + MoGe-2 ONNX (`--nn-model`) | `44668.4 ms` (`2822.5 ms`) | `934,393` / `1,795,088` | `97.9%` | `24.93 dB` | `0.7402` |
