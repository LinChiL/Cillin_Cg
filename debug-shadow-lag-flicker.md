# Debug Session: shadow-lag-flicker

Status: [OPEN]

## Symptoms
- 自阴影区域像素持续抖动。
- 转动摄像机时，阴影看起来慢半拍/掉队。

## Constraints
- 在拿到运行时证据前，不修改业务逻辑。
- 第一阶段只做观测与插桩。

## Hypotheses
1. Shadow trace 使用的 `params.prev_view_proj` / 相机参数比当前画面滞后一帧，导致阴影贴图相对主画面慢半拍。
2. `pixel_shadow_tex` 在 AP3 / 背景 pass 中读取的是上一帧 shadow 输出，因为同一帧内读写纹理视图或 bind group 顺序存在滞后。
3. `warpBuffer` 的同像素三角选择不稳定，导致 `source_tri_id` 帧间跳变，自阴影结果抖动。
4. Shadow binning 的链表插入顺序由 atomic 写入决定，同一 cell 内遍历顺序不稳定，近距离自阴影命中帧间变化。
5. 当前自阴影 bias 仍不足，或 bias 与场景尺度/法线方向不匹配，导致相邻三角近场命中仍被误判。

## Evidence From Static Pipeline Inspection
- `run_compute_pass` 当前在 `shadow_system.trace_shadow` 之前执行。
- `cs_main` 在背景/地面绘制时读取 `pixel_shadow_tex`。
- 因此地面/背景阴影在当前帧会读到上一帧 shadow texture，符合“转摄像机时阴影慢半拍”的症状。

## Applied Fix
- 将 Shadow Binning / Shadow Trace 移到 Sky & Clouds 之前。
- 同步调整 GPU timestamp 变量顺序，避免性能面板错位。

## Verification Needed
- 请转动摄像机观察：地面/背景阴影是否仍慢半拍。
- 请观察模型自阴影区域：像素闪动是否仍存在。
- 若仍闪动，下一步插桩 `warpBuffer.source_tri_id` 与 shadow 输出，验证同像素三角归属是否帧间跳变。
