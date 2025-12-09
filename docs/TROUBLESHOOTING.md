# 故障排除指南

## 常见错误及解决方案

### 1. EOFError in multiprocessing

**错误信息**：
```
EOFError in multiprocessing/connection.py
```

**原因**：
- 多进程环境中子进程意外退出
- OpenCV在多进程环境中的兼容性问题
- 图像处理函数异常导致子进程崩溃

**解决方案**：
1. **已修复**：图像增强函数已添加错误处理，不会导致进程崩溃
2. **如果问题仍然存在**，可以暂时禁用图像增强：
   ```bash
   # 移除 --image_enhance 参数
   python main.py ... # 其他参数
   ```

3. **检查OpenCV安装**：
   ```bash
   python -c "import cv2; print(cv2.__version__)"
   ```

4. **使用单进程模式测试**：
   ```bash
   --num_processes 1 --auto_gpu_config 0
   ```

### 2. PyTorch UserWarning: align_corners

**警告信息**：
```
UserWarning: Default grid_sample and affine_grid behavior has changed...
```

**原因**：PyTorch版本兼容性问题

**解决方案**：
- 这是警告，不影响功能，可以忽略
- 如果需要消除警告，可以在代码中显式指定 `align_corners=True`

### 3. 显存不足 (CUDA Out of Memory)

**错误信息**：
```
RuntimeError: CUDA out of memory
```

**解决方案**：
1. **降低分辨率**：
   ```bash
   --env_frame_width 384 --env_frame_height 384
   --frame_width 192 --frame_height 192
   ```

2. **使用更小的YOLO模型**：
   ```bash
   --semantic_model yolov8n.pt
   ```

3. **减少进程数**：
   ```bash
   --num_processes 1
   ```

### 4. YOLO模型下载失败

**错误信息**：
```
Failed to download yolov8s.pt
```

**解决方案**：
1. **手动下载模型**：
   ```bash
   # 下载到项目根目录
   wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt
   ```

2. **使用本地模型路径**：
   ```bash
   --semantic_model /path/to/yolov8s.pt
   ```

### 5. 语义检测数量仍然很少

**可能原因**：
- 置信度阈值过高
- 图像质量仍然不够
- YOLO模型太小

**解决方案**：
1. **进一步降低置信度阈值**：
   ```bash
   --semantic_conf_thresh 0.08
   ```

2. **使用更大的模型和增强**：
   ```bash
   --semantic_model yolov8m.pt
   --semantic_use_augment
   --image_enhance
   ```

3. **提高分辨率**（如果显存允许）：
   ```bash
   --env_frame_width 640 --env_frame_height 640
   --frame_width 320 --frame_height 320
   ```

### 6. 图像增强导致性能下降

**解决方案**：
- 图像增强会增加处理时间，如果速度是主要考虑：
  1. 移除 `--image_enhance`
  2. 使用 `--image_interpolation nearest`（最快）
  3. 降低分辨率

## 调试建议

### 1. 逐步启用功能

如果遇到问题，建议逐步启用功能来定位问题：

```bash
# 步骤1：基础配置（无优化）
python main.py --split val --eval 1 ... --use_semantic

# 步骤2：添加分辨率提升
python main.py ... --env_frame_width 512 --frame_width 256

# 步骤3：添加图像增强
python main.py ... --image_enhance

# 步骤4：使用更大模型
python main.py ... --semantic_model yolov8s.pt
```

### 2. 检查日志

查看日志文件了解详细错误信息：
```bash
cat tmp/your_exp_name/models/train.log
```

### 3. 单进程调试

使用单进程模式更容易调试：
```bash
python main.py --num_processes 1 --auto_gpu_config 0 ...
```

## 性能优化建议

### 如果速度太慢：

1. **降低分辨率**（最快的方法）
2. **使用yolov8n.pt**（最小模型）
3. **禁用图像增强**
4. **禁用TTA**（`--semantic_use_augment`）
5. **增加semantic_interval**（减少检测频率）

### 如果检测效果不好：

1. **提高分辨率**
2. **使用yolov8m.pt**（最大模型）
3. **启用图像增强**
4. **启用TTA**
5. **降低置信度阈值**
6. **使用bicubic插值**

## 联系支持

如果问题仍然存在，请提供：
1. 完整的错误信息
2. 使用的命令和参数
3. 系统信息（GPU型号、显存、Python版本）
4. 日志文件

