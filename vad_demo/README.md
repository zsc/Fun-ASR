# FunASR Real-time VAD + ASR Demo

这是一个基于 FunASR 和 Silero VAD 实现的实时语音识别 Demo。包含命令行（CLI）版本和 Web 版本。

## 特性 (Features)

*   **实时语音活动检测 (VAD)**: 集成 Silero VAD，精准检测语音片段。
*   **流式识别**: 语音检测期间实时更新部分结果 (Partial)，句子结束（静音检测）后提交最终结果 (Final)。
*   **Apple Silicon 优化**: 自动检测并使用 MPS (Metal Performance Shaders) 加速推理。
*   **Web 可视化**: 提供无闪烁的 Web 界面展示实时转写结果。
*   **双模式输入**: 支持麦克风实时输入和音频文件模拟输入。

## 依赖安装 (Requirements)

请确保项目根目录下的依赖已安装，并额外安装以下库：

```bash
pip install pyaudio fastapi uvicorn websockets
```

*   **PyAudio 注意事项**:
    *   **macOS**: 如果安装失败，需先安装 portaudio: `brew install portaudio`
    *   **Linux**: `sudo apt-get install python3-pyaudio`

## 使用说明 (Usage)

**注意**: 请在项目根目录下运行以下命令，以确保正确加载 `model.py`。

### 1. 命令行演示 (CLI Demo)

直接在终端输出识别结果。

**麦克风输入:**
```bash
python vad_demo/demo_vad.py
```

**文件模拟输入:**
```bash
python vad_demo/demo_vad.py --file /path/to/your/audio.wav
```

### 2. 网页演示 (Web Demo)

启动 Web 服务，在浏览器中查看实时字幕。

**启动服务 (麦克风):**
```bash
python vad_demo/web_demo.py
```

**启动服务 (文件模拟):**
```bash
python vad_demo/web_demo.py --file /path/to/your/audio.wav
```

启动后，在浏览器访问: [http://localhost:8000](http://localhost:8000)

### 3. VAD 模型测试

检测 Silero VAD 模型是否能正确加载及使用 MPS 加速。

```bash
python vad_demo/test_silero.py
```

## 目录结构

*   `demo_vad.py`: 核心逻辑实现，CLI 版本入口。
*   `web_demo.py`: 基于 FastAPI 的 Web 服务入口，复用了核心 ASR/VAD 逻辑。
*   `test_silero.py`: VAD 环境测试脚本。
