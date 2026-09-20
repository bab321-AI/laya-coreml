![Laya Core ML 在本机玩贪吃蛇](https://raw.githubusercontent.com/mizorewww/laya-coreml/main/docs/assets/snake-demo.gif)

# Laya-CoreML

**在 Apple Silicon 上运行开放权重的决策模型：Core ML、Neural Engine、零输出 token。**

[PyPI](https://pypi.org/project/laya-coreml/) · [Hugging Face](https://huggingface.co/aac6fef/laya-multilingual-coreml-ane) · [English](https://github.com/mizorewww/laya-coreml)

上面的贪吃蛇由真实 Core ML 推理驱动，显示方向概率、分数、蛇长、延迟和安全层接管次数。
GIF 保持录制时的 **1× 速度**。代码提供确定性的路径特征和可见的循环安全层，模型负责给出概率。

完整游戏循环在三个种子、各 600 步的不限速测试中达到 **49.1–50.0 步/秒**，零死亡、两次安全层干预。
包含终端内容生成和序列化，排除终端窗口自身绘制；[定时模式限制与完整数据](docs/SNAKE_BENCHMARKS.md) 均已公开。

**M3 Max 上，ANE FP16 单个短问题 P50 4.98 ms / P95 5.31 ms。**
同场测试中，每次决策的整机能耗比编译后的 MLX 改善 **2.78 倍**。
单独验证的 8-bit 版本为 4.88 ms、3.19 倍能效改善。**十倍目标尚未达到。**
这些数字是单问题 API 的结果，不能当成完整贪吃蛇的每帧耗时。

## 安装后直接演示

需要 Apple Silicon、macOS 15+、Python 3.11–3.13。

```bash
pip install 'laya-coreml[demo]'
hf download aac6fef/laya-multilingual-coreml-ane --local-dir models/snake
laya-coreml-snake --model ./models/snake
```

下载一次后即可离线运行。推理不依赖 PyTorch、Transformers 或 MLX。
终端至少需要 104 列 × 35 行。空格暂停，上下箭头调速，R 重开，Q 退出。
首次 Core ML 加载和编译可能需要几十秒；默认 12 步/秒便于看清概率变化。

[操作、录制与视频导出](docs/SNAKE_DEMO.md) · [稳定速度测试](docs/SNAKE_BENCHMARKS.md) · [发布素材](docs/LAUNCH.md)

## Python 调用

```python
import laya_coreml as laya

agent = laya.load("aac6fef/laya-multilingual-coreml-ane")
result = agent.predict(
    "客户要求退还重复扣除的款项。",
    {
        "refund": {
            "type": "noul",
            "instructions": "Does the customer request a refund?",
        }
    },
)
print(result["answers"]["refund"])
```

支持 `choice`、有序 `score` 和布尔 `noul`；直接返回概率，不逐 token 生成文本。
远程模型 ID 首次会下载；`local_files_only=True` 强制只使用已有缓存，也可以直接传本地目录。
ANE 快速版本的 **96-token 总预算包括问题、选项和状态**，超出会报错。
需要更长输入时使用 `aac6fef/laya-multilingual-coreml` 的 1024-token 通用模型。
[完整 API 和模型选择](docs/USAGE.md)。

## 性能与精度

M3 Max（40 核 GPU、128 GiB），macOS 27.2。相同短问题，MLX 开启 compile、提示词缓存和长度档位。
每组六个 20 秒区间交替执行，共 **65,598 次稳定调用**。包含分词、输入准备、同步推理和结果格式化，排除加载与预热。

| 指标 | 编译后的 MLX FP16 | ANE FP16 | ANE 8-bit |
|---|---:|---:|---:|
| P50 / P95 | 6.94 / 7.39 ms | **4.98 / 5.31 ms** | **4.88 / 5.23 ms** |
| 平均整机功率估计 | 61.39 W | 30.75 W | 27.39 W |
| 每次决策整机能耗 | 0.4288 J | 0.1540 J | 0.1344 J |
| 速度提升 | 1× | **1.39×** | **1.42×** |
| 每次决策能效改善 | 1× | **2.78×** | **3.19×** |

功耗直接读取 SMC PSTR 整机传感器，保留原始样本并拒绝异常整轮；存在传感器与后台应用误差。
速度比乘以平均功率比等于每次决策能效改善，不能把能耗比再乘一次速度。
8-bit 压缩的是权重，计算仍用 FP16；模型主体包缩小不等于同比例提速。

三个通用 FP16 模型共 **189/189** 个验证问题与原版选项一致，各完成 100 次稳定重复调用。
ANE FP16 L96 通过 **59/59**，最大校准概率偏差 0.002925；W8 同一子集偏差 0.014393，
通过既定 0.02 门槛。6-bit / 4-bit 没有通过门槛，未作为发布权重上传。
这些数字验证移植一致性，不代表任意任务的正确率。

独立导出的 ANE FP16 L1024 通过完整 **63/63**，但真正 1024-token 请求耗时约 **91.7 ms**，
没有显示出长上下文加速优势。ANE Snake 对照通过 600/600 动作一致、零死亡、零接管；
当前每步串行回答三个问题，完整决策没有显示出稳定加速，不能宣传为约 5 ms 一帧。

[完整速度/能耗报告](docs/ANE_BENCHMARKS.md) · [通用 Core ML benchmark](BENCHMARKS.md) · [原始数据](benchmarks/results)

## 六个可直接下载的模型包

| Hugging Face 模型 | 默认引擎 | 总长度 / batch |
|---|---|---|
| [laya-coreml](https://huggingface.co/aac6fef/laya-coreml) | CPU + GPU | 512 / 1 |
| [laya-multilingual-coreml](https://huggingface.co/aac6fef/laya-multilingual-coreml) | CPU + GPU | 1024 / 1 |
| [laya-typed-decisions-coreml](https://huggingface.co/aac6fef/laya-typed-decisions-coreml) | CPU + GPU | 1024 / 1 |
| [laya-multilingual-coreml-snake](https://huggingface.co/aac6fef/laya-multilingual-coreml-snake) | CPU + GPU | 64 / 3 |
| [laya-multilingual-coreml-ane](https://huggingface.co/aac6fef/laya-multilingual-coreml-ane) | CPU + ANE | 96 / 1 |
| [laya-multilingual-coreml-ane-w8](https://huggingface.co/aac6fef/laya-multilingual-coreml-ane-w8) | CPU + ANE | 96 / 1 |

每个包包含模型卡、配置、tokenizer、校验和、来源和打包后验证。ANE 包自带所需的原始 embedding
与 action head 张量，无需原始训练仓库。普通 SDPA 导出和专门改写的 ANE 图是不同路径；
仅修改普通模型的 compute units 不会自动获得 ANE 快速路径。

## 文档

- [安装、Python 与 CLI](docs/USAGE.md)
- [贪吃蛇演示和录制](docs/SNAKE_DEMO.md)
- [发布版本和固定模型 revision](docs/RELEASE.md)
- [ANE 工程实现](docs/ANE_ENGINEERING.md)
- [十倍目标的数学调查](docs/ANE_MATH.md)
- [转换问题与修复](docs/CONVERSION.md)

代码采用 Apache-2.0。原始 [Laya](https://github.com/NandhaKishorM/laya) 模型由 Convai Innovations
及贡献者发布；本项目基于 [laya-mlx](https://github.com/mizorewww/laya-mlx) 完成独立 Core ML 移植。
这不是 Convai Innovations 或 Apple 官方发行版，归属见 [NOTICE](NOTICE)。
