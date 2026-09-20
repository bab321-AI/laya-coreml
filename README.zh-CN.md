# Laya-CoreML

**让 Laya 的结构化决策模型通过 Apple Core ML 在本机运行。**

这是 [Laya](https://github.com/NandhaKishorM/laya) 的独立实验性移植，与
[laya-mlx](https://github.com/mizorewww/laya-mlx) 配套。
提供完整模型转换、Python 推理 API、精度验证、设备计算计划与可复现 benchmark。
模型输出 `choice` / `score` / `noul` 的概率，不逐 token 生成文本。

**ANE 路径已跑通：短问题 P50 4.98 ms / P95 5.31 ms，整机每次决策能耗比编译后的 MLX FP16 改善 2.78 倍。**
这是 M3 Max 上 multilingual 模型的专门改写，十倍目标尚未达到。
计算计划与独立 Instruments 硬件记录共同支持 Neural Engine 执行；测量边界见
[ANE 速度与能耗报告](docs/ANE_BENCHMARKS.md)。

**常规移植也已完成：三个 FP16 模型共 189/189 个选择与原版一致，每个模型连续 100 次 API 调用保持一致。**

## Neural Engine 实测

同一个短问题、同一份原始模型，MLX 启用 compile、prompt cache 和长度档位。
每种实现连续测试 6 个 20 秒区间，以三组平衡顺序交替，共完成 **65,598 次稳定调用**。
8-bit 是独立的近似权重压缩模型，计算仍使用 FP16。

| 指标 | 编译后的 MLX FP16 | ANE FP16 | ANE 8-bit K-means |
|---|---:|---:|---:|
| P50 / P95 | 6.94 / 7.39 ms | **4.98 / 5.31 ms** | **4.88 / 5.23 ms** |
| 平均整机功率估计 | 61.39 W | 30.75 W | 27.39 W |
| 整机每次决策能耗 | 0.4288 J | 0.1540 J | 0.1344 J |
| 速度提升 | 1× | **1.39×** | **1.42×** |
| 每次决策能耗改善 | 1× | **2.78×** | **3.19×** |

速度比乘以平均功率比，才是每次决策能耗改善；不能再乘一次速度。
扣除相邻空闲区间后，增量能耗改善分别约 **3.82 / 4.75 倍**；这不是整台电脑功耗下降倍数。
采用直接读取 SMC PSTR 的整机功率估计，并保留全部 1,101 条原始采样。
历史采样器出现过分项计数异常，受影响的整轮能耗数据已作废并保留；最终结果重新测量，未删点或裁剪。
这仍不是外接功率计测量，也不是固定游戏帧率下的功耗测试。

这需要改写完整模型图：`B,C,1,L` 布局、1×1 卷积、逐 head attention，以及 CPU 两端处理。
L96 原型的 6,390 个计算算子均为 ANE preferred；固定短输入验证 **59/59** 选择一致。
单独导出的 L192 / L1024 分别通过 **60/60、63/63**，各完成 100 次稳定重复调用。
4.98 ms 只适用于 L96；把短问题填充到 L1024 会耗时约 88 ms。
8-bit L96 通过 59/59，最大概率偏差为 0.0144；未验证完整长上下文。
6-bit / 4-bit 分别出现 0.052 / 0.200 的最大概率偏差，未通过门槛。
固定长度版本不会悄悄截断输入，也不会为压缩模型放宽精度门槛。
ANE FP16 的 Snake 对照也通过 600/600 动作一致、零死亡、零安全接管；
但当前每步串行回答三个问题，完整决策耗时没有显示出稳定优势，不能宣传为约 5 ms 一帧。
[工程实现与运行命令](docs/ANE_ENGINEERING.md) · [数学分析](docs/ANE_MATH.md)

## 本机结果

M3 Max（40 核 GPU、128 GB 内存），macOS 27.2，Core ML Tools 9.0，MLX 0.32.2。
同一台机器重新测量，单个短问题端到端耗时，包含提示词、分词、输入数组、同步推理、校准及结果格式化；排除加载与预热。

| 模型 | Core ML CPU+GPU P50 / P95 | MLX GPU P50 / P95 | FP16 选项一致性 |
|---|---:|---:|---:|
| Multilingual 322M | 11.28 / 17.38 ms | 7.87 / 9.87 ms | 63/63 |
| Laya 421M | 13.71 / 14.31 ms | 13.33 / 13.73 ms | 63/63 |
| Typed Decisions 421M | 13.70 / 14.38 ms | 13.37 / 14.39 ms | 63/63 |

FP16 最大校准概率偏差分别为 0.001578、0.003308、0.002727。
Multilingual FP32 也通过 63/63，最大偏差约 0.000000632。
这表示移植在这组验证样本中保留了原模型行为，不表示模型在任意任务上都答对。

**常规 SDPA 导出**的 Multilingual `all` / `cpu_ne` 配置分别约 78 / 81 ms。
计算计划显示这两种配置优先使用 CPU，没有算子优先分配到 Neural Engine。
因此常规 API 默认使用通过验证的 **`cpu_gpu`**。上面的 ANE 专门改写使用独立的模型图，
并补做了 Instruments 和能耗测量，不能仅靠修改常规导出的设备选项来获得相同效果。

Snake 使用专门的 `B=3 / L=64 / K=4` 导出，两组种子共运行 600 步：
**零死亡、零安全接管、600/600 个动作与 MLX 一致**。
Core ML 的决策耗时 P50 为 11.89–12.26 ms，MLX 为 11.53–11.69 ms。
两者接收相同状态，交替执行顺序，保留相同的确定性路径特征和安全机制。
这个测试不包含终端绘制，不是完整游戏 FPS，也不证明无限生存。

[完整 benchmark](BENCHMARKS.md) · [转换问题与修复](docs/CONVERSION.md) · [原始数据](benchmarks/results)

## 安装与使用

需要 Apple Silicon、macOS 15+、Python 3.11–3.13。
本次实测系统为 macOS 27.2；尚未在旧版 macOS 或 iPhone/iPad 上验证运行。

```bash
git clone https://github.com/mizorewww/laya-coreml
cd laya-coreml
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[convert]'

# 首次下载固定版本的原始权重，在 FP32 模块中加载，再导出为 Core ML FP16。
laya-coreml convert laya-multilingual models/laya-multilingual
```

```python
import laya_coreml as laya

agent = laya.load("models/laya-multilingual")
result = agent.predict(
    "发票被重复扣款，请退还多扣的钱。",
    {
        "department": {
            "type": "choice",
            "instructions": "Who should handle this?",
            "criteria": ["billing", "technical", "sales"],
        }
    },
)
print(result["answers"]["department"])
```

推理只读取本地导出目录，可以离线使用。
PyTorch 只负责转换；在只推理的环境安装不带 `[convert]` 的本项目即可，
不需要 PyTorch、Transformers 或 MLX。模型权重与 `.mlpackage` 不纳入 Git。

## 当前边界

- 默认采用有限长度档位，并将输入填充到合适档位；原模型的上下文限制保持不变。
- 默认导出 batch 为 1、最多 32 个选项。多个问题顺序执行，因此不能把它和 MLX 的批量结果当作相同张量形状的对照。
- 已测试的英文模型上下文为 512，其余两个为 1024。固定短长度导出遇到过长输入会报错。
- 在本机上，`RangeDim + CPU_AND_GPU` 会出现明显错误和不稳定输出，运行时默认禁止此组合。
- 修复了短输入测试暴露的 MPSGraph 布尔常量切片编译崩溃：先切片整数位置，再构造布尔注意力掩码。
- 同一份双向 encoder 的 state 表示依赖问题；没有跨问题隐藏状态缓存。

默认转换保存参数、源权重 SHA256、固定的上游版本、模型与分词器文件哈希。
失败实验同样保留，详见原始报告的 `passed` 字段。

## 复现

```bash
pip install -e '.[convert,dev,compare]'
pytest -q

python -m benchmarks.validate models/laya-multilingual \
  --name laya-multilingual --compute-units cpu_gpu --repeats 100 \
  --output artifacts/validation.json

python -m benchmarks.run models/laya-multilingual \
  --compute-units cpu_gpu --iterations 100 --plan \
  --output artifacts/benchmark.json
```

其余模型、Snake 的导出与完整复现命令见 [英文 README](README.md)。
Apache-2.0；上游归属与改动说明见 [NOTICE](NOTICE)。
