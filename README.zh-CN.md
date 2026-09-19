# Laya-CoreML

**让 Laya 的结构化决策模型通过 Apple Core ML 在本机运行。**

这是 [Laya](https://github.com/NandhaKishorM/laya) 的独立实验性移植，与
[laya-mlx](https://github.com/mizorewww/laya-mlx) 配套。
提供完整模型转换、Python 推理 API、精度验证、设备计算计划与可复现 benchmark。
模型输出 `choice` / `score` / `noul` 的概率，不逐 token 生成文本。

**迁移成功：三个 FP16 模型共 189/189 个选择与原版一致，每个模型连续 100 次 API 调用保持一致。**
性能上，本次实现还没有超过 MLX，也没有获得 Neural Engine 加速的证据。

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

Multilingual 的 `all` / `cpu_ne` 配置分别约 78 / 81 ms。
计算计划显示这两种配置优先使用 CPU，没有算子优先分配到 Neural Engine。
这是计划信息，未进行 Instruments 硬件执行追踪或能耗测量。
因此 API 默认使用通过验证的 **`cpu_gpu`**。

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

# 首次下载固定版本的原始 FP32 权重，然后转换为 Core ML FP16。
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
