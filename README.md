# AlphaX

AlphaX 是一个量化研究项目，包含 AlphaCrafter 因子研究流水线和基于 Qlib 的模型基线。项目提供数据处理、因子筛选、策略验证和历史回放。

| 路径 | 内容 |
| --- | --- |
| [quant-method/alphacrafter/](quant-method/alphacrafter/) | Miner、Screener、Trader、运行入口和因子预设 |
| [quant-method/machine-learn/](quant-method/machine-learn/) | 数据准备、特征工程、Qlib 模型和组合回测 |
| [docs/](docs/) | 架构、数据审计和历史实验报告 |

从[架构说明](docs/ARCHITECTURE.md)了解数据和信号的流向。运行命令和依赖见 [AlphaCrafter 使用说明](quant-method/alphacrafter/README.md)与[模型基线使用说明](quant-method/machine-learn/README.md)。行情、特征缓存和回测产物位于 `quant-method/machine-learn/data/`，大部分由运行脚本生成。

## 测试

在已安装依赖的 Python 3.10 或更新版本环境中，从仓库根目录运行：

```sh
make test
```

可用 `make test PYTHON=/path/to/python` 指定解释器。也可以用 `make test-alphacrafter` 或 `make test-machine-learn` 单独运行一套测试。
