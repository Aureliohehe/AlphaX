# 依赖环境

根目录 `pyproject.toml` 定义 Python 版本、核心依赖范围及可选依赖分组。安装项目后可导入 `alphax.factor_dsl`，命令行仍沿用仓库中的原有脚本路径。

## 安装核心环境

从仓库根目录，在 Python 3.10–3.13 创建的虚拟环境中执行：

```bash
python -m pip install --require-hashes -r requirements/core.lock
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
make test PYTHON=python
```

锁文件包含核心运行依赖、传递依赖及 setuptools、wheel 构建工具。`--no-build-isolation` 让项目安装使用这些固定的构建工具。核心参考验证环境为 Python 3.13，CI 同时检查 Python 3.10。

## 更新核心锁文件

在单独的维护虚拟环境中执行，避免修改正在使用的研究环境：

```bash
python -m pip install pip==25.3 pip-tools==7.5.2
pip-compile --allow-unsafe --generate-hashes --strip-extras --no-emit-index-url --no-emit-trusted-host \
  --output-file=requirements/core.lock requirements/core.in
```

`core.in` 固定直接依赖版本，更新时同步核对 `pyproject.toml` 的允许范围。锁文件生成后，必须在新环境安装并运行全量测试，再提交依赖变化。需要更新传递依赖时，为生成命令添加 `--upgrade`；普通重新生成会优先保留现有版本。

## 可选运行依赖与静态检查

各环境独立创建虚拟环境。下列锁文件均包含核心依赖；不要把多个可选锁文件叠加安装。

| 环境 | 参考平台与 Python | 锁文件 | 验证命令 |
| --- | --- | --- | --- |
| LLM | Linux x86_64，3.13 | `llm.lock` | `make test-llm PYTHON=python` |
| Qlib 与 XGBoost | Linux x86_64，3.12 | `qlib.lock` | `make test-qlib PYTHON=python` |
| 静态检查 | Linux x86_64，3.13 | `dev.lock` | `make lint PYTHON=python` |

例如，在 Python 3.13 虚拟环境安装 LLM 依赖：

```bash
python -m pip install --require-hashes -r requirements/core.lock
python -m pip install --no-build-isolation --require-hashes -r requirements/llm.lock
python -m pip install --no-deps --no-build-isolation -e '.[llm]'
python -m pip check
make test-llm PYTHON=python
```

安装 Qlib 时使用 Python 3.12，把上述命令中的 `llm` 换成 `qlib`。先安装核心锁文件提供固定构建工具，再用 `--no-build-isolation` 安装可选依赖，避免 Gym 等源码包临时拉取未锁定的构建依赖。Qlib 使用可从包索引安装的 0.9.7，不依赖已有 Conda 环境中的 0.9.8。静态检查使用 `dev` 分组，当前只检查语法、控制流和未定义名称，不强制重排历史代码格式。

LLM 集成测试实际运行 LangGraph 分发、合并及局部失败处理，替换外部模型响应；LiteLLM 使用本地 mock_response 验证响应协议。测试禁用模型费用表的远程加载和 LangSmith 跟踪，并阻止测试期间的 socket 连接。它不验证供应商认证或线上模型行为。Qlib 测试实际运行归一化、DatasetH、XGBModel 训练和周频打分器，不下载行情。

更新可选锁文件时，在表中指定 Python 的维护环境中运行相同的 pip-compile 命令，将 `core` 替换为 `llm`、`qlib` 或 `dev`。这些锁文件仅针对各自参考环境解析；其他操作系统及 Python 版本需要单独验证。子目录原有 requirements 入口继续保留兼容，但不提供同等的传递依赖锁定保证。
