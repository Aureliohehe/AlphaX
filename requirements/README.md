# 依赖环境

根目录 `pyproject.toml` 定义 Python 版本、核心依赖范围及可选依赖分组。当前仍从仓库中的原有脚本路径运行，安装项目只提供依赖与项目元数据，尚未迁移为可导入的共享包。

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

## 可选运行依赖

```bash
python -m pip install -e '.[llm]'
# 或安装 Qlib 和 XGBoost 基线依赖
python -m pip install -e '.[qlib]'
```

这两个可选分组目前没有独立锁文件，安装可能调整环境中的依赖。正式实验应使用单独环境，保存实际安装版本；可选依赖锁定及真实库集成测试在工程规划阶段 1B 完成。子目录原有 requirements 入口继续保留兼容。
