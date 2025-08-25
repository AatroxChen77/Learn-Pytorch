### Argparse + Config 规范

本项目中 `src/train.py` 与 `src/eval.py` 均采用“配置文件作为默认值 + 命令行可覆写”的统一模式，实现参数管理的一致性与可维护性。

## 核心原则
- 默认值来源于 YAML 配置文件：`configs/train.yaml`、`configs/eval.yaml`。
- 命令行参数优先级更高：CLI 显式传入的值会覆写 YAML 默认值。
- 通过两阶段解析获取 `--config` 路径，再加载 YAML 作为默认值；最终解析器继续接受 `--config` 并在帮助中展示。
- 共享基础逻辑集中在 `src/utils.py`，避免重复代码。

## 关键实现
- 共享工具（`src/utils.py`）
  - `parse_base_config_arg(default_config_path)`: 仅解析 `--config`，用于提前确定配置文件路径。
  - `load_yaml_defaults(config_path)`: 加载 YAML 并返回扁平字典，失败则返回 `{}`。
- 两阶段解析
  1) 使用基础解析器获取 `--config`。
  2) 读取 YAML 得到默认值字典 `yaml_defaults`。
  3) 构造最终解析器：`ArgumentParser(..., parents=[base_parser])`，逐项设置 `default=yaml_defaults.get(..., <fallback>)`。

为什么使用 `parents=[base_parser]`：
- 最终解析器也“认识”`--config`（避免未知参数/重复定义）。
- `--config` 会出现在最终 `-h/--help` 中。
- 用户可将 `--config` 放在任意位置；前置解析用来加载 YAML，最终解析则保留该选项。

## 参数优先级
1. 命令行传入值（最高）
2. YAML 配置文件中的默认值
3. 代码中兜底的默认值（`default=...` 的 fallback）

## 如何新增参数
1. 在对应的 YAML 中添加键值（如 `configs/train.yaml`）：
   - 例：`warmup_steps: 500`
2. 在脚本的 `get_args()` 中新增 `add_argument`，默认值来自 `yaml_defaults`：
   - `parser.add_argument("--warmup_steps", type=int, default=yaml_defaults.get("warmup_steps", 500), help="...", metavar="")`
3. 在业务逻辑中读取 `args.warmup_steps` 并使用。

## 布尔参数建议
- 如果仅由配置控制，直接删除该 CLI 参数；只读 YAML 值。
- 若需 CLI 覆写，优先使用 Python 3.9+ 的 `argparse.BooleanOptionalAction`，支持 `--flag`/`--no-flag`：
```python
parser.add_argument(
    "--use_gpu",
    action=argparse.BooleanOptionalAction,
    default=yaml_defaults.get("use_gpu", True),
    help="use GPU if available"
)
```
- 使用 `action="store_true"` 时，务必提供 `default=...` 来承接 YAML 值；否则“未传参”时无法从 YAML 注入默认。

## 目录与文件
- 配置：
  - 训练：`configs/train.yaml`
  - 评估：`configs/eval.yaml`
- 代码：
  - 训练脚本：`src/train.py`
  - 评估脚本：`src/eval.py`
  - 工具方法：`src/utils.py`

## 运行示例
- 使用默认配置：
```bash
python src/train.py
python src/eval.py -ckpt experiments/checkpoints/exp-YYYYMMDD-HHMMSS/best_model.pth
```
- 指定自定义配置文件：
```bash
python src/train.py --config configs/train.yaml -lr 0.001 -bs 32
python src/eval.py --config configs/eval.yaml -ckpt experiments/checkpoints/exp-YYYYMMDD-HHMMSS/best_model.pth
```

## 依赖
- 需要 `PyYAML`（环境已在 `environment.yml` 中提供 `pyyaml`）。

## 常见问题
- “既用了 `action=store_true`，为什么还要 `default`？”
  - `store_true` 仅在命令行出现该参数时设为 True；未出现时使用 `default`。为了让 YAML 默认值生效，必须设置 `default=yaml_defaults.get("key", fallback)`。
