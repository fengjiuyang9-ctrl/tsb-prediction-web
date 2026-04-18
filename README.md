# TSB Prediction Web (Demo Deployment Version)

## 结论
这是一个可部署到 Railway 的 Python Tornado 演示版。当前默认以单 fold（`MAX_FOLDS=1`）运行，用于显著降低镜像体积。

## 本地运行
```bash
python -m pip install -r requirements.txt
python src/web_app_tornado.py
```
默认访问：`http://127.0.0.1:8501`

## Railway 部署
推荐 Start Command：
```bash
MAX_FOLDS=1 python src/web_app_tornado.py
```
也可直接使用仓库内 `Procfile`。

## 环境变量
- `PORT`：监听端口（Railway 会自动注入）
- `RUN_DIR`：模型目录（默认已指向当前内置 demo 模型路径）
- `MAX_FOLDS`：推理使用的 fold 数，demo 默认 `1`

## Python 版本建议
建议 Python 3.10。

## 说明
- 当前为 demo deployment version，仅保留最小运行文件与单 fold 模型。
- 如需恢复 5-fold，请补回 `fold_1` 到 `fold_4`，并去掉 `MAX_FOLDS=1`。
