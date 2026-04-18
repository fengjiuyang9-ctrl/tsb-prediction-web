# TSB Prediction Web (5-Fold Railway Build-Optimized Version)

## 结论
这是可部署到 Railway 的 Python Tornado 版本，默认使用 5-fold 聚合推理，并针对 Railway 构建超时做了依赖优化。

## 本地运行
```bash
python -m pip install -r requirements.txt
python src/web_app_tornado.py
```
默认访问：`http://127.0.0.1:8501`

## Railway 部署
推荐 Start Command：
```bash
python src/web_app_tornado.py
```
也可直接使用仓库内 `Procfile`。

## 环境变量
- `PORT`：监听端口（Railway 会自动注入）
- `RUN_DIR`：模型目录（默认已指向当前内置模型路径）
- `MAX_FOLDS`：可选，默认不设置（即使用全部 fold）

## Python 版本约束
- `runtime.txt`: `3.10.14`

## 依赖说明
- Linux（Railway）使用：
  - `torch==2.2.2+cpu`
  - `torchvision==0.17.2+cpu`
- Windows 本地开发使用：
  - `torch==2.2.2`
  - `torchvision==0.17.2`
- `numpy` 固定为 `<2.0.0` 以保证与 `torch 2.2.2` 兼容。

## 说明
- 保持 5-fold 聚合，不降级业务逻辑。
- 模型改为懒加载：服务启动时不立即加载 5 个 fold，首次预测请求时再加载，降低启动压力。
