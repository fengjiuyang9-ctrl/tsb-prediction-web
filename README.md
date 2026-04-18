# TSB Prediction Web (Tornado)

## 结论
这是一个 Python Tornado 网页应用，入口为 `src/web_app_tornado.py`，可直接部署到 Railway。

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

应用已支持从环境变量读取端口：`PORT`。
Railway 会自动注入 `PORT`，无需手动填写。

## 可选环境变量
- `PORT`：监听端口（默认 `8501`）
- `RUN_DIR`：模型运行目录，默认：
  `experiments_实验输出/e2_final_F3_reg_lr_tune_5fold10/raw_5fold10_F3_reg_lr_tune/run_20260417_223410`

## Python 版本建议
建议 `Python 3.10`（Railway 中可使用 3.10 或兼容版本）。
