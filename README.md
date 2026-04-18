# TSB Prediction Web (Online Demo Version)

This directory is the **online demo version** optimized for Railway responsiveness.

- full/research version uses 5-fold ensemble.
- online demo version uses a lightweight configuration for faster response.

## Online Demo Defaults
- Procfile runs with `MAX_FOLDS=1`
- Lazy-loads model bundle on first prediction request, then reuses it in memory
- Limits images per request (default `ONLINE_MAX_IMAGES=2`)
- Resizes large uploaded images before inference (`ONLINE_MAX_EDGE=1280`)

## Start
```bash
python -m pip install -r requirements.txt
python src/web_app_tornado.py
```

## Railway
```bash
MAX_FOLDS=1 python src/web_app_tornado.py
```

## Optional Env
- `PORT`
- `RUN_DIR`
- `MAX_FOLDS`
- `ONLINE_MAX_IMAGES`
- `ONLINE_ZIP_MAX_IMAGES`
- `ONLINE_MAX_EDGE`
