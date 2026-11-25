"""
Build a real-text training corpus from the READMEs (long descriptions) of popular PyPI packages.

Why: the demo model must be trained on documents it is *not* evaluated on. These READMEs are
real, varied technical prose, unrelated to the handwritten evaluation documents in
data/sample_docs (company handbook, library rules, clinic info, ...), so evaluation measures
whether the model learned to *read the context*, not whether it memorised the test documents.

    python scripts/build_corpus.py --out data/pypi_corpus      # ~1-2 min, a few MB
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import urllib.request
from pathlib import Path

PACKAGES = """
requests urllib3 certifi idna charset-normalizer numpy pandas scipy matplotlib seaborn scikit-learn
statsmodels sympy networkx pillow opencv-python imageio scikit-image tqdm rich click typer colorama
fastapi starlette uvicorn flask django bottle tornado aiohttp httpx gunicorn werkzeug jinja2 markupsafe
itsdangerous sqlalchemy alembic psycopg2 pymysql peewee redis celery kombu pika boto3 botocore s3transfer
google-cloud-storage azure-storage-blob pyyaml toml tomli tomlkit python-dotenv pydantic attrs marshmallow
cerberus jsonschema orjson ujson simplejson msgpack protobuf grpcio pyarrow polars dask xarray h5py
netcdf4 zarr numba cython pybind11 cffi setuptools wheel pip virtualenv tox nox pytest hypothesis
coverage pytest-cov mock responses freezegun faker factory-boy black isort flake8 pylint mypy ruff
autopep8 yapf bandit pre-commit sphinx mkdocs mkdocs-material docutils pygments markdown mistune
beautifulsoup4 lxml html5lib scrapy selenium playwright parsel feedparser python-dateutil pytz arrow
pendulum babel humanize tabulate prettytable openpyxl xlrd xlsxwriter python-docx reportlab pypdf
pdfminer.six camelot-py fpdf2 qrcode cryptography pyjwt bcrypt passlib paramiko fabric invoke ansible
docker kubernetes psutil watchdog schedule apscheduler loguru structlog sentry-sdk prometheus-client
opentelemetry-api tenacity backoff retrying cachetools diskcache joblib cloudpickle dill multiprocess
more-itertools toolz cytoolz funcy boltons sortedcontainers bidict pyrsistent wrapt decorator six
typing-extensions mypy-extensions packaging pluggy filelock platformdirs appdirs pathspec gitpython
dulwich pygithub python-gitlab jira slack-sdk tweepy praw discord.py python-telegram-bot twilio stripe
paypalrestsdk shopifyapi geopy shapely fiona geopandas pyproj folium plotly bokeh altair dash streamlit
gradio panel holoviews ipython jupyter notebook jupyterlab ipywidgets nbconvert nbformat papermill
voila torch torchvision torchaudio tensorflow keras jax flax optax transformers datasets tokenizers
sentencepiece accelerate peft diffusers safetensors huggingface-hub evaluate spacy nltk gensim
textblob langdetect fuzzywuzzy rapidfuzz jellyfish python-levenshtein unidecode ftfy emoji regex
xgboost lightgbm catboost optuna hyperopt mlflow wandb tensorboard onnx onnxruntime openvino
pytorch-lightning fastai sentence-transformers faiss-cpu annoy hnswlib chromadb qdrant-client
pymilvus weaviate-client elasticsearch opensearch-py pymongo motor cassandra-driver neo4j influxdb-client
clickhouse-driver duckdb sqlite-utils datasette tinydb zodb pyodbc cx-oracle snowflake-connector-python
pyspark findspark great-expectations pandera dbt-core prefect airflow luigi dagster kedro hydra-core
omegaconf python-box dynaconf environs click-log halo yaspin alive-progress blessed urwid textual
prompt-toolkit questionary inquirer pyfiglet termcolor art asciimatics pygame arcade pyglet kivy
pyqt5 pyside6 wxpython tkinter-tooltip pywebview eel nicegui flet toga briefcase pyinstaller nuitka
cx-freeze py2app shiv pex poetry pdm hatch flit twine build setuptools-scm versioneer bumpversion
commitizen semver pyserial pyusb bleak paho-mqtt pymodbus scapy netaddr dnspython zeroconf pysnmp
""".split()


def clean_markdown(text: str) -> str:
    """Keep prose paragraphs: drop code, HTML, badges, tables, link targets and headings' markup."""
    text = re.sub(r"```.*?```|~~~.*?~~~", "\n", text, flags=re.S)
    text = re.sub(r"<!--.*?-->|<[^>]+>", " ", text, flags=re.S)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)               # images / badges
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)            # links -> anchor text
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    paras = []
    for p in re.split(r"\n\s*\n", text):
        p = re.sub(r"^\s*#+\s*", "", p.strip())
        if not p or p.startswith(("|", ">>>", "$ ", "..", "    ", "\t", "=", "-" * 3)):
            continue
        p = re.sub(r"\s+", " ", p)
        words = p.split()
        alpha = sum(w.isalpha() for w in words) / max(len(words), 1)
        if len(words) >= 12 and alpha > 0.6 and p.count(".") >= 1:  # looks like prose
            paras.append(p)
    return "\n\n".join(paras)


def fetch(name: str):
    try:
        with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/json", timeout=20) as r:
            info = json.load(r)["info"]
        return name, info.get("summary") or name, clean_markdown(info.get("description") or "")
    except Exception:  # noqa: BLE001 - missing packages are simply skipped
        return name, None, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/pypi_corpus")
    ap.add_argument("--min-words", type=int, default=150)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kept, words = 0, 0
    with cf.ThreadPoolExecutor(16) as ex:
        for name, summary, body in ex.map(fetch, sorted(set(PACKAGES))):
            n = len(body.split())
            if n < args.min_words:
                continue
            (out / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', name)}.md").write_text(f"# {name}: {summary}\n\n{body}\n")
            kept += 1
            words += n
    print(f"wrote {kept} documents, {words} words of prose to {out}")


if __name__ == "__main__":
    main()
