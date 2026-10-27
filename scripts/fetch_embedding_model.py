"""下载本地向量模型（BAAI/bge-small-zh-v1.5）到 models/ 目录。

国内网络直接连 huggingface.co 往往超时，这里默认走 hf-mirror 镜像，
并关闭 hf-xet（镜像上的 xet 分块下载会 401）。

用法：
    python scripts/fetch_embedding_model.py
"""
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import snapshot_download  # noqa: E402

REPO = "BAAI/bge-small-zh-v1.5"
TARGET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "bge-small-zh-v1.5")

if __name__ == "__main__":
    path = snapshot_download(
        REPO,
        local_dir=TARGET,
        allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model", "1_Pooling/*"],
    )
    print(f"已下载到 {path}")
    print("接着执行：python main.py index --force")
    sys.exit(0)
