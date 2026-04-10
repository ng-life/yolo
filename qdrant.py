import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from transformers import AutoImageProcessor, AutoModel, AutoProcessor


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


MODEL_CONFIGS = {
    "siglip": {
        "hf_model_id": "google/siglip-so400m-patch14-384",
        "collection": "warehouse_skus_siglip",
    },
    "siglip2": {
        "hf_model_id": "google/siglip2-so400m-patch14-384",
        "collection": "warehouse_skus_siglip2",
    },
    "clip": {
        "hf_model_id": "openai/clip-vit-base-patch32",
        "collection": "warehouse_skus_clip",
    },
    "clipl": {
        "hf_model_id": "openai/clip-vit-large-patch14-336",
        "collection": "warehouse_skus_clipl",
    },
    "dinov2": {
        "hf_model_id": "facebook/dinov2-base",
        "collection": "warehouse_skus_dinov2",
    },
}


class ImageEmbedder:
    def __init__(self, model_name: str, model_id: str) -> None:
        self.model_name = model_name
        self.model_id = model_id
        self.device = self._resolve_device()
        if model_name == "dinov2":
            self.processor = AutoImageProcessor.from_pretrained(model_id)
        else:
            self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        self.model.eval()

    @staticmethod
    def _resolve_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def encode_image(self, image_path: Path) -> list[float]:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            if hasattr(self.model, "get_image_features"):
                features = self.model.get_image_features(**inputs)
            else:
                features = self.model(**inputs)

            if hasattr(features, "pooler_output") and features.pooler_output is not None:
                features = features.pooler_output
            elif hasattr(features, "last_hidden_state") and features.last_hidden_state is not None:
                features = features.last_hidden_state.mean(dim=1)
            elif not torch.is_tensor(features):
                raise TypeError("Unsupported image feature output type from model")

            features = F.normalize(features, p=2, dim=-1)

        return features[0].detach().cpu().tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qdrant grouped retrieval demo with selectable vector model")
    parser.add_argument(
        "--vector-model",
        choices=sorted(MODEL_CONFIGS.keys()),
        default="siglip",
        help="选择向量模型配置（影响模型权重、向量维度与集合名）",
    )
    parser.add_argument(
        "--images-root",
        type=str,
        default="./images/sku_images",
        help="SKU 图片目录，要求结构为 images_root/<sku_id>/*.jpg",
    )
    parser.add_argument(
        "--query-image",
        type=str,
        default="./images/sku_images/小爆米花.png",
        help="待检索图片路径",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.75,
        help="最低相似度阈值，低于该值的 SKU 不返回",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="最多返回多少个 SKU 分组",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=2,
        help="每个 SKU 分组最多返回多少张图",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="运行前删除并重建当前模型对应集合，便于重复演示",
    )
    return parser.parse_args()


def list_images(sku_dir: Path) -> list[Path]:
    return sorted(
        path for path in sku_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_points(images_root: Path, embedder: ImageEmbedder) -> list[PointStruct]:
    sku_dirs = sorted(path for path in images_root.iterdir() if path.is_dir())
    if not sku_dirs:
        raise ValueError("images_root 下没有 SKU 子目录，期望结构为 images_root/<sku_id>/*.jpg")

    points: list[PointStruct] = []
    point_id = 1
    for sku_dir in sku_dirs:
        sku_id = sku_dir.name
        image_paths = list_images(sku_dir)
        if not image_paths:
            continue

        for image_path in image_paths:
            vector = embedder.encode_image(image_path)
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "sku_id": sku_id,
                        "name": sku_id,
                        "image": str(image_path.relative_to(images_root)),
                    },
                )
            )
            point_id += 1

    if not points:
        raise ValueError("未找到可用图片，请确认 SKU 子目录下存在 jpg/png 等图片")

    return points


def get_existing_collection_dim(client: QdrantClient, collection_name: str) -> int | None:
    if not client.collection_exists(collection_name):
        return None
    info = client.get_collection(collection_name)
    vectors_cfg = info.config.params.vectors
    if isinstance(vectors_cfg, VectorParams):
        return vectors_cfg.size
    return None


def main() -> None:
    args = parse_args()
    cfg = MODEL_CONFIGS[args.vector_model]
    images_root = Path(args.images_root).expanduser().resolve()
    query_image = Path(args.query_image).expanduser().resolve()

    if not images_root.exists() or not images_root.is_dir():
        raise ValueError(f"images_root 不存在或不是目录: {images_root}")
    if not query_image.exists() or not query_image.is_file():
        raise ValueError(f"query_image 不存在或不是文件: {query_image}")

    # 1. 初始化客户端 (在 Mac M4 上建议直接指定存储路径，数据会持久化到本地)
    client = QdrantClient(path="./qdrant_data")
    embedder = ImageEmbedder(args.vector_model, cfg["hf_model_id"])
    print(f"使用模型: {cfg['hf_model_id']} ({args.vector_model})")
    print(f"推理设备: {embedder.device}")

    collection_name = cfg["collection"]
    query_vector = embedder.encode_image(query_image)
    vector_dim = len(query_vector)

    if args.reset and client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    existing_dim = get_existing_collection_dim(client, collection_name)
    if existing_dim is not None and existing_dim != vector_dim:
        raise ValueError(
            f"集合 {collection_name} 维度为 {existing_dim}，当前模型维度为 {vector_dim}。"
            "请使用 --reset 重建集合。"
        )

    # 2. 创建集合
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )

    # 3. 提取图片向量并写入
    points = build_points(images_root, embedder)
    client.upsert(collection_name=collection_name, points=points)
    print(f"已写入向量数量: {len(points)}")

    # 4. 执行检索 (核心：使用 Group By 逻辑)
    search_result = client.query_points_groups(
        collection_name=collection_name,
        query=query_vector,
        group_by="sku_id",  # 关键：按 SKU ID 分组，防止结果被同一 SKU 的多张图霸屏
        limit=args.limit,
        group_size=args.group_size,
        score_threshold=args.score_threshold,
    )

    # 5. 打印结果
    print(f"--- 检索结果 ({args.vector_model}, dim={vector_dim}) ---")
    if not search_result.groups:
        print(f"没有命中相似度 >= {args.score_threshold:.2f} 的 SKU")
    else:
        for group in search_result.groups:
            print(f"SKU ID: {group.id}")
            print(f"最高置信度: {group.hits[0].score:.4f}")
            print(f"商品名称: {group.hits[0].payload['name']}")
            print("-" * 20)

    client.close()


if __name__ == "__main__":
    main()