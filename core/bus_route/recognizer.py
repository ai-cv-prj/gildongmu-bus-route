# 파일명: core/bus_route/recognizer.py
# 설명: PARSeq 모델 로딩, 번호 이미지 전처리, 추론을 담당하는 파일.

from dataclasses import dataclass
import hashlib
from pathlib import Path
import time

from PIL import Image, ImageOps
import torch


PARSEQ_COMMIT = "1902db043c029a7e03a3818c616c06600af574be"
DEFAULT_CACHE = Path(__file__).resolve().parents[2] / ".cache/torch"


@dataclass(frozen=True)
class Prediction:
    text: str
    token_scores: list[float]
    elapsed_ms: float


class PARSeqRecognizer:
    """원본 문자열을 반환한다. 점수는 보정된 정답 확률이 아니다."""

    def __init__(self, device: str = "cuda", cache_dir: Path = DEFAULT_CACHE):
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA를 사용할 수 없습니다. GPU 환경을 확인하거나 --device cpu를 사용하세요.")
        if self.device.type not in {"cuda", "cpu"}:
            raise ValueError("지원 장치는 cuda 또는 cpu입니다.")
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(self.cache_dir))
        # 공식 저장소 코드를 commit으로 고정하고 공식 가중치를 해시 검사하여 로드한다.
        self.model = torch.hub.load(
            f"baudm/parseq:{PARSEQ_COMMIT}", "parseq",
            pretrained=True, trust_repo=True, skip_validation=True,
        ).eval().to(self.device)
        # Torch Hub가 내려받은 공식 저장소의 전처리를 사용한다.
        from strhub.data.module import SceneTextDataModule

        self.transform = SceneTextDataModule.get_transform(self.model.hparams.img_size)
        with torch.inference_mode():
            size = self.model.hparams.img_size
            self.model(torch.zeros(1, 3, *size, device=self.device))
        self._synchronize()

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def predict(self, image: Image.Image) -> Prediction:
        """RGB 변환·전처리·전송·추론·디코딩 포함 시간. 파일 읽기는 제외."""
        self._synchronize()
        start = time.perf_counter()
        image = ImageOps.exif_transpose(image).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = self.model(tensor).softmax(-1)
            labels, scores = self.model.tokenizer.decode(probabilities)
        token_scores = scores[0].detach().cpu().tolist()
        self._synchronize()
        return Prediction(labels[0], token_scores, (time.perf_counter() - start) * 1000)

    def predict_batch(self, images: list[Image.Image]) -> list[Prediction]:
        """후보들을 한 번에 추론. elapsed_ms는 총 소요 시간의 후보별 균등 배분값."""
        if not images:
            return []
        self._synchronize()
        start = time.perf_counter()
        batch = torch.stack([self.transform(ImageOps.exif_transpose(im).convert("RGB"))
                             for im in images]).to(self.device)
        with torch.inference_mode():
            labels, scores = self.model.tokenizer.decode(self.model(batch).softmax(-1))
        values = [score.detach().cpu().tolist() for score in scores]
        self._synchronize()
        elapsed = (time.perf_counter()-start)*1000/len(images)
        return [Prediction(text, score, elapsed) for text,score in zip(labels,values)]

    def metadata(self) -> dict:
        weights = self.cache_dir / "checkpoints/parseq-bb5792a6.pt"
        digest = hashlib.sha256()
        with weights.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return {
            "model": "parseq", "repository": "https://github.com/baudm/parseq",
            "commit": PARSEQ_COMMIT, "weights": str(weights),
            "weights_sha256": digest.hexdigest(), "torch": torch.__version__,
            "cuda_build": torch.version.cuda, "device": str(self.device),
            "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
            "image_size_hw": list(self.model.hparams.img_size),
            "score_note": "token_scores includes EOS; not calibrated correctness probability",
        }
