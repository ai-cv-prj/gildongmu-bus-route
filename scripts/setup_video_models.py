"""공식 모델 코드/가중치 준비. 다운로드는 이 명시적 명령에서만 수행."""
import hashlib
from pathlib import Path
import subprocess
import tempfile
import urllib.request
import zipfile
from bus_route.detector import ROOT, CRAFT_COMMIT


def main():
    cache = ROOT / '.cache'
    cache.mkdir(exist_ok=True)
    repo = cache / 'CRAFT-pytorch'
    if not repo.exists():
        subprocess.run(['git', 'clone', 'https://github.com/clovaai/CRAFT-pytorch.git', str(repo)], check=True)
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != CRAFT_COMMIT:
        raise RuntimeError(f'CRAFT commit 불일치: {commit}; 필요: {CRAFT_COMMIT}')
    # torchvision 0.22 호환: 사전학습 VGG는 불필요하며 CRAFT 전체 가중치를 로드한다.
    base = repo / 'basenet/vgg16_bn.py'
    text = base.read_text()
    text = text.replace('from torchvision.models.vgg import model_urls\n', '')
    text = text.replace("        model_urls['vgg16_bn'] = model_urls['vgg16_bn'].replace('https://', 'http://')\n", '')
    text = text.replace('models.vgg16_bn(pretrained=pretrained)',
                        'models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT if pretrained else None)')
    base.write_text(text)
    yolo = cache / 'yolo11n.pt'
    if not yolo.exists():
        urllib.request.urlretrieve('https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt', yolo)
    craft = cache / 'craft_mlt_25k.pth'
    if not craft.exists():
        with tempfile.TemporaryDirectory(prefix='bus-model-') as tmp:
            archive = Path(tmp) / 'craft.zip'
            urllib.request.urlretrieve('https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip', archive)
            with zipfile.ZipFile(archive) as z:
                craft.write_bytes(z.read('craft_mlt_25k.pth'))
    if hashlib.md5(craft.read_bytes()).hexdigest() != '2f8227d2def4037cdb3b34389dcf9ec1':
        raise RuntimeError('CRAFT 배포 체크섬 불일치')
    for path in [yolo, craft]:
        print(path.name, hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
