"""
Hazard Killer - 설정 파일
모델 경로, 차원, 디바이스 등 전역 설정 관리

작성자: 박상원
작성일: 2025년 2학기
"""

import torch
import os
from pathlib import Path

# 프로젝트 루트 및 경로 설정
PROJECT_ROOT = Path(__file__).parent

import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 모델 파일 경로
DEMO_WEIGHTS_DIR = PROJECT_ROOT / "weights"

IMAGE_MODEL_PATH = DEMO_WEIGHTS_DIR / "image_model_best.pth"
if not IMAGE_MODEL_PATH.exists():
    raise FileNotFoundError(f"이미지 모델 파일 없음: {IMAGE_MODEL_PATH}")

VIDEO_MODEL_PATH = DEMO_WEIGHTS_DIR / "video_model_best.pth"

YOLO_MODEL_PATH = DEMO_WEIGHTS_DIR / "yolov8n.pt"
if not YOLO_MODEL_PATH.exists():
    raise FileNotFoundError(f"YOLO 모델 파일 없음: {YOLO_MODEL_PATH}")

# 데이터 경로 자동 탐색
current = PROJECT_ROOT
DATA_ROOT = None

# 상위 폴더에서 데이터셋 폴더 찾기
search_current = current
while search_current.parent != search_current:
    if search_current.name == "무하유_유해콘텐츠_데이터_모델선정":
        DATA_ROOT = search_current
        break
    search_current = search_current.parent

# 절대 경로로 재시도
if DATA_ROOT is None:
    abs_path = Path(r"C:\Users\psw20\OneDrive\바탕 화면\PSW\한국항공대학교_3-2\무하유\Github\무하유_유해콘텐츠_데이터_모델선정")
        if abs_path.exists():
            DATA_ROOT = abs_path

# 없으면 빈 경로 설정
if DATA_ROOT is None:
    DATA_ROOT = Path("")

LABELS_FILE = DATA_ROOT / "3_라벨링_파일" / "박상원" / "박상원_labels_categorized.json"
IMAGE_DIR = DATA_ROOT / "2_실제_수집_데이터" / "박상원" / "이미지"
SAFE_IMAGE_DIR = DATA_ROOT / "2_실제_수집_데이터" / "박상원" / "안전_이미지"
VIDEO_DIR = DATA_ROOT / "2_실제_수집_데이터" / "박상원" / "비디오"
SAFE_VIDEO_DIR = DATA_ROOT / "2_실제_수집_데이터" / "박상원" / "안전_비디오"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Threshold는 체크포인트에서 자동 로드
IMAGE_THRESHOLD = None
VIDEO_THRESHOLD = None

# 프레임 샘플링 설정
FRAME_SAMPLE = 32  # SlowFast 모델 입력용

# 모델 이름
CLIP_MODEL_NAME = "ViT-B/32"

# 차원 설정
SLOWFAST_DIM = 400
YOLO_DIM = 20
CLIP_DIM = 512
BEHAVIOR_DIM = 8

IMAGE_INPUT_DIM = YOLO_DIM + CLIP_DIM + BEHAVIOR_DIM
VIDEO_INPUT_DIM = YOLO_DIM + CLIP_DIM + SLOWFAST_DIM + BEHAVIOR_DIM

# Gradio 서버 설정
GRADIO_SERVER_NAME = "0.0.0.0"
GRADIO_SERVER_PORT = 7860
GRADIO_SHARE = True

def print_config():
    """현재 설정 정보 출력"""
    print("=" * 60)
    print("설정 정보")
    print("=" * 60)
    print(f"디바이스: {DEVICE}")
    print(f"이미지 모델: {IMAGE_MODEL_PATH}")
    print(f"비디오 모델: {VIDEO_MODEL_PATH}")
    print(f"YOLO 모델: {YOLO_MODEL_PATH}")
    print(f"차원 - YOLO: {YOLO_DIM}, CLIP: {CLIP_DIM}, 행동: {BEHAVIOR_DIM}")
    print(f"입력 차원 - 이미지: {IMAGE_INPUT_DIM}, 비디오: {VIDEO_INPUT_DIM}")
    print(f"서버: {GRADIO_SERVER_NAME}:{GRADIO_SERVER_PORT}")
    print("=" * 60)
