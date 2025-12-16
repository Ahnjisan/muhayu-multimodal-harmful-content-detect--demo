"""
설정 파일 - 유해 콘텐츠 탐지 시스템
모델 경로, 하이퍼파라미터, 디바이스 설정
- 이미지 모델: 박상원 (IMAGE_PARK 기반)
- 비디오 모델: 임영재 (VIDEO_IM 기반)

작성자: 박상원
작성일: 2025년 2학기
"""

import torch
import os
from pathlib import Path

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent

import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 모델 경로 설정 (현재 위치의 weights 폴더 우선 사용)
DEMO_WEIGHTS_DIR = PROJECT_ROOT / "weights"

# 이미지 모델 경로 (현재 위치의 weights 폴더에서 찾기)
IMAGE_MODEL_PATH = DEMO_WEIGHTS_DIR / "image_model_best.pth"
if not IMAGE_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"이미지 모델 파일을 찾을 수 없습니다: {IMAGE_MODEL_PATH}\n"
        f"weights 폴더에 image_model_best.pth 파일이 있어야 합니다."
    )

# 비디오 모델 경로 (Fusion 방식이므로 사용하지 않지만 구조 유지)
VIDEO_MODEL_PATH = DEMO_WEIGHTS_DIR / "video_model_best.pth"

# YOLO 모델 경로
YOLO_MODEL_PATH = DEMO_WEIGHTS_DIR / "yolov8n.pt"
if not YOLO_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"YOLO 모델 파일을 찾을 수 없습니다: {YOLO_MODEL_PATH}\n"
        f"weights 폴더에 yolov8n.pt 파일이 있어야 합니다."
    )

# 데이터 경로 설정
current = PROJECT_ROOT
DATA_ROOT = None

# 방법 1: 상위 폴더에서 "무하유_유해콘텐츠_데이터_모델선정" 찾기
search_current = current
while search_current.parent != search_current:
    if search_current.name == "무하유_유해콘텐츠_데이터_모델선정":
        DATA_ROOT = search_current
        break
    search_current = search_current.parent

# 방법 2: 절대 경로로 시도 (원래 위치)
if DATA_ROOT is None:
    abs_paths = [
        Path(r"C:\Users\psw20\OneDrive\바탕 화면\PSW\한국항공대학교_3-2\무하유\Github\무하유_유해콘텐츠_데이터_모델선정"),
    ]
    for abs_path in abs_paths:
        if abs_path.exists():
            DATA_ROOT = abs_path
            break

# DATA_ROOT이 없어도 에러를 발생시키지 않음 (evaluate_category.py에서 별도로 처리)
if DATA_ROOT is None:
    DATA_ROOT = Path("")  # 빈 경로로 설정 (evaluate_category.py에서 처리)
LABELS_FILE = DATA_ROOT / "3_라벨링_파일" / "박상원" / "박상원_labels_categorized.json"
IMAGE_DIR = DATA_ROOT / "2_실제_수집_데이터" / "박상원" / "이미지"
SAFE_IMAGE_DIR = DATA_ROOT / "2_실제_수집_데이터" / "박상원" / "안전_이미지"
VIDEO_DIR = DATA_ROOT / "2_실제_수집_데이터" / "박상원" / "비디오"
SAFE_VIDEO_DIR = DATA_ROOT / "2_실제_수집_데이터" / "박상원" / "안전_비디오"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Threshold는 체크포인트에서 자동 로드
IMAGE_THRESHOLD = None
VIDEO_THRESHOLD = None

FRAME_SAMPLE = 32  # SlowFast 호환을 위해 32프레임 사용 (final_model11 학습 시와 동일)

CLIP_MODEL_NAME = "ViT-B/32"

SLOWFAST_DIM = 400

# 모델 차원 설정 (카테고리 기반 구조)
YOLO_DIM = 20
CLIP_DIM = 512
BEHAVIOR_DIM = 8

IMAGE_INPUT_DIM = YOLO_DIM + CLIP_DIM + BEHAVIOR_DIM  # 540
VIDEO_INPUT_DIM = YOLO_DIM + CLIP_DIM + SLOWFAST_DIM + BEHAVIOR_DIM  # 940

GRADIO_SERVER_NAME = "0.0.0.0"
GRADIO_SERVER_PORT = 7860
GRADIO_SHARE = True  # Gradio 공개 링크 자동 생성 (외부 접속 가능)

def print_config():
    """설정 정보 출력"""
    print("=" * 60)
    print("설정 정보 (Demo 버전)")
    print("=" * 60)
    print(f"디바이스: {DEVICE}")
    print(f"이미지 모델 경로: {IMAGE_MODEL_PATH}")
    print(f"비디오 모델 경로: {VIDEO_MODEL_PATH}")
    print(f"YOLO 모델 경로: {YOLO_MODEL_PATH}")
    print(f"YOLO 차원: {YOLO_DIM}, CLIP 차원: {CLIP_DIM}, 행동 차원: {BEHAVIOR_DIM}")
    print(f"이미지 입력 차원: {IMAGE_INPUT_DIM}, 비디오 입력 차원: {VIDEO_INPUT_DIM}")
    print(f"Gradio 서버: {GRADIO_SERVER_NAME}:{GRADIO_SERVER_PORT}")
    print("=" * 60)
