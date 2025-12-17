"""
Hazard Killer - 유해 콘텐츠 탐지 웹 애플리케이션
이미지와 비디오에서 유해 콘텐츠를 자동으로 탐지하는 Gradio 기반 웹 인터페이스

작성자: 박상원
작성일: 2025년 2학기
"""

import gradio as gr
import torch
from PIL import Image
import os
import sys
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import HarmfulImageClassifier, HarmfulVideoClassifier
from config import (
    DEVICE, IMAGE_MODEL_PATH, VIDEO_MODEL_PATH, YOLO_MODEL_PATH,
    YOLO_DIM, CLIP_DIM, BEHAVIOR_DIM,
    SLOWFAST_DIM, GRADIO_SERVER_NAME, GRADIO_SERVER_PORT, GRADIO_SHARE,
    CLIP_MODEL_NAME, FRAME_SAMPLE, PROJECT_ROOT
)
from inference import predict_image, predict_video

yolo_model = None
clip_model = None
clip_preprocess = None
image_model = None
video_model = None
slowfast_model = None
image_model_threshold = None
video_model_threshold = None
clip_text_features_cache = None


def load_models():
    """YOLO, CLIP, SlowFast 등 필요한 모든 모델을 메모리에 로드"""
    global yolo_model, clip_model, clip_preprocess, image_model, video_model, slowfast_model
    global image_model_threshold, video_model_threshold
    
    print("=" * 60)
    print("모델 로딩 중...")
    print("=" * 60)
    
    try:
        # YOLO 모델 로드
        print("1. YOLO 모델 로딩...")
        yolo_model = YOLO(str(YOLO_MODEL_PATH))
        if DEVICE == 'cuda':
            yolo_model.to(DEVICE)
        print(f"   YOLO 로드 완료 (디바이스: {DEVICE})")
        
        # CLIP 모델 로드 및 텍스트 특징 캐싱
        print("2. CLIP 모델 로딩...")
        import clip
        from models import BEHAVIOR_PROMPTS, BEHAVIOR_CATEGORIES
        clip_model, clip_preprocess = clip.load(CLIP_MODEL_NAME, device=DEVICE)
        
        global clip_text_features_cache
        print("   CLIP 텍스트 특징 캐싱 중...")
        clip_text_features_cache = {}
        with torch.no_grad():
            use_amp = DEVICE == 'cuda'
            for category in BEHAVIOR_CATEGORIES:
                prompts = BEHAVIOR_PROMPTS[category]
                text_tokens = clip.tokenize(prompts).to(DEVICE)
                
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        text_features = clip_model.encode_text(text_tokens)
                        text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
                else:
                    text_features = clip_model.encode_text(text_tokens)
                    text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
                
                clip_text_features_cache[category] = text_features
        
        from inference import set_clip_text_features_cache, set_clip_weapon_features_cache, WEAPON_PROMPTS
        set_clip_text_features_cache(clip_text_features_cache)
        
        # 무기 특징 캐싱
        print("   CLIP 무기 특징 캐싱 중...")
        weapon_prompts_list = list(WEAPON_PROMPTS.values())
        weapon_tokens = clip.tokenize(weapon_prompts_list).to(DEVICE)
        with torch.no_grad():
            clip_weapon_features_cache = clip_model.encode_text(weapon_tokens)
            clip_weapon_features_cache = torch.nn.functional.normalize(clip_weapon_features_cache, p=2, dim=-1)
        
        set_clip_weapon_features_cache(clip_weapon_features_cache)
        print(f"   CLIP 캐시 완료 ({len(BEHAVIOR_CATEGORIES)}개 카테고리, {len(WEAPON_PROMPTS)}개 무기)")
        
        # 이미지 분류 모델 로드
        print("3. 이미지 분류 모델 로딩...")
        global image_model_threshold
        image_model = HarmfulImageClassifier(YOLO_DIM, CLIP_DIM, BEHAVIOR_DIM).to(DEVICE)
        checkpoint = torch.load(IMAGE_MODEL_PATH, map_location=DEVICE, weights_only=False)
        image_model.load_state_dict(checkpoint['model_state_dict'])
        image_model_threshold = checkpoint.get('best_threshold', 0.5)
        image_model.eval()
        print(f"   이미지 모델 로드 완료 (Threshold: {image_model_threshold:.4f})")
        
        # SlowFast 모델 로드
        print("4. SlowFast 모델 로딩...")
        from pytorchvideo.models.hub import slowfast_r101
        slowfast_model = slowfast_r101(pretrained=True).to(DEVICE)
        slowfast_model.eval()
        print(f"   SlowFast R101 로드 완료")
        
        # 비디오 분류 모델 로드
        print("5. 비디오 분류 모델 로딩...")
        global video_model_threshold
        video_model = HarmfulVideoClassifier(YOLO_DIM, CLIP_DIM, SLOWFAST_DIM, BEHAVIOR_DIM).to(DEVICE)
        video_model_threshold = 0.63
        video_model.eval()
        print(f"   비디오 모델 준비 완료 (Threshold: {video_model_threshold:.4f})")
        
        print("=" * 60)
        print("모든 모델 로드 완료!")
        print(f"디바이스: {DEVICE}")
        if DEVICE == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print("=" * 60)
        
        return True
    
    except Exception as e:
        print(f"모델 로딩 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


def analyze_image(input_image: Image.Image) -> tuple:
    """이미지 유해 콘텐츠 분석 및 결과 반환"""
    if input_image is None:
        return (
            "이미지를 업로드해주세요.",
            "알 수 없음",
            "0.0%",
            "없음",
            "없음",
            "알 수 없음"
        )
    
    try:
        result = predict_image(
            input_image, image_model, yolo_model, 
            clip_model, clip_preprocess, image_model_threshold if image_model_threshold is not None else 0.5
        )
        
        if "error" in result:
            return (
                f"오류 발생: {result['error']}",
                "오류",
                "0.0%",
                "없음",
                "없음",
                "알 수 없음"
            )
        
        # 결과 포맷팅
        status_emoji = "⚠️" if result["is_harmful"] else "✅"
        status_text = "위험" if result["is_harmful"] else "안전"
        status = f"{status_emoji} {status_text}"
        confidence = f"{result['confidence'] * 100:.2f}%"
        
        objects_text = ", ".join(result["detected_objects"]) if result["detected_objects"] else "없음"
        behaviors_text = ", ".join(result["detected_behaviors"]) if result["detected_behaviors"] else "없음"
        
        # 위험도 계산
        if result["confidence"] >= 0.8:
            risk_level = "높음"
        elif result["confidence"] >= 0.5:
            risk_level = "중간"
        else:
            risk_level = "낮음"
        
        model_info = get_model_info()
        result_text = f"""
## 분석 결과

<div style="text-align: center; padding: 30px 0;">
    <div style="font-size: 48px; font-weight: bold; color: {'#FF4444' if result['is_harmful'] else '#00C2FF'}; margin-bottom: 10px;">
        {status_emoji} {status_text}
    </div>
</div>

**사용 모델**: {model_info['image_model']['type']}  
**아키텍처**: {model_info['image_model']['architecture']}
"""
        
        return (
            result_text,
            status,
            confidence,
            objects_text,
            behaviors_text,
            risk_level
        )
    
    except Exception as e:
        error_msg = f"분석 중 오류 발생: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return (
            error_msg,
            "오류",
            "0.0%",
            "없음",
            "없음",
            "알 수 없음"
        )


def analyze_video(input_video: str) -> tuple:
    """비디오 유해 콘텐츠 분석 및 결과 반환"""
    if input_video is None:
        return (
            "비디오를 업로드해주세요.",
            "알 수 없음",
            "0.0%",
            "없음",
            "없음",
            "알 수 없음"
        )
    
    try:
        result = predict_video(
            input_video, video_model, yolo_model, slowfast_model,
            clip_model, clip_preprocess, video_model_threshold if video_model_threshold is not None else 0.4
        )
        
        if "error" in result:
            return (
                f"오류 발생: {result['error']}",
                "오류",
                "0.0%",
                "없음",
                "없음",
                "알 수 없음"
            )
        
        # 결과 포맷팅
        status_emoji = "⚠️" if result["is_harmful"] else "✅"
        status_text = "위험" if result["is_harmful"] else "안전"
        status = f"{status_emoji} {status_text}"
        confidence = f"{result['confidence'] * 100:.2f}%"
        
        objects_text = ", ".join(result["detected_objects"]) if result["detected_objects"] else "없음"
        behaviors_text = ", ".join(result["detected_behaviors"]) if result["detected_behaviors"] else "없음"
        
        # 위험도 계산
        if result["confidence"] >= 0.8:
            risk_level = "높음"
        elif result["confidence"] >= 0.5:
            risk_level = "중간"
        else:
            risk_level = "낮음"
        
        model_info = get_model_info()
        result_text = f"""
## 분석 결과

<div style="text-align: center; padding: 30px 0;">
    <div style="font-size: 48px; font-weight: bold; color: {'#FF4444' if result['is_harmful'] else '#00C2FF'}; margin-bottom: 10px;">
        {status_emoji} {status_text}
    </div>
</div>

**사용 모델**: {model_info['video_model']['type']}  
**아키텍처**: {model_info['video_model']['architecture']}
"""
        
        return (
            result_text,
            status,
            confidence,
            objects_text,
            behaviors_text,
            risk_level
        )
    
    except Exception as e:
        error_msg = f"분석 중 오류 발생: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return (
            error_msg,
            "오류",
            "0.0%",
            "없음",
            "없음",
            "알 수 없음"
        )


def get_model_info():
    """현재 로드된 모델의 상세 정보 반환"""
    model_info = {
        "image_model": {
            "type": "이미지 분류 모델",
            "architecture": "YOLOv8 + CLIP + 행동 인식 + 차원 축소 MLP",
            "input_dim": f"{YOLO_DIM} (YOLO) + {CLIP_DIM} (CLIP) + {BEHAVIOR_DIM} (행동) = {YOLO_DIM + CLIP_DIM + BEHAVIOR_DIM}차원",
            "reduced_dim": "256차원",
            "components": ["YOLOv8 (객체 탐지)", "CLIP ViT-B/32 (맥락 이해)", "Zero-shot 행동 감지", "MLP 분류기"],
            "threshold": image_model_threshold if image_model_threshold is not None else 0.5
        },
        "video_model": {
            "type": "비디오 분류 모델 (Fusion 방식)",
            "architecture": "CLIP + ViT + SlowFast R101 Fusion",
            "input_dim": "32프레임 균등 샘플링",
            "reduced_dim": "Fusion 가중치 결합",
            "components": [
                "CLIP (harmful/benign 프롬프트 비교, 가중치 0.8)",
                "ViT (jaranohaal/vit-base-violence-detection, 가중치 0.1)",
                "SlowFast R101 (Kinetics-400 행동 인식, 가중치 0.1)"
            ],
            "frame_sample": FRAME_SAMPLE,
            "threshold": video_model_threshold if video_model_threshold is not None else 0.63
        },
        "device": DEVICE,
        "yolo_model": "YOLOv8n (nano)",
        "clip_model": CLIP_MODEL_NAME
    }
    return model_info


def create_interface():
    """Gradio 기반 웹 인터페이스 생성 및 설정"""
    
    if not load_models():
        print("모델 로딩 실패. 앱을 종료합니다.")
        sys.exit(1)
    
    model_info = get_model_info()
    
    examples_dir = PROJECT_ROOT / "examples" / "muhayu"
    hazard_logo_path = examples_dir / "Hazard_Killer.png"
    muhayu_logo_path = examples_dir / "muhayu.png"
    favicon_path = examples_dir / "Favicon.jpg"
    
    custom_css = """
    /* === 라이트 모드 (기본) === */
    .gradio-container {
        background: linear-gradient(180deg, #E0F8FF 0%, #EFFBFF 78.85%, #FFF 100%) !important;
        font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, sans-serif !important;
        min-height: 100vh !important;
    }
    
    /* 헤더 - 다크 블루 배경 고정 */
    .header-section {
        background: linear-gradient(180deg, #001B3A 0%, #002958 100%) !important;
        padding: 50px 40px;
        border-radius: 16px;
        margin-bottom: 40px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }
    
    /* 로고 배치 */
    .logo-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 30px;
        gap: 20px;
        flex-wrap: wrap;
        position: relative;
    }
    
    .logo-container .muhayu-logo {
        max-height: 50px;
        object-fit: contain;
        display: block;
        position: absolute;
        left: 0;
        top: 0;
    }
    
    .logo-container .hazard-logo {
        max-height: 80px;
        object-fit: contain;
        display: block;
        margin: 0 auto;
    }
    
    /* QR코드 래퍼 */
    .qr-wrapper {
        position: absolute;
        right: 0;
        top: 0;
        z-index: 10;
    }
    
    /* QR코드 - 오른쪽 상단 */
    .qr-code {
        max-height: 120px;
        max-width: 120px;
        object-fit: contain;
        display: block;
        border-radius: 8px;
        background: white;
        padding: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .qr-code:hover {
        box-shadow: 0 6px 20px rgba(0, 194, 255, 0.5);
        transform: scale(1.05);
    }
    
    /* QR코드 확대 상태 */
    .qr-code.qr-enlarged {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%) scale(2);
        max-height: none;
        max-width: 300px;
        z-index: 9999;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
        animation: qrEnlarge 0.3s ease;
    }
    
    /* 클릭 확대 이미지 (pipeline, research) */
    .clickable-image {
        cursor: pointer;
        transition: transform 0.3s ease;
    }
    
    .clickable-image:hover {
        opacity: 0.9;
    }
    
    .image-enlarged {
        position: fixed !important;
        top: 50% !important;
        left: 50% !important;
        transform: translate(-50%, -50%) scale(1.5) !important;
        z-index: 9999 !important;
        box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.7) !important;
        border-radius: 8px;
        max-width: 70vw !important;
        max-height: 70vh !important;
        width: auto !important;
        height: auto !important;
    }
    
    @media (max-width: 768px) {
        .image-enlarged {
            transform: translate(-50%, -50%) scale(1.2) !important;
            max-width: 85vw !important;
            max-height: 85vh !important;
        }
    }
    
    @keyframes qrEnlarge {
        from {
            transform: translate(-50%, -50%) scale(1);
        }
        to {
            transform: translate(-50%, -50%) scale(2);
        }
    }
    
    /* 타이틀 - 흰색 고정 */
    .hero-title {
        color: #FFFFFF !important;
        font-size: 42px;
        font-weight: bold;
        margin: 20px 0;
        line-height: 1.3;
        text-align: center;
    }
    
    .hero-title * {
        color: #FFFFFF !important;
    }
    
    .hero-title .brand-name,
    .hero-title .highlight {
        color: #00C2FF !important;
    }
    
    /* 설명 텍스트 - 흰색 고정 */
    .hero-description,
    .hero-description *,
    .hero-description p,
    .hero-description p * {
        color: #FFFFFF !important;
    }
    
    .hero-description {
        font-size: 18px;
        line-height: 1.8;
        margin-top: 20px;
        text-align: center;
    }
    
    .hero-description .highlight {
        color: #00C2FF !important;
        font-weight: 600;
    }
    
    /* 카드 - 흰 배경 고정 */
    .card-section {
        background: white !important;
        border-radius: 16px;
        padding: 40px;
        margin: 30px 0;
        box-shadow: rgba(0, 194, 255, 0.15) 0px 0px 32px 0px;
        transition: box-shadow 0.3s ease;
    }
    
    .card-section:hover {
        box-shadow: rgba(0, 194, 255, 0.25) 0px 0px 40px 0px;
    }
    
    /* 카드 내 텍스트 색상 고정 */
    .card-section h2,
    .card-section h3,
    .card-section h4,
    .card-section p,
    .card-section li,
    .card-section strong {
        color: #001B3A !important;
    }
    
    .card-section h2 {
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    
    .card-section h3 {
        font-size: 22px;
        font-weight: 600;
        margin-top: 24px;
        margin-bottom: 16px;
    }
    
    /* 탭 스타일 */
    .tab-nav {
        background: white !important;
        border-radius: 12px;
        padding: 8px;
        margin-bottom: 30px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .tab-nav button {
        border-radius: 8px;
        transition: all 0.3s ease;
        font-weight: 500;
    }
    
    .tab-nav button:hover:not([aria-selected="true"]) {
        background: rgba(0, 194, 255, 0.1) !important;
    }
    
    .tab-nav button[aria-selected="true"] {
        background: #00C2FF !important;
        color: white !important;
        box-shadow: 0 2px 8px rgba(0, 194, 255, 0.3);
    }
    
    /* 버튼 */
    .primary-button {
        background: #00C2FF !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 14px 28px !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        transition: all 0.3s !important;
        box-shadow: 0 4px 12px rgba(0, 194, 255, 0.3) !important;
    }
    
    .primary-button:hover {
        background: #00A8E0 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(0, 194, 255, 0.4) !important;
    }
    
    /* 입력/결과 컨테이너 */
    .input-container,
    .result-container {
        background: white !important;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(0, 194, 255, 0.1);
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    
    .input-container:focus-within {
        border-color: rgba(0, 194, 255, 0.4);
        box-shadow: 0 4px 16px rgba(0, 194, 255, 0.15);
    }
    
    /* Accordion 토글 제목 크기 */
    button.label-wrap,
    button[class*="label-wrap"],
    .label-wrap span:not(.icon),
    button.label-wrap > span:not(.icon),
    button[class*="label-wrap"] > span:not(.icon),
    button[class*="svelte"] span:not([class*="icon"]),
    button.label-wrap[class*="svelte"] span:not([class*="icon"]) {
        font-size: 22px !important;
        font-weight: 700 !important;
        color: #001B3A !important;
        line-height: 1.5 !important;
    }
    
    button.label-wrap,
    button[class*="label-wrap"],
    .label-wrap {
        padding: 16px 20px !important;
        min-height: 60px !important;
        flex-direction: row-reverse !important;
    }
    
    .label-wrap .icon {
        order: -1 !important;
        margin-right: 10px !important;
    }
    
    /* 성능 지표 */
    .metrics-container {
        display: flex;
        flex-direction: column;
        gap: 12px;
        margin: 20px 0;
        position: relative;
    }
    
    /* 스크롤 힌트 (모바일용) */
    .metrics-scroll-hint {
        display: none;
        text-align: center;
        color: #666;
        font-size: 12px;
        margin-top: 8px;
        opacity: 0.7;
    }
    
    /* 평가 데이터셋 통계 그리드 - PC에서 3열 */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 15px;
    }
    
    /* 소개 섹션 - 중앙 정렬 레이아웃 */
    .intro-section-centered {
        max-width: 1000px;
        margin: 0 auto;
        text-align: center;
    }
    
    .intro-section-centered h2 {
        color: #001B3A;
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    
    .intro-lead {
        color: #001B3A;
        font-size: 18px;
        line-height: 1.8;
        margin-bottom: 40px;
    }
    
    /* 그리드 레이아웃 */
    .intro-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 25px;
        margin-bottom: 40px;
    }
    
    .intro-box {
        background: #f8f9fa;
        padding: 25px;
        border-radius: 12px;
        text-align: left;
        border: 2px solid rgba(0, 194, 255, 0.1);
        transition: all 0.3s ease;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .intro-box:hover {
        border-color: rgba(0, 194, 255, 0.3);
        box-shadow: 0 4px 16px rgba(0, 194, 255, 0.15);
    }
    
    .intro-box h3 {
        color: #001B3A;
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 15px;
    }
    
    .intro-box p {
        color: #001B3A;
        line-height: 1.8;
        margin-bottom: 0;
    }
    
    /* 주요 기능 섹션 */
    .intro-features {
        margin-top: 30px;
    }
    
    .intro-features h3 {
        color: #001B3A;
        font-size: 24px;
        font-weight: 600;
        margin-bottom: 25px;
    }
    
    .features-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 20px;
    }
    
    .feature-item {
        background: linear-gradient(135deg, #f8f9fa 0%, #e3f2fd 100%);
        padding: 20px;
        border-radius: 10px;
        text-align: left;
        border: 2px solid rgba(0, 194, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .feature-item:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0, 194, 255, 0.2);
    }
    
    .feature-item strong {
        display: block;
        color: #00C2FF;
        font-size: 16px;
        margin-bottom: 8px;
    }
    
    .feature-item p {
        color: #001B3A;
        font-size: 14px;
        line-height: 1.6;
        margin: 0;
    }
    
    .metrics-row,
    .metrics-row-header {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
    }
    
    .metrics-row-header {
        margin-bottom: 8px;
    }
    
    .metric-header {
        text-align: center;
        font-weight: bold;
        color: #001B3A !important;
        font-size: 14px;
        padding: 8px;
        background: #f0f0f0 !important;
        border-radius: 8px;
    }
    
    .metric-card {
        border-radius: 10px;
        padding: 16px;
        color: white !important;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s, box-shadow 0.3s;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
    }
    
    .metric-card.accuracy {
        background: linear-gradient(135deg, #B3E5FC 0%, #81D4FA 100%) !important;
    }
    
    .metric-card.precision {
        background: linear-gradient(135deg, #C5E1F5 0%, #A8D5F0 100%) !important;
    }
    
    .metric-card.recall {
        background: linear-gradient(135deg, #C8E6C9 0%, #A5D6A7 100%) !important;
    }
    
    .metric-card.f1 {
        background: linear-gradient(135deg, #F8BBD0 0%, #F48FB1 100%) !important;
    }
    
    .metric-card * {
        color: white !important;
    }
    
    .metric-label {
        font-size: 11px;
        font-weight: 600;
        opacity: 0.9;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 2px;
    }
    
    .metrics-section-title {
        text-align: center;
        color: #001B3A !important;
        font-size: 22px;
        font-weight: bold;
        margin: 30px 0 15px 0;
    }
    
    /* 이미지 */
    .pipeline-image,
    .research-image {
        max-width: 500px;
        max-height: 300px;
        margin: 15px 0;
        display: block;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        object-fit: contain;
    }
    
    /* 모델 구조 */
    .model-structure {
        background: #f8f9fa !important;
        border-left: 4px solid #00C2FF;
        padding: 16px;
        margin: 12px 0;
        border-radius: 4px;
    }
    
    .model-structure h4 {
        color: #001B3A !important;
        margin-top: 0;
        margin-bottom: 8px;
        font-size: 16px;
    }
    
    .model-structure code {
        background: white !important;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 13px;
        color: #00C2FF !important;
    }
    
    /* === 모바일 반응형 (768px 이하) === */
    @media (max-width: 768px) {
        .header-section {
            padding: 30px 20px !important;
            border-radius: 12px;
            margin-bottom: 30px;
        }
        
        .logo-container {
            flex-direction: column;
            align-items: center;
            gap: 15px;
        }
        
        .logo-container .muhayu-logo {
            position: static !important;
            max-height: 40px;
            margin-bottom: 10px;
        }
        
        .logo-container .hazard-logo {
            max-height: 60px;
        }
        
        /* QR코드 모바일에서 숨김 */
        .qr-wrapper {
            display: none !important;
        }
        
        /* 소개 섹션 모바일 레이아웃 */
        .intro-grid {
            grid-template-columns: 1fr !important;
            gap: 15px !important;
        }
        
        .features-grid {
            grid-template-columns: 1fr !important;
            gap: 15px !important;
        }
        
        .intro-box,
        .feature-item {
            padding: 18px !important;
        }
        
        .hero-title {
            font-size: 28px !important;
            margin: 15px 0;
            line-height: 1.4;
        }
        
        .hero-description {
            font-size: 15px !important;
            line-height: 1.6;
            padding: 0 10px;
        }
        
        .card-section {
            padding: 20px !important;
            margin: 20px 0;
            border-radius: 12px;
        }
        
        .card-section h2 {
            font-size: 22px !important;
            margin-bottom: 16px;
        }
        
        .card-section h3 {
            font-size: 18px !important;
            margin-top: 20px;
        }
        
        /* 성능 지표를 가로 스크롤 가능하게 */
        .metrics-container {
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            padding-bottom: 10px;
            scroll-snap-type: x mandatory;
        }
        
        /* 스크롤 힌트 표시 */
        .metrics-scroll-hint {
            display: block !important;
        }
        
        /* 스크롤바 스타일링 */
        .metrics-container::-webkit-scrollbar {
            height: 6px;
        }
        
        .metrics-container::-webkit-scrollbar-track {
            background: rgba(0, 194, 255, 0.1);
            border-radius: 3px;
        }
        
        .metrics-container::-webkit-scrollbar-thumb {
            background: rgba(0, 194, 255, 0.4);
            border-radius: 3px;
        }
        
        .metrics-container::-webkit-scrollbar-thumb:hover {
            background: rgba(0, 194, 255, 0.6);
        }
        
        .metrics-row,
        .metrics-row-header {
            min-width: 600px;
            grid-template-columns: repeat(4, 1fr) !important;
            gap: 8px !important;
        }
        
        .metric-header {
            font-size: 11px !important;
            padding: 6px 4px !important;
        }
        
        .metric-card {
            padding: 10px 8px !important;
            min-width: 140px;
        }
        
        .metric-value {
            font-size: 18px !important;
        }
        
        .metric-label {
            font-size: 9px !important;
        }
        
        /* 평가 데이터셋 통계 가로 스크롤 */
        .stats-grid {
            display: flex !important;
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            gap: 12px !important;
            padding-bottom: 10px;
        }
        
        .stats-grid .stats-card {
            min-width: 170px;
            flex-shrink: 0;
        }
        
        /* 평가 데이터셋 통계 스크롤바 */
        .stats-grid::-webkit-scrollbar {
            height: 6px;
        }
        
        .stats-grid::-webkit-scrollbar-track {
            background: rgba(0, 194, 255, 0.1);
            border-radius: 3px;
        }
        
        .stats-grid::-webkit-scrollbar-thumb {
            background: rgba(0, 194, 255, 0.4);
            border-radius: 3px;
        }
        
        /* 카테고리별 성능 요약 가로 스크롤 */
        .category-summary > div[style*="grid-template-columns: 1fr 1fr"] {
            display: flex !important;
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            gap: 15px !important;
            padding-bottom: 10px;
        }
        
        .category-summary > div[style*="grid-template-columns: 1fr 1fr"] > div {
            min-width: 280px;
            flex-shrink: 0;
        }
        
        /* 카테고리 성능 요약 스크롤바 */
        .category-summary > div[style*="grid-template-columns: 1fr 1fr"]::-webkit-scrollbar {
            height: 6px;
        }
        
        .category-summary > div[style*="grid-template-columns: 1fr 1fr"]::-webkit-scrollbar-track {
            background: rgba(0, 194, 255, 0.1);
            border-radius: 3px;
        }
        
        .category-summary > div[style*="grid-template-columns: 1fr 1fr"]::-webkit-scrollbar-thumb {
            background: rgba(0, 194, 255, 0.4);
            border-radius: 3px;
        }
        
        /* Accordion 버튼 */
        button.label-wrap,
        button[class*="label-wrap"],
        .label-wrap span:not(.icon) {
            font-size: 18px !important;
        }
        
        button.label-wrap,
        button[class*="label-wrap"] {
            padding: 12px 15px !important;
            min-height: 50px !important;
        }
        
        /* 분석 버튼 - 터치하기 쉽게 */
        .primary-button {
            width: 100% !important;
            padding: 16px 20px !important;
            font-size: 15px !important;
            min-height: 48px !important;
        }
        
        /* 이미지 */
        .pipeline-image,
        .research-image {
            max-width: 100% !important;
            max-height: 250px !important;
        }
        
        /* 탭 버튼 크기 조정 */
        .tab-nav {
            padding: 6px;
        }
        
        .tab-nav button {
            padding: 10px 16px;
            font-size: 14px;
        }
    }
    
    /* === 다크 모드 === */
    @media (prefers-color-scheme: dark) {
        /* 전체 배경 */
        .gradio-container {
            background: linear-gradient(180deg, #0A1929 0%, #1A2332 78.85%, #1E2A3A 100%) !important;
        }
        
        /* 헤더 - 더 진한 다크 */
        .header-section {
            background: linear-gradient(180deg, #000B1A 0%, #001B3A 100%) !important;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
        }
        
        /* 타이틀과 설명은 그대로 흰색 유지 */
        .hero-title,
        .hero-title * {
            color: #FFFFFF !important;
        }
        
        .hero-title .brand-name,
        .hero-title .highlight {
            color: #00D9FF !important;
        }
        
        .hero-description,
        .hero-description *,
        .hero-description p,
        .hero-description p * {
            color: #E3F2FD !important;
        }
        
        .hero-description .highlight {
            color: #00D9FF !important;
        }
        
        /* 카드 - 다크 배경 */
        .card-section {
            background: #1E2A3A !important;
            box-shadow: rgba(0, 194, 255, 0.2) 0px 0px 32px 0px;
        }
        
        .card-section:hover {
            box-shadow: rgba(0, 194, 255, 0.35) 0px 0px 40px 0px;
        }
        
        /* 카드 내 텍스트 - 밝은 색 */
        .card-section h2,
        .card-section h3,
        .card-section h4,
        .card-section p,
        .card-section li,
        .card-section strong {
            color: #E3F2FD !important;
        }
        
        /* 탭 */
        .tab-nav {
            background: #263545 !important;
        }
        
        .tab-nav button:hover:not([aria-selected="true"]) {
            background: rgba(0, 217, 255, 0.15) !important;
        }
        
        .tab-nav button[aria-selected="true"] {
            background: #00D9FF !important;
            color: #0A1929 !important;
            box-shadow: 0 2px 8px rgba(0, 217, 255, 0.4);
        }
        
        /* 버튼 */
        .primary-button {
            background: #00D9FF !important;
            color: #0A1929 !important;
            box-shadow: 0 4px 12px rgba(0, 217, 255, 0.4) !important;
        }
        
        .primary-button:hover {
            background: #00B8E0 !important;
            box-shadow: 0 6px 16px rgba(0, 217, 255, 0.5) !important;
        }
        
        /* 입력/결과 컨테이너 */
        .input-container,
        .result-container {
            background: #263545 !important;
            border: 1px solid rgba(0, 217, 255, 0.2);
        }
        
        .input-container:focus-within {
            border-color: rgba(0, 217, 255, 0.5);
            box-shadow: 0 4px 16px rgba(0, 217, 255, 0.2);
        }
        
        /* Accordion 버튼 */
        button.label-wrap,
        button[class*="label-wrap"],
        .label-wrap span:not(.icon),
        button.label-wrap > span:not(.icon),
        button[class*="label-wrap"] > span:not(.icon),
        button[class*="svelte"] span:not([class*="icon"]),
        button.label-wrap[class*="svelte"] span:not([class*="icon"]) {
            color: #E3F2FD !important;
        }
        
        /* 성능 지표 헤더 */
        .metric-header {
            color: #E3F2FD !important;
            background: #263545 !important;
        }
        
        .metrics-section-title {
            color: #E3F2FD !important;
        }
        
        /* 모델 구조 */
        .model-structure {
            background: #263545 !important;
            border-left: 4px solid #00D9FF;
        }
        
        .model-structure h4 {
            color: #E3F2FD !important;
        }
        
        .model-structure code {
            background: #1A2332 !important;
            color: #00D9FF !important;
        }
        
        /* 푸터 hr */
        .card-section hr {
            border-top: 1px solid rgba(0, 217, 255, 0.3);
        }
        
        /* 통계 섹션 외부 블록 */
        .stats-section {
            background: #1A2332 !important;
        }
        
        /* 통계 제목 */
        .stats-title,
        .stats-subtitle {
            color: #E3F2FD !important;
        }
        
        /* 통계 카드 - 더 밝은 색 + 테두리 */
        .stats-card {
            background: #2A3F54 !important;
            border: 1px solid rgba(0, 217, 255, 0.3) !important;
        }
        
        /* 카테고리 성능 요약 내부 카드 - 더욱 밝게 */
        .category-summary .stats-card {
            background: #354B65 !important;
            border: 1px solid rgba(0, 217, 255, 0.4) !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
        }
        
        /* 통계 라벨과 상세 */
        .stats-label,
        .stats-detail,
        .stats-note {
            color: #B0BEC5 !important;
        }
        
        /* 통계 값 (큰 숫자) */
        .stats-value {
            color: #00D9FF !important;
        }
        
        /* 통계 아이템 */
        .stats-item {
            color: #E3F2FD !important;
        }
        
        .stats-item strong {
            color: #00D9FF !important;
        }
        
        /* 소개 섹션 다크모드 */
        .intro-section-centered h2,
        .intro-lead {
            color: #E3F2FD !important;
        }
        
        .intro-box {
            background: #263545 !important;
            border-color: rgba(0, 217, 255, 0.3) !important;
        }
        
        .intro-box:hover {
            border-color: rgba(0, 217, 255, 0.5) !important;
        }
        
        .intro-box h3,
        .intro-box p {
            color: #E3F2FD !important;
        }
        
        .intro-features h3 {
            color: #E3F2FD !important;
        }
        
        .feature-item {
            background: linear-gradient(135deg, #263545 0%, #2A3F54 100%) !important;
            border-color: rgba(0, 217, 255, 0.3) !important;
        }
        
        .feature-item strong {
            color: #00D9FF !important;
        }
        
        .feature-item p {
            color: #B0BEC5 !important;
        }
        
        /* 스크롤 힌트 색상 */
        .metrics-scroll-hint {
            color: #B0BEC5 !important;
        }
        
        /* 스크롤바 다크 모드 */
        .metrics-container::-webkit-scrollbar-track,
        .stats-grid::-webkit-scrollbar-track,
        .category-summary > div[style*="grid-template-columns: 1fr 1fr"]::-webkit-scrollbar-track {
            background: rgba(0, 217, 255, 0.1) !important;
        }
        
        .metrics-container::-webkit-scrollbar-thumb,
        .stats-grid::-webkit-scrollbar-thumb,
        .category-summary > div[style*="grid-template-columns: 1fr 1fr"]::-webkit-scrollbar-thumb {
            background: rgba(0, 217, 255, 0.4) !important;
        }
        
        .metrics-container::-webkit-scrollbar-thumb:hover,
        .stats-grid::-webkit-scrollbar-thumb:hover,
        .category-summary > div[style*="grid-template-columns: 1fr 1fr"]::-webkit-scrollbar-thumb:hover {
            background: rgba(0, 217, 255, 0.6) !important;
        }
    }
    
    /* === 작은 모바일 (480px 이하) === */
    @media (max-width: 480px) {
        .header-section {
            padding: 20px 15px !important;
            margin-bottom: 20px;
        }
        
        .logo-container .muhayu-logo {
            max-height: 35px;
        }
        
        .logo-container .hazard-logo {
            max-height: 50px;
        }
        
        .hero-title {
            font-size: 24px !important;
            line-height: 1.3;
        }
        
        .hero-description {
            font-size: 14px !important;
            padding: 0 5px;
        }
        
        .card-section {
            padding: 15px !important;
            margin: 15px 0;
        }
        
        .card-section h2 {
            font-size: 20px !important;
            margin-bottom: 12px;
        }
        
        .card-section h3 {
            font-size: 16px !important;
            margin-top: 16px;
        }
        
        /* 성능 지표는 그대로 가로 스크롤 유지 */
        .metrics-row,
        .metrics-row-header {
            min-width: 500px;
            gap: 6px !important;
        }
        
        .metric-header {
            font-size: 10px !important;
            padding: 5px 3px !important;
        }
        
        .metric-card {
            padding: 8px 6px !important;
            min-width: 120px;
        }
        
        .metric-value {
            font-size: 16px !important;
        }
        
        .metric-label {
            font-size: 8px !important;
        }
        
        /* 평가 데이터셋 통계 카드 더 작게 */
        .stats-grid .stats-card {
            min-width: 160px;
            flex-shrink: 0;
            padding: 12px !important;
        }
        
        .stats-value {
            font-size: 20px !important;
        }
        
        .stats-label,
        .stats-detail {
            font-size: 11px !important;
        }
        
        button.label-wrap,
        button[class*="label-wrap"] {
            padding: 10px 12px !important;
            min-height: 45px !important;
        }
        
        button.label-wrap span:not(.icon),
        button[class*="label-wrap"] span:not(.icon) {
            font-size: 16px !important;
        }
        
        /* 카테고리 성능 요약 작은 모바일에서도 가로 스크롤 유지 */
        .category-summary > div[style*="grid-template-columns: 1fr 1fr"] > div {
            min-width: 260px;
        }
    }
    """
    
    favicon_html = ""
    if favicon_path.exists():
        try:
            import base64 as b64
            with open(favicon_path, 'rb') as f:
                favicon_data = b64.b64encode(f.read()).decode()
                favicon_html = f'<link rel="icon" type="image/jpeg" href="data:image/jpeg;base64,{favicon_data}">'
        except Exception as e:
            print(f"파비콘 로드 실패: {e}")
    
    with gr.Blocks(title="Hazard Killer | 유해 콘텐츠 탐지 시스템") as app:
        with gr.Column(elem_classes=["header-section"]):
            import base64
            
            # QR코드 로드
            qr_code_path = examples_dir / "QRcode.png"
            qr_html = ''
            if qr_code_path.exists():
                try:
                    with open(qr_code_path, 'rb') as f:
                        qr_data = base64.b64encode(f.read()).decode()
                        qr_html = f'''
                        <div class="qr-wrapper">
                            <img class="qr-code" src="data:image/png;base64,{qr_data}" alt="QR Code" onclick="this.classList.toggle('qr-enlarged')" title="클릭하여 확대/축소">
                        </div>
                        '''
                except Exception as e:
                    print(f"QR코드 로드 실패: {e}")
            
            logo_html = '<div class="logo-container">'
            
            # 무하유 로고 로드
            if muhayu_logo_path.exists():
                try:
                    with open(muhayu_logo_path, 'rb') as f:
                        muhayu_logo_data = base64.b64encode(f.read()).decode()
                        logo_html += f'<img class="muhayu-logo" src="data:image/png;base64,{muhayu_logo_data}" alt="무하유">'
                except Exception as e:
                    print(f"무하유 로고 로드 실패: {e}")
            
            # Hazard Killer 로고 로드
            if hazard_logo_path.exists():
                try:
                    with open(hazard_logo_path, 'rb') as f:
                        hazard_logo_data = base64.b64encode(f.read()).decode()
                        logo_html += f'<img class="hazard-logo" src="data:image/png;base64,{hazard_logo_data}" alt="Hazard Killer">'
                except Exception as e:
                    print(f"Hazard Killer 로고 로드 실패: {e}")
            
            logo_html += '</div>'
            
            # QR코드와 로고를 함께 표시
            header_html = qr_html + logo_html
            gr.HTML(header_html)
            
            gr.HTML(
                """
                <div style="text-align: center;">
                    <h1 class="hero-title">
                        <span>웹 어디서나 <span class="brand-name">클릭 한 번</span>으로.</span><br>
                        <span style="display: flex; align-items: center; justify-content: center; gap: 10px;">
                            Hazard Killer
                        </span>
                    </h1>
                    <div class="hero-description">
                        <p style="margin: 0;">더 쉽고 강력해진 유해 콘텐츠 탐지 서비스를 사용해 보세요.<br>
                        <span class="highlight">Hazard Killer</span>는 웹 사이트 어디서나,<br>
                        <span class="highlight">이미지와 비디오를 업로드하여 즉시 검사</span>하며 결과를 확인할 수 있습니다.</p>
                    </div>
                </div>
                """
            )
        
        with gr.Column(elem_classes=["card-section"]):
            content_html = """
            <div class="intro-section-centered">
                <h2>유해 콘텐츠 탐지 서비스</h2>
                
                <p class="intro-lead"><strong>Hazard Killer</strong>는 딥러닝 기반의 이미지 및 비디오 유해 콘텐츠 자동 탐지 시스템입니다.</p>
                
                <div class="intro-grid">
                    <div class="intro-box">
                        <h3>유해 콘텐츠 정의</h3>
                        <p><strong>무기·폭력·음주·흡연·약물·혈액/상처·위협·성적·위험행동</strong> 등<br>
                        사용자에게 위해를 줄 수 있는 <strong>객체·행동 기반 콘텐츠</strong></p>
                    </div>
                    
                    <div class="intro-box">
                        <h3>탐지 항목</h3>
                        <p><strong>무기 탐지 (12종):</strong><br>
                        칼(knife), 단검(dagger), 마체테(machete), 검(sword), 도끼(axe), 총(gun), 권총(pistol), 소총(rifle), 산탄총(shotgun), 기관총(machine_gun), 수류탄(grenade), 폭탄(bomb)</p>
                        
                        <p style="margin-top: 15px;"><strong>유해 행동 탐지 (8개 카테고리):</strong><br>
                        폭력, 음주, 흡연, 약물, 혈액/상처, 위협, 성적 콘텐츠, 위험행동</p>
                    </div>
                </div>
                
                <div class="intro-features">
                    <h3>주요 기능</h3>
                    <div class="features-grid">
                        <div class="feature-item">
                            <strong>실시간 분석</strong>
                            <p>이미지와 비디오를 업로드하면 즉시 분석 결과를 제공합니다</p>
                        </div>
                        <div class="feature-item">
                            <strong>다중 모델 융합</strong>
                            <p>YOLO, CLIP, SlowFast, ViT 등 최신 딥러닝 모델을 활용합니다</p>
                        </div>
                        <div class="feature-item">
                            <strong>정확한 탐지</strong>
                            <p>학습된 모델을 통해 높은 정확도로 유해 콘텐츠를 탐지합니다</p>
                        </div>
                        <div class="feature-item">
                            <strong>상세 정보 제공</strong>
                            <p>탐지된 객체와 행동에 대한 상세 정보를 제공합니다</p>
                        </div>
                    </div>
                </div>
            </div>
            """
            gr.HTML(content_html)
        
        with gr.Column(elem_classes=["card-section"]):
            gr.HTML(
                """
                <div class="metrics-section-title">모델 성능 지표</div>
                
                <div style="margin: 20px 0;">
                    <h3 style="color: #001B3A; font-size: 18px; margin-bottom: 15px;">평가 데이터 성능</h3>
                    <div class="metrics-scroll-hint">← 옆으로 스크롤하세요 →</div>
                    <div class="metrics-container">
                        <div class="metrics-row-header">
                            <div class="metric-header">Accuracy</div>
                            <div class="metric-header">Precision</div>
                            <div class="metric-header">Recall</div>
                            <div class="metric-header">F1-Score</div>
                        </div>
                        <div class="metrics-row">
                            <div class="metric-card accuracy">
                                <div class="metric-label">이미지 모델</div>
                                <div class="metric-value">67.06%</div>
                            </div>
                            <div class="metric-card precision">
                                <div class="metric-label">이미지 모델</div>
                                <div class="metric-value">55.24%</div>
                            </div>
                            <div class="metric-card recall">
                                <div class="metric-label">이미지 모델</div>
                                <div class="metric-value">92.31%</div>
                            </div>
                            <div class="metric-card f1">
                                <div class="metric-label">이미지 모델</div>
                                <div class="metric-value">69.12%</div>
                            </div>
                        </div>
                        <div class="metrics-row">
                            <div class="metric-card accuracy">
                                <div class="metric-label">비디오 모델</div>
                                <div class="metric-value">69.76%</div>
                            </div>
                            <div class="metric-card precision">
                                <div class="metric-label">비디오 모델</div>
                                <div class="metric-value">79.59%</div>
                            </div>
                            <div class="metric-card recall">
                                <div class="metric-label">비디오 모델</div>
                                <div class="metric-value">50.65%</div>
                            </div>
                            <div class="metric-card f1">
                                <div class="metric-label">비디오 모델</div>
                                <div class="metric-value">61.90%</div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="stats-section" style="margin: 30px 0; padding: 20px; background: #f8f9fa; border-radius: 12px;">
                    <h3 class="stats-title" style="color: #001B3A; font-size: 18px; margin-bottom: 15px;">평가 데이터셋 통계</h3>
                    <div class="stats-grid">
                        <div class="stats-card" style="text-align: center; padding: 15px; background: white; border-radius: 8px;">
                            <div class="stats-label" style="font-size: 14px; color: #666; margin-bottom: 5px;">이미지 데이터</div>
                            <div class="stats-value" style="font-size: 24px; font-weight: bold; color: #001B3A;">586개</div>
                            <div class="stats-detail" style="font-size: 12px; color: #666; margin-top: 5px;">유해: 234 (39.9%) / 안전: 352 (60.1%)</div>
                        </div>
                        <div class="stats-card" style="text-align: center; padding: 15px; background: white; border-radius: 8px;">
                            <div class="stats-label" style="font-size: 14px; color: #666; margin-bottom: 5px;">비디오 데이터</div>
                            <div class="stats-value" style="font-size: 24px; font-weight: bold; color: #001B3A;">635개</div>
                            <div class="stats-detail" style="font-size: 12px; color: #666; margin-top: 5px;">유해: 308 (48.5%) / 안전: 327 (51.5%)</div>
                        </div>
                        <div class="stats-card" style="text-align: center; padding: 15px; background: white; border-radius: 8px;">
                            <div class="stats-label" style="font-size: 14px; color: #666; margin-bottom: 5px;">전체 데이터</div>
                            <div class="stats-value" style="font-size: 24px; font-weight: bold; color: #001B3A;">1,221개</div>
                            <div class="stats-detail" style="font-size: 12px; color: #666; margin-top: 5px;">유해: 542 (44.4%) / 안전: 679 (55.6%)</div>
                        </div>
                    </div>
                </div>
                
                <div class="stats-section category-summary" style="margin: 30px 0; padding: 20px; background: #f8f9fa; border-radius: 12px;">
                    <h3 class="stats-title" style="color: #001B3A; font-size: 18px; margin-bottom: 15px;">카테고리별 성능 요약</h3>
                    <div class="metrics-scroll-hint">← 옆으로 스크롤하세요 →</div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                        <div>
                            <h4 class="stats-subtitle" style="color: #001B3A; font-size: 16px; margin-bottom: 10px;">이미지 모델 Top 3</h4>
                            <div class="stats-card" style="background: white; padding: 15px; border-radius: 8px;">
                                <div class="stats-item" style="margin-bottom: 10px;">
                                    <strong>혈액/상처</strong>: F1 1.0000 (Precision 1.0000, Recall 1.0000)
                                </div>
                                <div class="stats-item" style="margin-bottom: 10px;">
                                    <strong>약물</strong>: F1 1.0000 (Precision 1.0000, Recall 1.0000)
                                </div>
                                <div class="stats-item">
                                    <strong>성적 콘텐츠</strong>: F1 1.0000 (Precision 1.0000, Recall 1.0000)
                                </div>
                            </div>
                        </div>
                        <div>
                            <h4 class="stats-subtitle" style="color: #001B3A; font-size: 16px; margin-bottom: 10px;">비디오 모델 Top 3</h4>
                            <div class="stats-card" style="background: white; padding: 15px; border-radius: 8px;">
                                <div class="stats-item" style="margin-bottom: 10px;">
                                    <strong>폭력</strong>: F1 0.9478 (Precision 1.0000, Recall 0.9008)
                                </div>
                                <div class="stats-item" style="margin-bottom: 10px;">
                                    <strong>무기</strong>: F1 0.8000 (Precision 1.0000, Recall 0.6667)
                                </div>
                                <div class="stats-item">
                                    <strong>위협</strong>: F1 0.6429 (Precision 1.0000, Recall 0.4737)
                                </div>
                            </div>
                        </div>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px;">
                        <div>
                            <h4 class="stats-subtitle" style="color: #001B3A; font-size: 16px; margin-bottom: 10px;">이미지 모델 Bottom 3</h4>
                            <div class="stats-card" style="background: white; padding: 15px; border-radius: 8px;">
                                <div class="stats-item" style="margin-bottom: 10px;">
                                    <strong>폭력</strong>: F1 0.1667 (Precision 1.0000, Recall 0.0909)
                                </div>
                                <div class="stats-item" style="margin-bottom: 10px;">
                                    <strong>안전</strong>: F1 0.0000 (Precision 0.0000, Recall 0.0000)
                                </div>
                                <div class="stats-item">
                                    <span class="stats-note" style="font-size: 12px; color: #666;">데이터 부족으로 성능이 낮을 수 있음</span>
                                </div>
                            </div>
                        </div>
                        <div>
                            <h4 class="stats-subtitle" style="color: #001B3A; font-size: 16px; margin-bottom: 10px;">비디오 모델 Bottom 3</h4>
                            <div class="stats-card" style="background: white; padding: 15px; border-radius: 8px;">
                                <div class="stats-item" style="margin-bottom: 10px;">
                                    <strong>약물</strong>: F1 0.2778 (Precision 1.0000, Recall 0.1613)
                                </div>
                                <div class="stats-item" style="margin-bottom: 10px;">
                                    <strong>혈액/상처</strong>: F1 0.5556 (Precision 1.0000, Recall 0.3846)
                                </div>
                                <div class="stats-item">
                                    <strong>위협</strong>: F1 0.6429 (Precision 1.0000, Recall 0.4737)
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                """
            )
        
        with gr.Accordion("프로젝트 상세 정보", open=False):
            with gr.Accordion("최종 모델 구조", open=False):
                with gr.Column(elem_classes=["card-section"]):
                    # 파이프라인 이미지 표시
                    pipeline_path = examples_dir / "pipeline.png"
                    if pipeline_path.exists():
                        try:
                            with open(pipeline_path, 'rb') as f:
                                pipeline_data = base64.b64encode(f.read()).decode()
                                gr.HTML(f'<img src="data:image/png;base64,{pipeline_data}" class="pipeline-image clickable-image" alt="파이프라인" onclick="this.classList.toggle(\'image-enlarged\')" title="클릭하여 확대/축소">')
                        except Exception as e:
                            print(f"파이프라인 이미지 로드 실패: {e}")
                    
                    gr.Markdown(
                        """
                        ### 이미지 모델 (Image Model)
                        <div class="model-structure">
                            <h4>박상원 기반 모델</h4>
                            <code>YOLOv8</code> → <code>CLIP</code> → <code>Behavior Logic</code> → <code>Feature Fusion</code> → <code>Classifier</code>
                        </div>
                        
                        ### 비디오 모델 (Video Model)
                        <div class="model-structure">
                            <h4>임영재 기반 모델</h4>
                            <code>CLIP</code> + <code>ViT</code> + <code>SlowFast</code> → <code>Multimodal Fusion</code>
                        </div>
                        
                        **융합 가중치:**
                        - CLIP: 0.8 (80%)
                        - ViT: 0.1 (10%)
                        - SlowFast: 0.1 (10%)
                        """
                    )
            
            gr.HTML('<hr style="margin: 20px 0; border: none; border-top: 1px solid #e0e0e0;">')
            
            with gr.Accordion("연구 방법", open=False):
                with gr.Column(elem_classes=["card-section"]):
                    # 연구방법 이미지 표시
                    research_path = examples_dir / "research.png"
                    if research_path.exists():
                        try:
                            with open(research_path, 'rb') as f:
                                research_data = base64.b64encode(f.read()).decode()
                                gr.HTML(f'<img src="data:image/png;base64,{research_data}" class="research-image clickable-image" alt="연구방법" onclick="this.classList.toggle(\'image-enlarged\')" title="클릭하여 확대/축소">')
                        except Exception as e:
                            print(f"연구방법 이미지 로드 실패: {e}")
                    
                    gr.Markdown(
                        """
                        ### 연구 진행 방식
                        
                        **초반**
                        - 각 팀원이 객체 기반 유해 콘텐츠 탐지 관련 논문 검토
                        - 이미지/비디오 모델 후보 조사 및 구조 비교
                        - 파일럿 테스트 진행: 작은 샘플 데이터로 모델 성능 확인
                        
                        **중반**
                        - 각자 평가용 데이터 수집 (이미지 200개, 비디오 200개씩) 진행
                        - 매주 중간 공유 및 피드백 세션
                        - 모델 구조, 데이터셋, 성능 비교
                        - 개선 사항 논의 및 적용
                        
                        **후반**
                        - 학습: 공개 데이터셋 활용
                        - 평가: 세 명이 수집한 총 1,200개 데이터 (이미지 600개, 비디오 600개)
                        - 각자의 모델을 통합 평가하여 성능 기준으로 최종 모델 선정
                        - 최종 보고서 작성 및 발표 자료 준비
                        """
                    )
            
            gr.HTML('<hr style="margin: 20px 0; border: none; border-top: 1px solid #e0e0e0;">')
            
            with gr.Accordion("기대효과 및 개선점", open=False):
                with gr.Column(elem_classes=["card-section"]):
                    gr.Markdown(
                        """
                        ### 기대효과
                        - **실제 사용자 기반의 유해 콘텐츠 자동 판별**
                        - **YOLO·CLIP·SlowFast 융합으로 이미지·비디오 동시 처리**
                        - **Recall 향상으로 유해 누락 감소 및 운영 리스크 축소**
                        - **무하유 자체 AI와 연계해 통합 서비스로 확대 가능**
                        
                        ### 개선점
                        - 오탐 감소 위해 Safe 데이터·Hard-Negative 보강
                        - 소수 카테고리용 전용 데이터 확보 및 증강 필요
                        - 비디오 탐지 안정화 위한 Smoothing·Threshold 조정
                        - 서비스 적용 위한 판단 근거(Explainability) 강화
                        """
                    )
        
        with gr.Accordion("상세 모델 정보", open=False):
            with gr.Column(elem_classes=["card-section"]):
                gr.Markdown(
                    f"""
                    ### 이미지 분류 모델
                    - **유형**: {model_info['image_model']['type']}
                    - **아키텍처**: {model_info['image_model']['architecture']}
                    - **입력 차원**: {model_info['image_model']['input_dim']}
                    - **차원 축소 후**: {model_info['image_model']['reduced_dim']}
                    - **구성 요소**: {', '.join(model_info['image_model']['components'])}
                    - **분류 임계값**: {model_info['image_model']['threshold']}
                    
                    ### 비디오 분류 모델
                    - **유형**: {model_info['video_model']['type']}
                    - **아키텍처**: {model_info['video_model']['architecture']}
                    - **입력 차원**: {model_info['video_model']['input_dim']}
                    - **차원 축소 후**: {model_info['video_model']['reduced_dim']}
                    - **프레임 샘플링**: {model_info['video_model']['frame_sample']}개
                    - **구성 요소**: {', '.join(model_info['video_model']['components'])}
                    - **분류 임계값**: {model_info['video_model']['threshold']}
                    
                    ### 실행 환경
                    - **디바이스**: {model_info['device']}
                    """
                )
        
        with gr.Tabs(elem_classes=["tab-nav"]):
            with gr.Tab("이미지 분석"):
                gr.Markdown("### 이미지를 업로드하여 유해 콘텐츠를 탐지합니다.")
                
                with gr.Row():
                    with gr.Column(elem_classes=["input-container"]):
                        image_input = gr.Image(
                            type="pil",
                            label="이미지 업로드",
                            height=400
                        )
                        image_button = gr.Button("분석 시작", variant="primary", size="lg", elem_classes=["primary-button"])
                        
                        example_images = []
                        examples_dir = PROJECT_ROOT / "examples"
                        safe_img_path = examples_dir / "safe_image.jpg"
                        harmful_img_path = examples_dir / "harmful_image.jpg"
                        
                        if safe_img_path.exists():
                            example_images.append(str(safe_img_path))
                        if harmful_img_path.exists():
                            example_images.append(str(harmful_img_path))
                        
                        if example_images:
                            gr.Examples(
                                examples=[[img] for img in example_images],
                                inputs=image_input,
                                label="예시 이미지"
                            )
                    
                    with gr.Column(elem_classes=["result-container"]):
                        image_result = gr.Markdown(label="분석 결과")
                        
                        gr.Markdown("### 상세 정보")
                        image_status = gr.Textbox(label="상태", interactive=False, value="")
                        image_confidence = gr.Textbox(label="유해 확률", interactive=False)
                        image_objects = gr.Textbox(label="감지된 유해 객체", interactive=False)
                        image_behaviors = gr.Textbox(label="감지된 유해 행동", interactive=False)
                        image_risk = gr.Textbox(label="위험도", interactive=False)
                
                image_button.click(
                    fn=analyze_image,
                    inputs=image_input,
                    outputs=[
                        image_result,
                        image_status,
                        image_confidence,
                        image_objects,
                        image_behaviors,
                        image_risk
                    ]
                )
                
            with gr.Tab("비디오 분석"):
                gr.Markdown("### 비디오를 업로드하여 유해 콘텐츠를 탐지합니다.")
                
                with gr.Row():
                    with gr.Column(elem_classes=["input-container"]):
                        video_input = gr.Video(
                            label="비디오 업로드",
                            height=400
                        )
                        video_button = gr.Button("분석 시작", variant="primary", size="lg", elem_classes=["primary-button"])
                        
                        example_videos = []
                        examples_dir = PROJECT_ROOT / "examples"
                        safe_video_path = examples_dir / "safe_video.mp4"
                        harmful_video_path = examples_dir / "harmful_video.mp4"
                        
                        if safe_video_path.exists():
                            example_videos.append(str(safe_video_path))
                        if harmful_video_path.exists():
                            example_videos.append(str(harmful_video_path))
                        
                        if example_videos:
                            gr.Examples(
                                examples=[[vid] for vid in example_videos],
                                inputs=video_input,
                                label="예시 비디오"
                            )
                    
                    with gr.Column(elem_classes=["result-container"]):
                        video_result = gr.Markdown(label="분석 결과")
                        
                        gr.Markdown("### 상세 정보")
                        video_status = gr.Textbox(label="상태", interactive=False, value="")
                        video_confidence = gr.Textbox(label="유해 확률", interactive=False)
                        video_objects = gr.Textbox(label="감지된 유해 객체", interactive=False)
                        video_behaviors = gr.Textbox(label="감지된 유해 행동", interactive=False)
                        video_risk = gr.Textbox(label="위험도", interactive=False)
                
                video_button.click(
                    fn=analyze_video,
                    inputs=video_input,
                    outputs=[
                        video_result,
                        video_status,
                        video_confidence,
                        video_objects,
                        video_behaviors,
                        video_risk
                    ]
                )
        
        with gr.Column(elem_classes=["card-section"]):
            gr.Markdown(
                """
                ### 참고사항
                - 이미지/비디오 분석에는 시간이 소요될 수 있습니다.
                - 모델은 학습 데이터 기반으로 작동하며, 100% 정확도를 보장하지 않습니다.
                - 실제 판단은 전문가의 검토를 권장합니다.
                """
            )
            gr.HTML('<hr style="margin: 20px 0; border: none; border-top: 1px solid #e0e0e0;">')
            gr.HTML(
                """
                <div style="text-align: center; color: #666; padding: 20px;">
                    <p>© 2025 Hazard Killer. All rights reserved.</p>
                    <p style="font-size: 12px;">Powered by 박상원</p>
                </div>
                """
            )
    
    return app, custom_css, favicon_html


if __name__ == "__main__":
    try:
        app, custom_css, favicon_html = create_interface()
        
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
        except:
            local_ip = "확인 불가"
        
        # 대기열 설정 (동시 접속 시 안정성 향상)
        app.queue(
            max_size=20,  # 최대 20개 요청 대기
            default_concurrency_limit=2  # 동시 처리 2개 (GPU 메모리 고려)
        )
        
        app.launch(
            server_name=GRADIO_SERVER_NAME,
            server_port=GRADIO_SERVER_PORT,
            share=GRADIO_SHARE,
            show_error=True,
            css=custom_css,
            theme=gr.themes.Soft(),
            head=favicon_html
        )
        print("\n" + "="*60)
        print("웹 인터페이스 접속 정보:")
        print("="*60)
        print(f"로컬 접속: http://localhost:{GRADIO_SERVER_PORT}")
        print(f"또는: http://127.0.0.1:{GRADIO_SERVER_PORT}")
        
        if GRADIO_SHARE:
            print("\nGradio 공개 링크가 생성되었습니다 (위에 표시된 링크 사용)")
            print("  이 링크를 공유하면 누구나 접속 가능합니다.")
        else:
            print(f"\n내부 네트워크 접속: http://{local_ip}:{GRADIO_SERVER_PORT}")
            print("\n외부 접속을 원하시면 config.py에서 GRADIO_SHARE = True로 설정하세요")
        
        print("="*60)
    except Exception as e:
        print(f"앱 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

