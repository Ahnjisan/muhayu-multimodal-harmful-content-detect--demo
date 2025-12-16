"""
Gradio 웹앱 메인 파일
유해 콘텐츠 탐지 모델을 웹 인터페이스로 제공

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
    """모든 모델 로드"""
    global yolo_model, clip_model, clip_preprocess, image_model, video_model, slowfast_model
    global image_model_threshold, video_model_threshold
    
    print("=" * 60)
    print("모델 로딩 중...")
    print("=" * 60)
    
    try:
        print("1. YOLO 모델 로딩...")
        yolo_model = YOLO(str(YOLO_MODEL_PATH))
        if DEVICE == 'cuda':
            yolo_model.to(DEVICE)
        print(f"   YOLO 모델 로드 완료: {YOLO_MODEL_PATH} (디바이스: {DEVICE})")
        
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
        
        print("   CLIP 무기 특징 캐싱 중...")
        weapon_prompts_list = list(WEAPON_PROMPTS.values())
        weapon_tokens = clip.tokenize(weapon_prompts_list).to(DEVICE)
        with torch.no_grad():
            clip_weapon_features_cache = clip_model.encode_text(weapon_tokens)
            clip_weapon_features_cache = torch.nn.functional.normalize(clip_weapon_features_cache, p=2, dim=-1)
        
        set_clip_weapon_features_cache(clip_weapon_features_cache)
        
        print(f"   CLIP 모델 로드 완료: {CLIP_MODEL_NAME}")
        print(f"   CLIP 텍스트 특징 캐시 완료 ({len(BEHAVIOR_CATEGORIES)}개 카테고리)")
        print(f"   CLIP 무기 특징 캐시 완료 ({len(WEAPON_PROMPTS)}개 무기)")
        
        print("3. 이미지 분류 모델 로딩...")
        global image_model_threshold
        image_model = HarmfulImageClassifier(YOLO_DIM, CLIP_DIM, BEHAVIOR_DIM).to(DEVICE)
        checkpoint = torch.load(IMAGE_MODEL_PATH, map_location=DEVICE, weights_only=False)
        image_model.load_state_dict(checkpoint['model_state_dict'])
        image_model_threshold = checkpoint.get('best_threshold', 0.5)
        image_model.eval()
        print(f"   이미지 모델 로드 완료: {IMAGE_MODEL_PATH}")
        print(f"   Best Threshold: {image_model_threshold:.4f}")
        
        print("4. SlowFast 모델 로딩...")
        from pytorchvideo.models.hub import slowfast_r101
        slowfast_model = slowfast_r101(pretrained=True).to(DEVICE)
        slowfast_model.eval()
        print(f"   SlowFast R101 모델 로드 완료")
        
        print("5. 비디오 분류 모델 로딩...")
        global video_model_threshold
        video_model = HarmfulVideoClassifier(YOLO_DIM, CLIP_DIM, SLOWFAST_DIM, BEHAVIOR_DIM).to(DEVICE)
        video_model_threshold = 0.63
        video_model.eval()
        print(f"   비디오 모델 (Fusion 방식) 준비 완료")
        print(f"   Threshold: {video_model_threshold:.4f}")
        
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
    """이미지 분석 함수"""
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
        
        status_text = "위험" if result["is_harmful"] else "안전"
        status_emoji = "⚠️" if result["is_harmful"] else "✅"
        status = f"{status_emoji} {status_text}"
        confidence = f"{result['confidence'] * 100:.2f}%"
        
        detected_objects = result["detected_objects"]
        objects_text = ", ".join(detected_objects) if detected_objects else "없음"
        
        detected_behaviors = result["detected_behaviors"]
        behaviors_text = ", ".join(detected_behaviors) if detected_behaviors else "없음"
        
        if result["confidence"] >= 0.8:
            risk_level = "높음"
        elif result["confidence"] >= 0.5:
            risk_level = "중간"
        else:
            risk_level = "낮음"
        
        model_info = get_model_info()
        model_type = model_info['image_model']['type']
        model_arch = model_info['image_model']['architecture']
        
        result_text = f"""
## 분석 결과

<div style="text-align: center; padding: 30px 0;">
    <div style="font-size: 48px; font-weight: bold; color: {'#FF4444' if result['is_harmful'] else '#00C2FF'}; margin-bottom: 10px;">
        {status_emoji} {status_text}
    </div>
</div>

**사용 모델**: {model_type}  
**아키텍처**: {model_arch}
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
    """비디오 분석 함수"""
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
        
        status_text = "위험" if result["is_harmful"] else "안전"
        status_emoji = "⚠️" if result["is_harmful"] else "✅"
        status = f"{status_emoji} {status_text}"
        confidence = f"{result['confidence'] * 100:.2f}%"
        
        detected_objects = result["detected_objects"]
        objects_text = ", ".join(detected_objects) if detected_objects else "없음"
        
        detected_behaviors = result["detected_behaviors"]
        behaviors_text = ", ".join(detected_behaviors) if detected_behaviors else "없음"
        
        if result["confidence"] >= 0.8:
            risk_level = "높음"
        elif result["confidence"] >= 0.5:
            risk_level = "중간"
        else:
            risk_level = "낮음"
        
        model_info = get_model_info()
        model_type = model_info['video_model']['type']
        model_arch = model_info['video_model']['architecture']
        
        result_text = f"""
## 분석 결과

<div style="text-align: center; padding: 30px 0;">
    <div style="font-size: 48px; font-weight: bold; color: {'#FF4444' if result['is_harmful'] else '#00C2FF'}; margin-bottom: 10px;">
        {status_emoji} {status_text}
    </div>
</div>

**사용 모델**: {model_type}  
**아키텍처**: {model_arch}
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
    """모델 정보 반환"""
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
    """Gradio 인터페이스 생성"""
    
    if not load_models():
        print("모델 로딩 실패. 앱을 종료합니다.")
        sys.exit(1)
    
    model_info = get_model_info()
    
    examples_dir = PROJECT_ROOT / "examples" / "muhayu"
    hazard_logo_path = examples_dir / "Hazard_Killer.png"
    muhayu_logo_path = examples_dir / "muhayu.png"
    favicon_path = examples_dir / "Favicon.jpg"
    
    custom_css = """
    /* 전체 배경 그라데이션 */
    .gradio-container {
        background: linear-gradient(180deg, #E0F8FF 0%, #EFFBFF 78.85%, #FFF 100%);
        font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
        min-height: 100vh;
    }
    
    /* 헤더 스타일 */
    .header-section {
        background: linear-gradient(180deg, #001B3A 0%, #002958 100%);
        padding: 50px 40px;
        border-radius: 16px;
        margin-bottom: 40px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }
    
    .logo-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 30px;
        gap: 30px;
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
    
    .hero-title {
        color: #FFFFFF !important;
        font-size: 42px;
        font-weight: bold;
        margin: 20px 0;
        line-height: 1.3;
        text-align: center;
    }
    
    .hero-title .brand-name {
        color: #00C2FF !important;
    }
    
    .hero-description {
        color: #FFFFFF !important;
        font-size: 18px;
        line-height: 1.8;
        margin-top: 20px;
        text-align: center;
    }
    
    .hero-description p {
        color: #FFFFFF !important;
    }
    
    .hero-description .highlight {
        color: #00C2FF !important;
        font-weight: 600;
    }
    
    /* 카드 스타일 */
    .card-section {
        background: white;
        border-radius: 16px;
        padding: 40px;
        margin: 30px 0;
        box-shadow: rgba(0, 194, 255, 0.15) 0px 0px 32px 0px;
    }
    
    /* 탭 스타일 */
    .tab-nav {
        background: white;
        border-radius: 12px;
        padding: 8px;
        margin-bottom: 30px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .tab-nav button {
        border-radius: 8px;
        transition: all 0.3s;
    }
    
    .tab-nav button[aria-selected="true"] {
        background: #00C2FF;
        color: white;
    }
    
    /* 버튼 스타일 */
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
    
    /* 입력 필드 스타일 */
    .input-container {
        background: white;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(0, 194, 255, 0.1);
    }
    
    /* 결과 영역 스타일 */
    .result-container {
        background: white;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(0, 194, 255, 0.1);
    }
    
    /* 마크다운 스타일 개선 */
    .card-section h2 {
        color: #001B3A;
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    
    .card-section h3 {
        color: #001B3A;
        font-size: 22px;
        font-weight: 600;
        margin-top: 24px;
        margin-bottom: 16px;
    }
    
    /* 푸터 스타일 */
    .card-section hr {
        border: none;
        border-top: 1px solid rgba(0, 194, 255, 0.2);
        margin: 30px 0;
    }
    
    /* 토글 제목 크기 키우기 - 실제 DOM 구조 기반 */
    button.label-wrap,
    button[class*="label-wrap"],
    .label-wrap,
    .label-wrap span:not(.icon),
    button.label-wrap > span:not(.icon),
    button[class*="label-wrap"] > span:not(.icon) {
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
    
    /* 토글 아이콘을 앞으로 이동 */
    .label-wrap .icon {
        order: -1 !important;
        margin-right: 10px !important;
    }
    
    /* Svelte 스코프 클래스 대응 */
    button[class*="svelte"] span:not([class*="icon"]),
    button.label-wrap[class*="svelte"] span:not([class*="icon"]) {
        font-size: 22px !important;
        font-weight: 700 !important;
        color: #001B3A !important;
    }
    
    /* 성능 지표 카드 스타일 */
    .metrics-container {
        display: flex;
        flex-direction: column;
        gap: 12px;
        margin: 20px 0;
    }
    
    .metrics-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
    }
    
    .metrics-row-header {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 8px;
    }
    
    .metric-header {
        text-align: center;
        font-weight: bold;
        color: #001B3A;
        font-size: 14px;
        padding: 8px;
        background: #f0f0f0;
        border-radius: 8px;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 16px;
        color: white;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s, box-shadow 0.3s;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
    }
    
    .metric-card.accuracy {
        background: linear-gradient(135deg, #B3E5FC 0%, #81D4FA 100%);
    }
    
    .metric-card.precision {
        background: linear-gradient(135deg, #C5E1F5 0%, #A8D5F0 100%);
    }
    
    .metric-card.recall {
        background: linear-gradient(135deg, #C8E6C9 0%, #A5D6A7 100%);
    }
    
    .metric-card.f1 {
        background: linear-gradient(135deg, #F8BBD0 0%, #F48FB1 100%);
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
    
    .metric-model {
        font-size: 10px;
        opacity: 0.8;
        margin-top: 4px;
    }
    
    .metrics-section-title {
        text-align: center;
        color: #001B3A;
        font-size: 22px;
        font-weight: bold;
        margin: 30px 0 15px 0;
    }
    
    /* 이미지 섹션 스타일 */
    .pipeline-image, .research-image {
        max-width: 500px;
        max-height: 300px;
        margin: 15px 0;
        display: block;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        object-fit: contain;
    }
    
    /* 모델 구조 섹션 */
    .model-structure {
        background: #f8f9fa;
        border-left: 4px solid #00C2FF;
        padding: 16px;
        margin: 12px 0;
        border-radius: 4px;
    }
    
    .model-structure h4 {
        color: #001B3A;
        margin-top: 0;
        margin-bottom: 8px;
        font-size: 16px;
    }
    
    .model-structure code {
        background: white;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 13px;
        color: #00C2FF;
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
            logo_html = '<div class="logo-container" style="position: relative; width: 100%;">'
            if muhayu_logo_path.exists():
                try:
                    with open(muhayu_logo_path, 'rb') as f:
                        muhayu_logo_data = base64.b64encode(f.read()).decode()
                        logo_html += f'<img class="muhayu-logo" src="data:image/png;base64,{muhayu_logo_data}" style="max-height: 50px; object-fit: contain; display: block; position: absolute; left: 0; top: 0;" alt="무하유">'
                except Exception as e:
                    print(f"무하유 로고 로드 실패: {e}")
            if hazard_logo_path.exists():
                try:
                    with open(hazard_logo_path, 'rb') as f:
                        hazard_logo_data = base64.b64encode(f.read()).decode()
                        logo_html += f'<img class="hazard-logo" src="data:image/png;base64,{hazard_logo_data}" style="max-height: 80px; object-fit: contain; display: block; margin: 0 auto;" alt="Hazard Killer">'
                except Exception as e:
                    print(f"Hazard Killer 로고 로드 실패: {e}")
            logo_html += '</div>'
            gr.HTML(logo_html)
            
            gr.HTML(
                """
                <div style="text-align: center;">
                    <h1 class="hero-title" style="color: #FFFFFF !important;">
                        <span style="color: #FFFFFF !important;">웹 어디서나 <span class="brand-name" style="color: #00C2FF !important;">클릭 한 번</span><span style="color: #FFFFFF !important;">으로.</span></span><br>
                        <span style="display: flex; align-items: center; justify-content: center; gap: 10px; color: #FFFFFF !important;">
                            Hazard Killer
                        </span>
                    </h1>
                    <div class="hero-description" style="color: #FFFFFF !important;">
                        <p style="color: #FFFFFF !important; margin: 0;">더 쉽고 강력해진 유해 콘텐츠 탐지 서비스를 사용해 보세요.<br>
                        <span class="highlight" style="color: #00C2FF !important;">Hazard Killer</span>는 웹 사이트 어디서나,<br>
                        <span class="highlight" style="color: #00C2FF !important;">이미지와 비디오를 업로드하여 즉시 검사</span>하며 결과를 확인할 수 있습니다.</p>
                    </div>
                </div>
                """
            )
        
        with gr.Column(elem_classes=["card-section"]):
            gr.Markdown(
                """
                ## 유해 콘텐츠 탐지 서비스
                
                **Hazard Killer**는 딥러닝 기반의 이미지 및 비디오 유해 콘텐츠 자동 탐지 시스템입니다.
                
                ### 유해 콘텐츠 정의
                
                **무기·폭력·음주·흡연·약물·혈액/상처·위협·성적·위험행동** 등  
                사용자에게 위해를 줄 수 있는 **객체·행동 기반 콘텐츠**
                
                ### 탐지 항목
                
                **무기 탐지 (12종):**
                - 칼, 총기, 폭발물, 나이프, 권총, 소총, 폭탄, 수류탄, 화염병, 도끼, 망치, 쇠파이프
                
                **유해 행동 탐지 (8개 카테고리):**
                - 폭력, 음주, 흡연, 약물, 혈액/상처, 위협, 성적 콘텐츠, 위험행동
                
                ### 주요 기능
                - **실시간 분석**: 이미지와 비디오를 업로드하면 즉시 분석 결과를 제공합니다
                - **다중 모델 융합**: YOLO, CLIP, SlowFast, ViT 등 최신 딥러닝 모델을 활용합니다
                - **정확한 탐지**: 학습된 모델을 통해 높은 정확도로 유해 콘텐츠를 탐지합니다
                - **상세 정보 제공**: 탐지된 객체와 행동에 대한 상세 정보를 제공합니다
                """
            )
        
        with gr.Column(elem_classes=["card-section"]):
            gr.HTML(
                """
                <div class="metrics-section-title">모델 성능 지표</div>
                
                <div style="margin: 20px 0;">
                    <h3 style="color: #001B3A; font-size: 18px; margin-bottom: 15px;">평가 데이터 성능</h3>
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
                
                <div style="margin: 30px 0;">
                    <h3 style="color: #001B3A; font-size: 18px; margin-bottom: 15px;">학습 데이터 성능</h3>
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
                                <div class="metric-value">98.44%</div>
                            </div>
                            <div class="metric-card precision">
                                <div class="metric-label">이미지 모델</div>
                                <div class="metric-value">98.04%</div>
                            </div>
                            <div class="metric-card recall">
                                <div class="metric-label">이미지 모델</div>
                                <div class="metric-value">99.68%</div>
                            </div>
                            <div class="metric-card f1">
                                <div class="metric-label">이미지 모델</div>
                                <div class="metric-value">98.86%</div>
                            </div>
                        </div>
                        <div class="metrics-row">
                            <div class="metric-card accuracy">
                                <div class="metric-label">비디오 모델</div>
                                <div class="metric-value">82.83%</div>
                            </div>
                            <div class="metric-card precision">
                                <div class="metric-label">비디오 모델</div>
                                <div class="metric-value">84.03%</div>
                            </div>
                            <div class="metric-card recall">
                                <div class="metric-label">비디오 모델</div>
                                <div class="metric-value">81.21%</div>
                            </div>
                            <div class="metric-card f1">
                                <div class="metric-label">비디오 모델</div>
                                <div class="metric-value">82.59%</div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div style="margin: 30px 0; padding: 20px; background: #f8f9fa; border-radius: 12px;">
                    <h3 style="color: #001B3A; font-size: 18px; margin-bottom: 15px;">데이터셋 통계</h3>
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
                        <div style="text-align: center; padding: 15px; background: white; border-radius: 8px;">
                            <div style="font-size: 14px; color: #666; margin-bottom: 5px;">이미지 데이터</div>
                            <div style="font-size: 24px; font-weight: bold; color: #001B3A;">586개</div>
                            <div style="font-size: 12px; color: #666; margin-top: 5px;">유해: 234 (39.9%) / 안전: 352 (60.1%)</div>
                        </div>
                        <div style="text-align: center; padding: 15px; background: white; border-radius: 8px;">
                            <div style="font-size: 14px; color: #666; margin-bottom: 5px;">비디오 데이터</div>
                            <div style="font-size: 24px; font-weight: bold; color: #001B3A;">635개</div>
                            <div style="font-size: 12px; color: #666; margin-top: 5px;">유해: 308 (48.5%) / 안전: 327 (51.5%)</div>
                        </div>
                        <div style="text-align: center; padding: 15px; background: white; border-radius: 8px;">
                            <div style="font-size: 14px; color: #666; margin-bottom: 5px;">전체 데이터</div>
                            <div style="font-size: 24px; font-weight: bold; color: #001B3A;">1,221개</div>
                            <div style="font-size: 12px; color: #666; margin-top: 5px;">유해: 542 (44.4%) / 안전: 679 (55.6%)</div>
                        </div>
                    </div>
                </div>
                
                <div style="margin: 30px 0; padding: 20px; background: #f8f9fa; border-radius: 12px;">
                    <h3 style="color: #001B3A; font-size: 18px; margin-bottom: 15px;">카테고리별 성능 요약</h3>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                        <div>
                            <h4 style="color: #001B3A; font-size: 16px; margin-bottom: 10px;">이미지 모델 Top 3</h4>
                            <div style="background: white; padding: 15px; border-radius: 8px;">
                                <div style="margin-bottom: 10px;">
                                    <strong>혈액/상처</strong>: F1 1.0000 (Precision 1.0000, Recall 1.0000)
                                </div>
                                <div style="margin-bottom: 10px;">
                                    <strong>약물</strong>: F1 1.0000 (Precision 1.0000, Recall 1.0000)
                                </div>
                                <div>
                                    <strong>성적 콘텐츠</strong>: F1 1.0000 (Precision 1.0000, Recall 1.0000)
                                </div>
                            </div>
                        </div>
                        <div>
                            <h4 style="color: #001B3A; font-size: 16px; margin-bottom: 10px;">비디오 모델 Top 3</h4>
                            <div style="background: white; padding: 15px; border-radius: 8px;">
                                <div style="margin-bottom: 10px;">
                                    <strong>폭력</strong>: F1 0.9478 (Precision 1.0000, Recall 0.9008)
                                </div>
                                <div style="margin-bottom: 10px;">
                                    <strong>무기</strong>: F1 0.8000 (Precision 1.0000, Recall 0.6667)
                                </div>
                                <div>
                                    <strong>위협</strong>: F1 0.6429 (Precision 1.0000, Recall 0.4737)
                                </div>
                            </div>
                        </div>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px;">
                        <div>
                            <h4 style="color: #001B3A; font-size: 16px; margin-bottom: 10px;">이미지 모델 Bottom 3</h4>
                            <div style="background: white; padding: 15px; border-radius: 8px;">
                                <div style="margin-bottom: 10px;">
                                    <strong>폭력</strong>: F1 0.1667 (Precision 1.0000, Recall 0.0909)
                                </div>
                                <div style="margin-bottom: 10px;">
                                    <strong>안전</strong>: F1 0.0000 (Precision 0.0000, Recall 0.0000)
                                </div>
                                <div>
                                    <span style="font-size: 12px; color: #666;">데이터 부족으로 성능이 낮을 수 있음</span>
                                </div>
                            </div>
                        </div>
                        <div>
                            <h4 style="color: #001B3A; font-size: 16px; margin-bottom: 10px;">비디오 모델 Bottom 3</h4>
                            <div style="background: white; padding: 15px; border-radius: 8px;">
                                <div style="margin-bottom: 10px;">
                                    <strong>약물</strong>: F1 0.2778 (Precision 1.0000, Recall 0.1613)
                                </div>
                                <div style="margin-bottom: 10px;">
                                    <strong>혈액/상처</strong>: F1 0.5556 (Precision 1.0000, Recall 0.3846)
                                </div>
                                <div>
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
                    pipeline_path = examples_dir / "pipeline.png"
                    if pipeline_path.exists():
                        try:
                            with open(pipeline_path, 'rb') as f:
                                pipeline_data = base64.b64encode(f.read()).decode()
                                gr.HTML(f'<div style="text-align: left; margin-bottom: 20px;"><img src="data:image/png;base64,{pipeline_data}" class="pipeline-image" alt="파이프라인"></div>')
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
                    research_path = examples_dir / "research.png"
                    if research_path.exists():
                        try:
                            with open(research_path, 'rb') as f:
                                research_data = base64.b64encode(f.read()).decode()
                                gr.HTML(f'<div style="text-align: left; margin-bottom: 20px;"><img src="data:image/png;base64,{research_data}" class="research-image" alt="연구방법"></div>')
                        except Exception as e:
                            print(f"연구방법 이미지 로드 실패: {e}")
                    
                    gr.Markdown(
                        """
                        ### 연구 진행 방식
                        
                        **초반**
                        - 각 팀원이 객체 기반 유해 콘텐츠 탐지 관련 논문 검토
                        - 이미지/영상 모델 후보 조사 및 구조 비교
                        - 파일럿 테스트 진행: 작은 샘플 데이터로 모델 성능 확인
                        
                        **중반**
                        - 각자 데이터 수집 (200개 이미지/영상) 진행
                        - 매주 중간 공유 및 피드백 세션
                        - 모델 구조, 데이터셋, 성능 비교
                        - 개선 사항 논의 및 적용
                        
                        **후반**
                        - 세 명이 수집한 총 600개 데이터 + 공개 데이터셋 통합
                        - 각자의 모델을 통합 평가
                        - 성능 기준으로 최종 모델 선정
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
                        - **YOLO·CLIP·SlowFast 융합으로 이미지·영상 동시 처리**
                        - **Recall 향상으로 유해 누락 감소 및 운영 리스크 축소**
                        - **무하유 자체 AI와 연계해 통합 서비스로 확대 가능**
                        
                        ### 개선점
                        - 오탐 감소 위해 Safe 데이터·Hard-Negative 보강
                        - 소수 카테고리용 전용 데이터 확보 및 증강 필요
                        - 영상 탐지 안정화 위한 Smoothing·Threshold 조정
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
                    
                    ### 공통 모델
                    - **YOLO 모델**: {model_info['yolo_model']}
                    - **CLIP 모델**: {model_info['clip_model']}
                    - **실행 디바이스**: {model_info['device']}
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
            gr.HTML('<hr style="margin: 20px 0; border: none; border-top: 1px solid #e0e0e0;">')
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
        
        app.launch(
            server_name=GRADIO_SERVER_NAME,
            server_port=GRADIO_SERVER_PORT,
            share=GRADIO_SHARE,
            show_error=True,
            max_threads=10,
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

