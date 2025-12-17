"""
유해 콘텐츠 탐지 모델 평가 스크립트
- 이미지 모델: 박상원 (IMAGE_PARK 기반)
- 비디오 모델: 임영재 (VIDEO_IM 기반)

작성자: 박상원
작성일: 2025년 2학기
"""

import torch
import json
import os
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import clip
from ultralytics import YOLO
from pytorchvideo.models.hub import slowfast_r101

# 환경 변수 설정
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TRANSFORMERS_NO_TF'] = '1'

# GPU 최적화
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

from config import (
    IMAGE_MODEL_PATH, VIDEO_MODEL_PATH, YOLO_MODEL_PATH,
    LABELS_FILE, IMAGE_DIR, SAFE_IMAGE_DIR, VIDEO_DIR, SAFE_VIDEO_DIR,
    DEVICE, CLIP_MODEL_NAME, FRAME_SAMPLE,
    YOLO_DIM, CLIP_DIM, BEHAVIOR_DIM, SLOWFAST_DIM
)
from models import HarmfulImageClassifier, HarmfulVideoClassifier, BEHAVIOR_PROMPTS, BEHAVIOR_CATEGORIES
from inference import predict_image, predict_video, set_clip_text_features_cache, set_clip_weapon_features_cache, WEAPON_PROMPTS


def load_models():
    """모든 모델 로드"""
    print("=" * 60)
    print("모델 로딩 중...")
    print("=" * 60)
    
    # YOLO 모델
    print("1. YOLO 모델 로딩...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    if DEVICE == 'cuda':
        yolo_model.to(DEVICE)
    print(f"   ✓ YOLO 모델 로드 완료")
    
    # CLIP 모델
    print("2. CLIP 모델 로딩...")
    clip_model, clip_preprocess = clip.load(CLIP_MODEL_NAME, device=DEVICE)
    
    # CLIP 텍스트 특징 캐싱
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
    set_clip_text_features_cache(clip_text_features_cache)
    
    # CLIP 무기 특징 캐싱
    print("   CLIP 무기 특징 캐싱 중...")
    weapon_prompts_list = list(WEAPON_PROMPTS.values())
    weapon_tokens = clip.tokenize(weapon_prompts_list).to(DEVICE)
    with torch.no_grad():
        clip_weapon_features_cache = clip_model.encode_text(weapon_tokens)
        clip_weapon_features_cache = torch.nn.functional.normalize(clip_weapon_features_cache, p=2, dim=-1)
    set_clip_weapon_features_cache(clip_weapon_features_cache)
    
    print(f"   ✓ CLIP 모델 로드 완료")
    
    # 이미지 모델
    print("3. 이미지 분류 모델 로딩...")
    image_model = HarmfulImageClassifier(YOLO_DIM, CLIP_DIM, BEHAVIOR_DIM).to(DEVICE)
    checkpoint = torch.load(IMAGE_MODEL_PATH, map_location=DEVICE, weights_only=False)
    image_model.load_state_dict(checkpoint['model_state_dict'])
    image_model_threshold = checkpoint.get('best_threshold', 0.5)
    image_model.eval()
    print(f"   ✓ 이미지 모델 로드 완료: {IMAGE_MODEL_PATH}")
    print(f"   ✓ Best Threshold: {image_model_threshold:.4f}")
    
    print("4. SlowFast 모델 로딩...")
    slowfast_model = slowfast_r101(pretrained=True).to(DEVICE)
    slowfast_model.eval()
    print(f"   ✓ SlowFast R101 모델 로드 완료")
    
    print("5. Transformers CLIP 모델 로딩...")
    try:
        from transformers import CLIPProcessor, CLIPModel
        clip_processor_im = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model_im = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
        print(f"   ✓ Transformers CLIP 모델 로드 완료")
    except Exception as e:
        print(f"   ⚠ Transformers CLIP 모델 로드 실패: {e}")
        clip_processor_im = None
        clip_model_im = None
    
    print("6. Transformers ViT 모델 로딩...")
    try:
        from transformers import AutoImageProcessor, ViTForImageClassification
        import transformers
        import warnings
        transformers.logging.set_verbosity_error()
        warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
        
        vit_processor_im = AutoImageProcessor.from_pretrained("jaranohaal/vit-base-violence-detection")
        vit_model_im = ViTForImageClassification.from_pretrained("jaranohaal/vit-base-violence-detection").to(DEVICE).eval()
        print(f"   ✓ Transformers ViT 모델 로드 완료")
    except Exception as e:
        print(f"   ⚠ Transformers ViT 모델 로드 실패: {e}")
        vit_processor_im = None
        vit_model_im = None
    
    print("7. 비디오 분류 모델 로딩...")
    video_model = HarmfulVideoClassifier(YOLO_DIM, CLIP_DIM, SLOWFAST_DIM, BEHAVIOR_DIM).to(DEVICE)
    video_model_threshold = 0.63
    video_model.eval()
    print(f"   ✓ 비디오 모델 (Fusion 방식) 준비 완료")
    print(f"   ✓ Threshold: {video_model_threshold:.4f}")
    
    print("=" * 60)
    print("✓ 모든 모델 로드 완료!")
    print(f"디바이스: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)
    
    return {
        'yolo': yolo_model,
        'clip': clip_model,
        'clip_preprocess': clip_preprocess,
        'image': image_model,
        'video': video_model,
        'slowfast': slowfast_model,
        'image_threshold': image_model_threshold,
        'video_threshold': video_model_threshold,
        'clip_processor_im': clip_processor_im,
        'clip_model_im': clip_model_im,
        'vit_processor_im': vit_processor_im,
        'vit_model_im': vit_model_im,
    }


def load_data_from_json():
    """JSON 파일에서 데이터 로드 (세 명의 데이터 모두 포함)"""
    print("\n데이터 로딩 중...")
    
    # DATA_ROOT 경로 찾기
    current = Path(__file__).parent
    DATA_ROOT = None
    
    # 방법 1: 상위 폴더에서 "무하유_유해콘텐츠_데이터_모델선정" 찾기
    search_current = current
    while search_current.parent != search_current:
        if search_current.name == "무하유_유해콘텐츠_데이터_모델선정":
            DATA_ROOT = search_current
            break
        search_current = search_current.parent
    
    # 절대 경로로 시도
    if DATA_ROOT is None:
        abs_paths = [
            Path(r"C:\Users\psw20\OneDrive\바탕 화면\PSW\한국항공대학교_3-2\무하유\Github\무하유_유해콘텐츠_데이터_모델선정"),
            Path(r"C:\Users\psw20\OneDrive\바탕 화면\PSW\한국항공대학교_3-2\무하유\DEMO\VIDEO_IM\무하유_유해콘텐츠_데이터_모델선정"),
        ]
        for abs_path in abs_paths:
            if abs_path.exists():
                DATA_ROOT = abs_path
                break
    
    if DATA_ROOT is None:
        raise FileNotFoundError(
            "무하유_유해콘텐츠_데이터_모델선정 폴더를 찾을 수 없습니다.\n"
            "다음 경로 중 하나에 데이터 폴더가 있어야 합니다:\n"
            "- 상위 폴더의 '무하유_유해콘텐츠_데이터_모델선정'\n"
            "- C:\\Users\\psw20\\OneDrive\\바탕 화면\\PSW\\한국항공대학교_3-2\\무하유\\Github\\무하유_유해콘텐츠_데이터_모델선정"
        )
    
    print(f"데이터 루트 경로: {DATA_ROOT}")
    
    # 세 명의 라벨링 파일 경로
    label_files = {
        '박상원': DATA_ROOT / "3_라벨링_파일" / "박상원" / "박상원_labels_categorized.json",
        '안지산': DATA_ROOT / "3_라벨링_파일" / "안지산" / "안지산_labels_categorized.json",
        '임영재': DATA_ROOT / "3_라벨링_파일" / "임영재" / "임영재_labels_categorized.json"
    }
    
    # 각자의 데이터 디렉토리 경로
    data_dirs = {
        '박상원': {
            '이미지': DATA_ROOT / "2_실제_수집_데이터" / "박상원" / "이미지",
            '안전_이미지': DATA_ROOT / "2_실제_수집_데이터" / "박상원" / "안전_이미지",
            '비디오': DATA_ROOT / "2_실제_수집_데이터" / "박상원" / "비디오",
            '안전_비디오': DATA_ROOT / "2_실제_수집_데이터" / "박상원" / "안전_비디오"
        },
        '안지산': {
            '이미지': DATA_ROOT / "2_실제_수집_데이터" / "안지산" / "image",
            '안전_이미지': DATA_ROOT / "2_실제_수집_데이터" / "안지산" / "safe_image",
            '비디오': DATA_ROOT / "2_실제_수집_데이터" / "안지산" / "video",
            '안전_비디오': DATA_ROOT / "2_실제_수집_데이터" / "안지산" / "safe_video"
        },
        '임영재': {
            '이미지': DATA_ROOT / "2_실제_수집_데이터" / "임영재" / "이미지",
            '안전_이미지': DATA_ROOT / "2_실제_수집_데이터" / "임영재" / "안전_이미지",
            '비디오': DATA_ROOT / "2_실제_수집_데이터" / "임영재" / "비디오",
            '안전_비디오': DATA_ROOT / "2_실제_수집_데이터" / "임영재" / "안전_비디오"
        }
    }
    
    image_paths = []
    image_labels = []
    image_categories = []  # 카테고리 정보 추가
    video_paths = []
    video_labels = []
    video_categories = []  # 카테고리 정보 추가
    
    # 각 팀원의 라벨링 파일 읽기
    for member_name, label_file in label_files.items():
        if not label_file.exists():
            print(f"⚠ 경고: {member_name}의 라벨링 파일을 찾을 수 없습니다: {label_file}")
            continue
        
        print(f"  {member_name}의 데이터 로딩 중...")
        with open(label_file, 'r', encoding='utf-8') as f:
            labels_data = json.load(f)
        
        member_dirs = data_dirs[member_name]
        member_image_count = 0
        member_video_count = 0
        
        for filename, info in labels_data.items():
            file_type = info.get('type')
            source_folder = info.get('source_folder')
            is_harmful = info.get('is_harmful', False)
            category = info.get('category', 'unknown')
            label = 1 if is_harmful else 0
            
            if file_type == 'image':
                if source_folder == '이미지':
                    base_dir = member_dirs['이미지']
                elif source_folder == '안전_이미지':
                    base_dir = member_dirs['안전_이미지']
                else:
                    continue
                
                # 먼저 직접 경로 시도
                file_path = base_dir / filename
                
                # 파일이 없으면 하위 폴더에서 재귀적으로 찾기
                if not file_path.exists():
                    found = False
                    for subdir in base_dir.rglob('*'):
                        if subdir.is_file() and subdir.name == filename:
                            file_path = subdir
                            found = True
                            break
                    if not found:
                        print(f"    ⚠ 파일을 찾을 수 없습니다: {filename} (검색 위치: {base_dir})")
                        continue
                
                image_paths.append(str(file_path))
                image_labels.append(label)
                image_categories.append(category)
                member_image_count += 1
            
            elif file_type == 'video':
                if source_folder == '비디오':
                    base_dir = member_dirs['비디오']
                elif source_folder == '안전_비디오':
                    base_dir = member_dirs['안전_비디오']
                else:
                    continue
                
                # 먼저 직접 경로 시도
                file_path = base_dir / filename
                
                # 파일이 없으면 하위 폴더에서 재귀적으로 찾기
                if not file_path.exists():
                    found = False
                    for subdir in base_dir.rglob('*'):
                        if subdir.is_file() and subdir.name == filename:
                            file_path = subdir
                            found = True
                            break
                    if not found:
                        print(f"    ⚠ 파일을 찾을 수 없습니다: {filename} (검색 위치: {base_dir})")
                        continue
                
                video_paths.append(str(file_path))
                video_labels.append(label)
                video_categories.append(category)
                member_video_count += 1
        
        print(f"    ✓ {member_name}: 이미지 {member_image_count}개, 비디오 {member_video_count}개")
    
    print(f"\n✓ 총 이미지 데이터: {len(image_paths)}개 (유해: {sum(image_labels)}, 안전: {len(image_labels) - sum(image_labels)})")
    print(f"✓ 총 비디오 데이터: {len(video_paths)}개 (유해: {sum(video_labels)}, 안전: {len(video_labels) - sum(video_labels)})")
    
    return image_paths, image_labels, image_categories, video_paths, video_labels, video_categories


def evaluate_images(models, image_paths, image_labels, image_categories):
    """이미지 데이터 평가"""
    print("\n" + "=" * 60)
    print("이미지 평가 시작")
    print("=" * 60)
    
    yolo_model = models['yolo']
    clip_model = models['clip']
    clip_preprocess = models['clip_preprocess']
    image_model = models['image']
    threshold = models['image_threshold']
    
    predictions = []
    confidences = []
    true_labels = []
    
    for img_path, true_label in tqdm(zip(image_paths, image_labels), total=len(image_paths), desc="이미지 평가"):
        try:
            image = Image.open(img_path).convert('RGB')
            result = predict_image(image, image_model, yolo_model, clip_model, clip_preprocess, threshold, verbose=False)
            
            predictions.append(1 if result['is_harmful'] else 0)
            confidences.append(result['confidence'])
            true_labels.append(true_label)
        except Exception as e:
            print(f"  오류 ({img_path}): {e}")
            predictions.append(0)
            confidences.append(0.0)
            true_labels.append(true_label)
    
    # 메트릭 계산
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    cm = confusion_matrix(true_labels, predictions)
    
    print("\n이미지 평가 결과:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    [[TN={cm[0][0]}, FP={cm[0][1]}],")
    print(f"     [FN={cm[1][0]}, TP={cm[1][1]}]]")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': predictions,
        'confidences': confidences,
        'true_labels': true_labels,
        'categories': image_categories
    }


def evaluate_videos(models, video_paths, video_labels, video_categories):
    """비디오 데이터 평가"""
    print("\n" + "=" * 60)
    print("비디오 평가 시작")
    print("=" * 60)
    
    yolo_model = models['yolo']
    clip_model = models['clip']
    clip_preprocess = models['clip_preprocess']
    video_model = models['video']
    slowfast_model = models['slowfast']
    threshold = models['video_threshold']
    
    predictions = []
    confidences = []
    true_labels = []
    
    verbose_count = 0
    
    for idx, (video_path, true_label) in enumerate(tqdm(zip(video_paths, video_labels), total=len(video_paths), desc="비디오 평가")):
        try:
            is_verbose = (idx < verbose_count)
            if is_verbose:
                print(f"\n[비디오 {idx+1}/{len(video_paths)}] {Path(video_path).name}")
                print(f"정답: {'유해' if true_label == 1 else '안전'}")
            
            result = predict_video(
                video_path, video_model, yolo_model, slowfast_model, 
                clip_model, clip_preprocess, threshold, verbose=is_verbose,
                clip_processor_im=models.get('clip_processor_im'),
                clip_model_im=models.get('clip_model_im'),
                vit_processor_im=models.get('vit_processor_im'),
                vit_model_im=models.get('vit_model_im')
            )
            
            if is_verbose:
                print(f"예측: {'유해' if result['is_harmful'] else '안전'}, Confidence: {result['confidence']:.4f}")
            
            predictions.append(1 if result['is_harmful'] else 0)
            confidences.append(result['confidence'])
            true_labels.append(true_label)
        except Exception as e:
            print(f"  오류 ({video_path}): {e}")
            predictions.append(0)
            confidences.append(0.0)
            true_labels.append(true_label)
    
    # 메트릭 계산
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    cm = confusion_matrix(true_labels, predictions)
    
    print("\n비디오 평가 결과:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    [[TN={cm[0][0]}, FP={cm[0][1]}],")
    print(f"     [FN={cm[1][0]}, TP={cm[1][1]}]]")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': predictions,
        'confidences': confidences,
        'true_labels': true_labels,
        'categories': video_categories
    }


def analyze_by_category(results, data_type="이미지"):
    """카테고리별 성능 분석 (데이터 불균형 고려)"""
    print("\n" + "=" * 60)
    print(f"{data_type} 카테고리별 성능 분석")
    print("=" * 60)
    
    predictions = results['predictions']
    true_labels = results['true_labels']
    categories = results['categories']
    
    total_samples = len(predictions)
    
    # 카테고리별 데이터 수집
    category_data = {}
    for i, (pred, true_label, category) in enumerate(zip(predictions, true_labels, categories)):
        if category not in category_data:
            category_data[category] = {
                'predictions': [],
                'true_labels': [],
                'total': 0,
                'harmful_count': 0
            }
        category_data[category]['predictions'].append(pred)
        category_data[category]['true_labels'].append(true_label)
        category_data[category]['total'] += 1
        if true_label == 1:
            category_data[category]['harmful_count'] += 1
    
    # 카테고리별 성능 계산
    category_metrics = []
    category_label_map = {
        'safe': '안전',
        'alcohol': '음주',
        'smoking': '흡연',
        'dangerous': '위험행동',
        'blood': '혈액/상처',
        'weapons': '무기',
        'violence': '폭력',
        'threat': '위협',
        'sexual': '성적 콘텐츠',
        'drugs': '약물'
    }
    
    for category, data in sorted(category_data.items()):
        if data['total'] == 0:
            continue
        
        cat_pred = data['predictions']
        cat_true = data['true_labels']
        
        # Precision과 Recall 계산
        tp = sum(1 for p, t in zip(cat_pred, cat_true) if p == 1 and t == 1)
        fp = sum(1 for p, t in zip(cat_pred, cat_true) if p == 1 and t == 0)
        fn = sum(1 for p, t in zip(cat_pred, cat_true) if p == 0 and t == 1)
        tn = sum(1 for p, t in zip(cat_pred, cat_true) if p == 0 and t == 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # 전체 대비 비율 계산
        percentage = (data['total'] / total_samples * 100) if total_samples > 0 else 0.0
        
        category_metrics.append({
            'category': category,
            'label': category_label_map.get(category, category),
            'total': data['total'],
            'percentage': percentage,
            'harmful_count': data['harmful_count'],
            'safe_count': data['total'] - data['harmful_count'],
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'f1': f1
        })
    
    # F1 점수 기준 정렬
    category_metrics.sort(key=lambda x: x['f1'])
    
    # 데이터 불균형 정보 출력
    print(f"\n전체 데이터: {total_samples}개")
    print(f"카테고리별 데이터 분포:")
    print(f"{'카테고리':<15} {'개수':<8} {'비율':<10} {'유해':<8} {'안전':<8}")
    print("-" * 60)
    for metric in sorted(category_metrics, key=lambda x: x['total'], reverse=True):
        print(f"{metric['label']:<15} {metric['total']:<8} {metric['percentage']:>6.2f}%  {metric['harmful_count']:<8} {metric['safe_count']:<8}")
    
    # 데이터가 적은 카테고리 경고
    small_data_categories = [m for m in category_metrics if m['total'] < 20]
    if small_data_categories:
        print(f"\n⚠ 주의: 데이터가 적은 카테고리 (20개 미만, 통계적 신뢰도 낮을 수 있음):")
        for metric in small_data_categories:
            print(f"  - {metric['label']}: {metric['total']}개 ({metric['percentage']:.2f}%)")
    
    # 성능 테이블 출력
    print(f"\n{'카테고리':<15} {'전체':<8} {'비율':<8} {'TP':<6} {'FP':<6} {'FN':<6} {'TN':<6} {'정확도':<10} {'정밀도':<10} {'재현율':<10} {'F1':<10}")
    print("-" * 120)
    
    for metric in category_metrics:
        data_warning = "⚠" if metric['total'] < 20 else " "
        print(f"{data_warning}{metric['label']:<14} {metric['total']:<8} {metric['percentage']:>5.1f}%  "
              f"{metric['tp']:<6} {metric['fp']:<6} {metric['fn']:<6} {metric['tn']:<6} "
              f"{metric['accuracy']:<10.4f} {metric['precision']:<10.4f} {metric['recall']:<10.4f} {metric['f1']:<10.4f}")
    
    # 가장 성능이 낮은 카테고리 출력
    print("\n" + "=" * 60)
    print("⚠ 성능이 낮은 카테고리 분석")
    print("=" * 60)
    print("※ 데이터가 적은 카테고리(<20개)는 성능 지표의 신뢰도가 낮을 수 있습니다.")
    print("※ F1 < 0.7 또는 재현율 < 0.5인 카테고리를 표시합니다.\n")
    
    low_performance = [m for m in category_metrics if m['f1'] < 0.7 or m['recall'] < 0.5]
    if low_performance:
        for metric in low_performance:
            data_reliability = "⚠ 신뢰도 낮음" if metric['total'] < 20 else "✓ 신뢰도 양호"
            print(f"\n{metric['label']} ({metric['category']}) [{data_reliability}]")
            print(f"  - 데이터 수: {metric['total']}개 ({metric['percentage']:.2f}% of 전체)")
            print(f"  - 유해/안전: {metric['harmful_count']}개 / {metric['safe_count']}개")
            print(f"  - F1 점수: {metric['f1']:.4f}")
            print(f"  - 정밀도: {metric['precision']:.4f} (유해로 예측한 것 중 실제 유해 비율)")
            print(f"  - 재현율: {metric['recall']:.4f} (실제 유해인 것 중 유해로 예측한 비율)")
            print(f"  - 정확도: {metric['accuracy']:.4f}")
            print(f"  - 오분류 분석:")
            print(f"    * FP={metric['fp']} (안전을 유해로 잘못 예측)")
            print(f"    * FN={metric['fn']} (유해를 안전으로 잘못 예측)")
            if metric['total'] < 20:
                print(f"    ⚠ 데이터가 적어 성능 지표의 통계적 신뢰도가 낮을 수 있습니다.")
    else:
        print("모든 카테고리의 성능이 양호합니다!")
    
    return category_metrics


def print_summary_table(image_results, video_results):
    """이미지/비디오 모델 성능 비교 테이블 출력"""
    print("\n" + "=" * 80)
    print("모델 성능 비교 요약")
    print("=" * 80)
    print(f"{'지표':<15} {'이미지 모델':<20} {'비디오 모델':<20} {'비고':<25}")
    print("-" * 80)
    
    if image_results and video_results:
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for metric, name in zip(metrics, metric_names):
            img_val = image_results[metric]
            vid_val = video_results[metric]
            diff = vid_val - img_val
            diff_str = f"{diff:+.4f}" if abs(diff) > 0.001 else "≈"
            better = "비디오↑" if diff > 0.01 else ("이미지↑" if diff < -0.01 else "유사")
            
            print(f"{name:<15} {img_val:<20.4f} {vid_val:<20.4f} {better:<25}")
        
        # Confusion Matrix 비교
        img_cm = image_results['confusion_matrix']
        vid_cm = video_results['confusion_matrix']
        
        print("\n" + "-" * 80)
        print("Confusion Matrix 비교:")
        print(f"  이미지: TN={img_cm[0][0]}, FP={img_cm[0][1]}, FN={img_cm[1][0]}, TP={img_cm[1][1]}")
        print(f"  비디오: TN={vid_cm[0][0]}, FP={vid_cm[0][1]}, FN={vid_cm[1][0]}, TP={vid_cm[1][1]}")
    
    elif image_results:
        print(f"{'Accuracy':<15} {image_results['accuracy']:<20.4f} {'N/A':<20}")
        print(f"{'Precision':<15} {image_results['precision']:<20.4f} {'N/A':<20}")
        print(f"{'Recall':<15} {image_results['recall']:<20.4f} {'N/A':<20}")
        print(f"{'F1-Score':<15} {image_results['f1']:<20.4f} {'N/A':<20}")
    
    elif video_results:
        print(f"{'Accuracy':<15} {'N/A':<20} {video_results['accuracy']:<20.4f}")
        print(f"{'Precision':<15} {'N/A':<20} {video_results['precision']:<20.4f}")
        print(f"{'Recall':<15} {'N/A':<20} {video_results['recall']:<20.4f}")
        print(f"{'F1-Score':<15} {'N/A':<20} {video_results['f1']:<20.4f}")


def print_category_summary(category_metrics, data_type="이미지"):
    """카테고리별 성능 요약 (Top/Bottom)"""
    if not category_metrics:
        return
    
    print("\n" + "=" * 80)
    print(f"{data_type} 카테고리별 성능 요약")
    print("=" * 80)
    
    # F1 점수 기준 정렬
    sorted_metrics = sorted(category_metrics, key=lambda x: x['f1'], reverse=True)
    
    # Top 3 성능
    print("\n✓ 성능이 우수한 카테고리 (Top 3):")
    print(f"{'순위':<6} {'카테고리':<15} {'F1':<10} {'Precision':<12} {'Recall':<12} {'데이터 수':<10}")
    print("-" * 80)
    for i, metric in enumerate(sorted_metrics[:3], 1):
        if metric['f1'] > 0:
            print(f"{i:<6} {metric['label']:<15} {metric['f1']:<10.4f} {metric['precision']:<12.4f} {metric['recall']:<12.4f} {metric['total']:<10}")
    
    # Bottom 3 성능
    print("\n⚠ 개선이 필요한 카테고리 (Bottom 3):")
    print(f"{'순위':<6} {'카테고리':<15} {'F1':<10} {'Precision':<12} {'Recall':<12} {'데이터 수':<10}")
    print("-" * 80)
    bottom_metrics = [m for m in sorted_metrics if m['f1'] < 0.7 or m['recall'] < 0.5]
    if bottom_metrics:
        for i, metric in enumerate(bottom_metrics[:3], 1):
            data_warning = "⚠" if metric['total'] < 20 else " "
            print(f"{i:<6} {data_warning}{metric['label']:<14} {metric['f1']:<10.4f} {metric['precision']:<12.4f} {metric['recall']:<12.4f} {metric['total']:<10}")
    else:
        print("  모든 카테고리의 성능이 양호합니다.")


def print_dataset_statistics(image_paths, image_labels, video_paths, video_labels):
    """데이터셋 통계 요약"""
    print("\n" + "=" * 80)
    print("데이터셋 통계")
    print("=" * 80)
    
    if image_paths:
        img_harmful = sum(image_labels)
        img_safe = len(image_labels) - img_harmful
        img_harmful_pct = (img_harmful / len(image_labels) * 100) if image_labels else 0
        
        print(f"\n이미지 데이터:")
        print(f"  총 개수: {len(image_paths)}개")
        print(f"  유해: {img_harmful}개 ({img_harmful_pct:.1f}%)")
        print(f"  안전: {img_safe}개 ({100-img_harmful_pct:.1f}%)")
    
    if video_paths:
        vid_harmful = sum(video_labels)
        vid_safe = len(video_labels) - vid_harmful
        vid_harmful_pct = (vid_harmful / len(video_labels) * 100) if video_labels else 0
        
        print(f"\n비디오 데이터:")
        print(f"  총 개수: {len(video_paths)}개")
        print(f"  유해: {vid_harmful}개 ({vid_harmful_pct:.1f}%)")
        print(f"  안전: {vid_safe}개 ({100-vid_harmful_pct:.1f}%)")
    
    if image_paths and video_paths:
        total = len(image_paths) + len(video_paths)
        total_harmful = sum(image_labels) + sum(video_labels)
        total_safe = total - total_harmful
        print(f"\n전체 데이터:")
        print(f"  총 개수: {total}개")
        print(f"  유해: {total_harmful}개 ({total_harmful/total*100:.1f}%)")
        print(f"  안전: {total_safe}개 ({total_safe/total*100:.1f}%)")


def main():
    """메인 함수"""
    import time
    start_time = time.time()
    
    print("=" * 60)
    print("유해 콘텐츠 탐지 모델 평가 - 카테고리별 분석 (Demo 버전)")
    print("=" * 60)
    
    # 모델 로드
    models = load_models()
    
    # 데이터 로드
    image_paths, image_labels, image_categories, video_paths, video_labels, video_categories = load_data_from_json()
    
    # 데이터셋 통계 출력
    print_dataset_statistics(image_paths, image_labels, video_paths, video_labels)
    
    # 평가 수행
    image_results = None
    video_results = None
    
    if len(image_paths) > 0:
        image_results = evaluate_images(models, image_paths, image_labels, image_categories)
    
    if len(video_paths) > 0:
        video_results = evaluate_videos(models, video_paths, video_labels, video_categories)
    
    # 카테고리별 분석
    image_category_metrics = None
    video_category_metrics = None
    
    if image_results:
        image_category_metrics = analyze_by_category(image_results, "이미지")
        print_category_summary(image_category_metrics, "이미지")
    
    if video_results:
        video_category_metrics = analyze_by_category(video_results, "비디오")
        print_category_summary(video_category_metrics, "비디오")
    
    # 모델 성능 비교 테이블
    print_summary_table(image_results, video_results)
    
    # 최종 요약
    print("\n" + "=" * 80)
    print("최종 평가 요약")
    print("=" * 80)
    
    if image_results:
        print(f"\n이미지 모델:")
        print(f"  Accuracy: {image_results['accuracy']:.4f} ({image_results['accuracy']*100:.2f}%)")
        print(f"  Precision: {image_results['precision']:.4f} ({image_results['precision']*100:.2f}%)")
        print(f"  Recall: {image_results['recall']:.4f} ({image_results['recall']*100:.2f}%)")
        print(f"  F1-Score: {image_results['f1']:.4f}")
    
    if video_results:
        print(f"\n비디오 모델:")
        print(f"  Accuracy: {video_results['accuracy']:.4f} ({video_results['accuracy']*100:.2f}%)")
        print(f"  Precision: {video_results['precision']:.4f} ({video_results['precision']*100:.2f}%)")
        print(f"  Recall: {video_results['recall']:.4f} ({video_results['recall']*100:.2f}%)")
        print(f"  F1-Score: {video_results['f1']:.4f}")
    
    # 실행 시간
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    
    print("\n" + "=" * 80)
    print(f"평가 완료! (소요 시간: {minutes}분 {seconds}초)")
    print("=" * 80)


if __name__ == "__main__":
    main()
