"""
Hazard Killer - 추론 함수
이미지/비디오에서 유해 콘텐츠 탐지 및 분석

작성자: 박상원
작성일: 2025년 2학기
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import clip
from ultralytics import YOLO
from pytorchvideo.models.hub import slowfast_r50, slowfast_r101
from typing import Dict, List, Tuple, Optional
import torchvision.transforms as T
import time

from models import (
    HARMFUL_OBJECTS, BEHAVIOR_CATEGORIES, BEHAVIOR_PROMPTS,
    CONTEXTUAL_OBJECTS, ALL_OBJECTS, OBJECT_MAP
)
from config import (
    DEVICE, FRAME_SAMPLE, CLIP_MODEL_NAME, SLOWFAST_DIM,
    YOLO_DIM, CLIP_DIM, BEHAVIOR_DIM
)

_clip_text_features_cache = None
_clip_weapon_features_cache = None

def set_clip_text_features_cache(cache):
    """CLIP 텍스트 특징 캐시 설정"""
    global _clip_text_features_cache
    _clip_text_features_cache = cache

def set_clip_weapon_features_cache(cache):
    """CLIP 무기 특징 캐시 설정"""
    global _clip_weapon_features_cache
    _clip_weapon_features_cache = cache

# 무기 탐지용 프롬프트
WEAPON_PROMPTS = {
    'gun': "a photo of a gun",
    'firearm': "a photo of a firearm",
    'knife': "a photo of a knife",
    'blade': "a photo of a blade",
    'weapon': "a photo of a weapon"
}

def detect_weapons_with_clip(clip_features, clip_model, weapon_features_cache=None):
    """CLIP 기반 무기 Zero-shot 탐지"""
    try:
        with torch.no_grad():
            if weapon_features_cache is not None:
                text_features = weapon_features_cache
            else:
                weapon_prompts_list = list(WEAPON_PROMPTS.values())
                text_tokens = clip.tokenize(weapon_prompts_list).to(DEVICE)
                
                use_amp = DEVICE == 'cuda'
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        text_features = clip_model.encode_text(text_tokens)
                        text_features = F.normalize(text_features, p=2, dim=-1)
                else:
                    text_features = clip_model.encode_text(text_tokens)
                    text_features = F.normalize(text_features, p=2, dim=-1)
            
            image_features = clip_features.unsqueeze(0)
            
            use_amp = DEVICE == 'cuda'
            if use_amp:
                with torch.amp.autocast('cuda'):
                    similarities = (image_features @ text_features.T).squeeze()
            else:
                similarities = (image_features @ text_features.T).squeeze()
            
            similarities_np = similarities.cpu().numpy()
            max_idx = similarities_np.argmax()
            weapon_score = float(similarities_np[max_idx])
            weapon_types = list(WEAPON_PROMPTS.keys())
            weapon_type = weapon_types[max_idx]
            
            return weapon_score, weapon_type
            
    except Exception as e:
        return 0.0, None


def detect_behavior_with_clip_fast_optimized(clip_features, clip_model, text_features_cache=None):
    """CLIP 기반 행동 감지 - 카테고리별 여러 프롬프트 지원"""
    behavior_scores = {}
    
    try:
        with torch.no_grad():
            image_features = clip_features.unsqueeze(0)
            
            # 각 카테고리에 대해 여러 프롬프트의 평균 점수 계산
            for category in BEHAVIOR_CATEGORIES:
                prompts = BEHAVIOR_PROMPTS[category]
                
                if text_features_cache is not None and category in text_features_cache:
                    # 캐시된 텍스트 특징 사용
                    text_features = text_features_cache[category]
                else:
                    # 모든 프롬프트를 한 번에 인코딩
                    text_tokens = clip.tokenize(prompts).to(DEVICE)
                    
                    use_amp = DEVICE == 'cuda'
                    if use_amp:
                        with torch.amp.autocast('cuda'):
                            text_features = clip_model.encode_text(text_tokens)
                            text_features = F.normalize(text_features, p=2, dim=-1)
                    else:
                        text_features = clip_model.encode_text(text_tokens)
                        text_features = F.normalize(text_features, p=2, dim=-1)
                
                # 이미지 특징과 모든 프롬프트 특징의 유사도 계산
                use_amp = DEVICE == 'cuda'
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        similarities = (image_features @ text_features.T).squeeze()
                else:
                    similarities = (image_features @ text_features.T).squeeze()
                
                # 여러 프롬프트의 평균 점수
                if len(prompts) == 1:
                    behavior_scores[category] = similarities.item()
                else:
                    behavior_scores[category] = similarities.mean().item()
        
        # Min-Max 정규화
        min_score = min(behavior_scores.values())
        max_score = max(behavior_scores.values())
        if max_score > min_score:
            for category in behavior_scores:
                behavior_scores[category] = (behavior_scores[category] - min_score) / (max_score - min_score)
    
    except Exception as e:
        behavior_scores = {category: 0.0 for category in BEHAVIOR_CATEGORIES}
    
    return behavior_scores


def detect_behavior_with_clip_fast_from_features(clip_features_seq, clip_model, text_features_cache=None):
    """
    CLIP 기반 행동 감지 (비디오용, 이미 추출한 특징 시퀀스 사용) - 카테고리별 여러 프롬프트 지원
    
    Args:
        clip_features_seq: CLIP 특징 시퀀스 (N, 512차원, 정규화됨)
        clip_model: CLIP 모델
        text_features_cache: 캐시된 텍스트 특징 딕셔너리 {category: features} (선택적)
        
    Returns:
        behavior_scores: 카테고리별 점수 딕셔너리 (0~1, Min-Max 정규화)
    """
    behavior_scores = {}
    
    try:
        with torch.no_grad():
            sample_frames = clip_features_seq[:min(len(clip_features_seq), 4)]
            
            # 각 카테고리에 대해 여러 프롬프트의 평균 점수 계산
            for category in BEHAVIOR_CATEGORIES:
                prompts = BEHAVIOR_PROMPTS[category]
                
                if text_features_cache is not None and category in text_features_cache:
                    # 캐시된 텍스트 특징 사용
                    text_features = text_features_cache[category]
                else:
                    # 모든 프롬프트를 한 번에 인코딩
                    text_tokens = clip.tokenize(prompts).to(DEVICE)
                    
                    use_amp = DEVICE == 'cuda'
                    if use_amp:
                        with torch.amp.autocast('cuda'):
                            text_features = clip_model.encode_text(text_tokens)
                            text_features = F.normalize(text_features, p=2, dim=-1)
                    else:
                        text_features = clip_model.encode_text(text_tokens)
                        text_features = F.normalize(text_features, p=2, dim=-1)
                
                # 프레임별 유사도 계산
                use_amp = DEVICE == 'cuda'
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        similarities = sample_frames @ text_features.T
                else:
                    similarities = sample_frames @ text_features.T
                
                # 여러 프롬프트의 평균 점수
                if len(prompts) == 1:
                    frame_scores = similarities[:, 0].cpu().numpy()
                else:
                    frame_scores = similarities.mean(dim=1).cpu().numpy()
                
                behavior_scores[category] = float(np.mean(frame_scores))
        
        # Min-Max 정규화
        min_score = min(behavior_scores.values())
        max_score = max(behavior_scores.values())
        if max_score > min_score:
            for category in behavior_scores:
                behavior_scores[category] = (behavior_scores[category] - min_score) / (max_score - min_score)
    
    except Exception as e:
        behavior_scores = {category: 0.0 for category in BEHAVIOR_CATEGORIES}
    
    return behavior_scores


def detect_behavior_with_clip_fast(image_or_frames, clip_model, clip_preprocess):
    """CLIP 기반 행동 감지 - 카테고리별 여러 프롬프트 지원 (비디오용 원본 버전)"""
    behavior_scores = {}
    
    try:
        # 입력을 리스트로 변환
        if isinstance(image_or_frames, Image.Image):
            frames = [image_or_frames]
        else:
            frames = image_or_frames
        
        # 각 카테고리에 대해 CLIP 유사도 계산 (여러 프롬프트 평균)
        for category in BEHAVIOR_CATEGORIES:
            prompts = BEHAVIOR_PROMPTS[category]  # 프롬프트 리스트
            
            category_scores = []
            for frame in frames[:min(len(frames), 4)]:
                image_input = clip_preprocess(frame).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    # 이미지 특징 추출
                    image_features = clip_model.encode_image(image_input)
                    image_features = F.normalize(image_features, p=2, dim=-1)
                    
                    # 모든 프롬프트에 대한 유사도 계산
                    prompt_scores = []
                    for prompt in prompts:
                        text = clip.tokenize([prompt]).to(DEVICE)
                        text_features = clip_model.encode_text(text)
                        text_features = F.normalize(text_features, p=2, dim=-1)
                        similarity = (image_features @ text_features.T).squeeze()
                        prompt_scores.append(similarity.item())
                    
                    category_scores.append(np.mean(prompt_scores))
            
            behavior_scores[category] = np.mean(category_scores)
        
        # 점수 정규화
        min_score = min(behavior_scores.values())
        max_score = max(behavior_scores.values())
        if max_score > min_score:
            for category in behavior_scores:
                behavior_scores[category] = (behavior_scores[category] - min_score) / (max_score - min_score)
    
    except Exception as e:
        print(f"  [행동 감지 오류] {e}")
        # 에러 시 0으로 설정
        behavior_scores = {category: 0.0 for category in BEHAVIOR_CATEGORIES}
    
    return behavior_scores


def infer_behavior_from_objects(object_counts: Dict[str, int]) -> List[str]:
    """
    YOLO 객체 카운트 기반으로 유해 카테고리 추론 (규칙 기반 보조 로직)
    CLIP의 Zero-shot 감지를 보완하는 역할
    """
    inferred = []
    
    # 규칙 1: 담배 감지 → 흡연
    # - lighter 단독은 흡연으로 보지 않음
    if object_counts.get("cigarette", 0) > 0:
        inferred.append("smoking")
    
    # 규칙 2: 음주 관련 객체 1개 이상 → 음주
    # - wine glass / beer 등
    alcohol_objects = OBJECT_MAP.get("alcohol", [])
    if sum(object_counts.get(obj, 0) for obj in alcohol_objects) >= 1:
        inferred.append("alcohol")
    
    # 규칙 3: 주사기 감지 → 약물
    drug_objects = OBJECT_MAP.get("drugs", [])
    if sum(object_counts.get(obj, 0) for obj in drug_objects) > 0:
        inferred.append("drugs")
    
    # 규칙 4: 혈액/상처 관련 객체 감지 → blood
    blood_objects = OBJECT_MAP.get("blood", [])
    if sum(object_counts.get(obj, 0) for obj in blood_objects) > 0:
        inferred.append("blood")
    
    # 규칙 5: 무기 + 사람 → 위협
    weapon_objects = OBJECT_MAP.get("weapons", [])
    weapon_count = sum(object_counts.get(obj, 0) for obj in weapon_objects)
    person_count = object_counts.get("person", 0)
    if weapon_count > 0 and person_count >= 1:
        inferred.append("threat")
    
    # 중복 제거
    inferred = list(set(inferred))
    
    return inferred


def extract_yolo_features(yolo_results) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    YOLO 탐지 결과에서 특징 벡터 추출
    
    Args:
        yolo_results: YOLO 탐지 결과 (단일 결과 또는 리스트)
    
    Returns:
        feature_vector: 객체별 탐지 개수 벡터 (20차원, ALL_OBJECTS 개수)
        object_counts: 객체별 탐지 개수 딕셔너리
    """
    feature_vector = torch.zeros(YOLO_DIM, device=DEVICE)
    object_counts = {}
    
    if not isinstance(yolo_results, list):
        yolo_results = [yolo_results]
    
    for result in yolo_results:
        if result.boxes is not None:
            for box in result.boxes:
                class_name = result.names[int(box.cls)].lower()
                for i, obj in enumerate(ALL_OBJECTS):
                    if obj in class_name or class_name in obj:
                        feature_vector[i] += 1
                        object_counts[obj] = object_counts.get(obj, 0) + 1
    
    return feature_vector, object_counts


def predict_image(image: Image.Image, image_model, yolo_model, clip_model,
                  clip_preprocess, threshold: float = 0.4, verbose: bool = True) -> Dict:
    """이미지에 대한 유해 콘텐츠 탐지"""
    try:
        if image is None:
            return {
                "is_harmful": False,
                "confidence": 0.0,
                "detected_objects": [],
                "detected_behaviors": [],
                "error": "이미지를 불러올 수 없습니다."
            }
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_np = np.array(image)
        yolo_results = yolo_model(image_np, verbose=False, device=DEVICE, imgsz=640, conf=0.25)
        yolo_features, object_counts = extract_yolo_features(yolo_results)
        
        detected_objects = [obj for obj in HARMFUL_OBJECTS if object_counts.get(obj, 0) > 0]
        
        clip_image = clip_preprocess(image).unsqueeze(0).to(DEVICE, non_blocking=True)
        with torch.no_grad():
            if DEVICE == 'cuda':
                with torch.amp.autocast('cuda'):
                    clip_features = clip_model.encode_image(clip_image).squeeze()
            else:
                clip_features = clip_model.encode_image(clip_image).squeeze()
            clip_features = F.normalize(clip_features, p=2, dim=-1)
        
        # CLIP 무기 감지 (Zero-shot)
        weapon_score, weapon_type = detect_weapons_with_clip(
            clip_features, clip_model, weapon_features_cache=_clip_weapon_features_cache
        )
        
        behavior_scores = detect_behavior_with_clip_fast_optimized(
            clip_features, clip_model, text_features_cache=_clip_text_features_cache
        )
        inferred_categories = infer_behavior_from_objects(object_counts)
        
        behavior_features = torch.zeros(len(BEHAVIOR_CATEGORIES), device=DEVICE)
        for i, category in enumerate(BEHAVIOR_CATEGORIES):
            clip_score = behavior_scores.get(category, 0.0)
            rule_score = 1.0 if category in inferred_categories else 0.0
            behavior_features[i] = 0.6 * clip_score + 0.4 * rule_score
        
        # detected_behaviors: 규칙 기반 + 매우 높은 CLIP 점수
        # violence는 일상 동작과 혼동되어 제외
        # dangerous, sexual은 더 보수적으로 (오탐 심각)
        detected_behaviors = inferred_categories.copy()
        for category, score in behavior_scores.items():
            if category == 'violence':
                continue  # violence는 제외 (False Positive 너무 많음)
            elif category == 'dangerous' and score >= 0.99:
                # dangerous는 0.99 이상만 (거의 확실한 경우만)
                detected_behaviors.append(category)
            elif category == 'sexual' and score >= 0.98:
                # sexual은 0.98 이상만
                detected_behaviors.append(category)
            elif score >= 0.95 and category not in detected_behaviors:
                # 나머지 카테고리는 0.95 이상
                detected_behaviors.append(category)
        
        yolo_features = yolo_features.to(DEVICE)
        combined = torch.cat([yolo_features, clip_features, behavior_features]).unsqueeze(0)
        
        image_model.eval()
        with torch.no_grad():
            if DEVICE == 'cuda':
                with torch.amp.autocast('cuda'):
                    confidence = image_model(combined).item()
            else:
                confidence = image_model(combined).item()
        
        # CLIP 무기 감지 (Zero-shot): 이미지는 0.3, 비디오는 0.2
        weapon_threshold = 0.3  # 이미지용
        if weapon_score >= weapon_threshold and weapon_type:
            if weapon_type not in detected_objects:
                detected_objects.append(weapon_type)
            if verbose:
                print(f"  [CLIP 무기 감지] {weapon_type}: {weapon_score:.3f}")
        
        # 무기 휴리스틱: knife + threat 또는 CLIP 무기 감지
        weapon_detected = False
        if weapon_score >= weapon_threshold:
            weapon_detected = True
            is_harmful = True
        elif 'knife' in detected_objects and 'threat' in detected_behaviors:
            weapon_detected = True
            is_harmful = True
        
        # 균형잡힌 분류 로직
        if not weapon_detected:
            if len(detected_objects) == 0 and len(detected_behaviors) == 0:
                # 아무것도 감지 안 됨 → 안전
                is_harmful = False
            else:
                # 객체나 행동이 감지됨 → threshold 사용
                # threat만 더 민감하게 반응 (threshold 30% 낮춤)
                if 'threat' in detected_behaviors:
                    adjusted_threshold = threshold * 0.7
                else:
                    adjusted_threshold = threshold
                is_harmful = confidence > adjusted_threshold
        
        return {
            "is_harmful": is_harmful,
            "confidence": confidence,
            "detected_objects": detected_objects,
            "detected_behaviors": detected_behaviors
        }
    
    except Exception as e:
        print(f"이미지 추론 오류: {e}")
        return {
            "is_harmful": False,
            "confidence": 0.0,
            "detected_objects": [],
            "detected_behaviors": [],
            "error": str(e)
        }


# 비디오 모델용 함수들 (Fusion 방식)
IM_CLIP_HARMFUL_PROMPTS = [
    "a violent scene with people fighting, punching or kicking",
    "a person shooting a gun at another person",
    "visible blood, gore or serious injury",
    "a person holding a weapon in a threatening or aggressive way",
    "a brutal fight scene from an action movie",
    "an explicit violent scene that should not be shown to children",
]

IM_CLIP_BENIGN_PROMPTS = [
    "people calmly talking with no fighting or violence",
    "a person holding a harmless everyday object, no threat",
    "no blood or injury, just normal healthy people",
    "a person holding tools or everyday items in a safe way",
    "a normal peaceful scene with people standing or walking",
    "a safe and non-violent scene that is appropriate for all ages",
]

# SlowFast violence 키워드
SLOWFAST_VIOLENCE_KEYWORDS = [
    "fight", "fighting", "punch", "hit", "kick", "attack",
    "shoot", "shooting", "gun", "weapon", "stab", "strangle",
    "choke", "beat", "assault", "violence"
]


def compute_clip_violence_score_im(frames_pil: List[Image.Image], clip_model, clip_preprocess, 
                                   clip_processor_im=None, clip_model_im=None, verbose: bool = False) -> float:
    """CLIP harmful/benign 프롬프트 비교로 violence score 계산 (p95 사용)"""
    try:
        if clip_processor_im is None or clip_model_im is None:
            from transformers import CLIPProcessor, CLIPModel
            import os
            os.environ["TRANSFORMERS_NO_TF"] = "1"
            
            try:
                clip_processor_im = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                clip_model_im = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
            except:
                return compute_clip_violence_score_fallback(frames_pil, clip_model, clip_preprocess)
        
        texts = IM_CLIP_HARMFUL_PROMPTS + IM_CLIP_BENIGN_PROMPTS
        num_harm = len(IM_CLIP_HARMFUL_PROMPTS)
        
        violence_probs = []
        temperature = 2.0
        batch_size = 16
        for i in range(0, len(frames_pil), batch_size):
            chunk = frames_pil[i:i + batch_size]
            
            inputs = clip_processor_im(
                text=texts,
                images=chunk,
                return_tensors="pt",
                padding=True,
            ).to(DEVICE)
            
            with torch.no_grad():
                if DEVICE == 'cuda':
                    with torch.amp.autocast('cuda'):
                        outputs = clip_model_im(**inputs)
                        logits = outputs.logits_per_image  # (B, T)
                else:
                    outputs = clip_model_im(**inputs)
                    logits = outputs.logits_per_image
            
            logits = logits / temperature
            probs = logits.softmax(dim=-1).cpu().numpy()
            
            for prob_vec in probs:
                harm_prob = float(np.sum(prob_vec[:num_harm]))
                harm_prob = float(np.clip(harm_prob, 0.0, 1.0))
                violence_probs.append(harm_prob)
        
        if not violence_probs:
            return 0.0
        
        return float(np.percentile(violence_probs, 95))
    
    except Exception as e:
        if verbose:
            print(f"  [CLIP violence 계산 오류] {e}, fallback 사용")
        return compute_clip_violence_score_fallback(frames_pil, clip_model, clip_preprocess)


def compute_clip_violence_score_fallback(frames_pil: List[Image.Image], clip_model, clip_preprocess) -> float:
    """기존 clip 라이브러리 사용 (fallback)"""
    try:
        # 샘플링: 최대 4개 프레임만 사용
        sample_frames = frames_pil[:min(len(frames_pil), 4)]
        
        clip_images = torch.stack([clip_preprocess(frame) for frame in sample_frames]).to(DEVICE, non_blocking=True)
        with torch.no_grad():
            if DEVICE == 'cuda':
                with torch.amp.autocast('cuda'):
                    clip_features = clip_model.encode_image(clip_images)
            else:
                clip_features = clip_model.encode_image(clip_images)
            clip_features = F.normalize(clip_features, p=2, dim=-1)
        
        # harmful 프롬프트와의 유사도 계산
        harm_scores = []
        for prompt in IM_CLIP_HARMFUL_PROMPTS:
            text = clip.tokenize([prompt]).to(DEVICE)
            text_features = clip_model.encode_text(text)
            text_features = F.normalize(text_features, p=2, dim=-1)
            similarity = (clip_features @ text_features.T).squeeze()
            harm_scores.append(similarity.mean().item())
        
        return float(np.mean(harm_scores)) if harm_scores else 0.0
    except:
        return 0.0


def compute_vit_violence_score_im(frames_pil: List[Image.Image], 
                                  vit_processor_im=None, vit_model_im=None, verbose: bool = False) -> float:
    """ViT (jaranohaal/vit-base-violence-detection)로 violence score 계산 (p95 사용)"""
    try:
        if vit_processor_im is None or vit_model_im is None:
            from transformers import AutoImageProcessor, ViTForImageClassification
            import transformers
            import warnings
            
            transformers.logging.set_verbosity_error()
            warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
            
            MODEL_ID = "jaranohaal/vit-base-violence-detection"
            vit_processor_im = AutoImageProcessor.from_pretrained(MODEL_ID)
            vit_model_im = ViTForImageClassification.from_pretrained(MODEL_ID).to(DEVICE).eval()
        
        v_idx = 1  # Violent 클래스 인덱스
        
        violence_probs = []
        batch_size = 16
        
        for i in range(0, len(frames_pil), batch_size):
            chunk = frames_pil[i:i + batch_size]
            
            inputs = vit_processor_im(images=chunk, return_tensors="pt").to(DEVICE)
            
            with torch.no_grad():
                if DEVICE == 'cuda':
                    with torch.amp.autocast('cuda'):
                        outputs = vit_model_im(**inputs)
                else:
                    outputs = vit_model_im(**inputs)
                
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
            
            for prob_vec in probs:
                prob_vec = prob_vec.cpu().numpy()
                v_prob = float(prob_vec[v_idx])
                v_prob = float(np.clip(v_prob, 0.0, 1.0))
                violence_probs.append(v_prob)
        
        if not violence_probs:
            return 0.0
        
        return float(np.percentile(violence_probs, 95))
    
    except Exception as e:
        if verbose:
            print(f"  [ViT violence 계산 오류] {e}")
        return 0.0


def compute_slowfast_violence_score_im(slowfast_model, frame_tensors: List[torch.Tensor], verbose: bool = False) -> float:
    """SlowFast R101로 violence score 계산 (violence 키워드 매칭)"""
    try:
        from pytorchvideo.models.hub import slowfast_r101
        import torchvision.transforms as T
        
        if slowfast_model is None:
            slowfast_model = slowfast_r101(pretrained=True).to(DEVICE).eval()
        
        transform = T.Resize((224, 224))
        frames = torch.stack([transform(fr) for fr in frame_tensors])
        
        frames = frames.permute(1, 0, 2, 3)
        fast_pathway = frames
        
        T_len = frames.shape[1]
        alpha = 4
        num_slow = max(T_len // alpha, 1)
        idxs = torch.linspace(0, T_len - 1, num_slow).long()
        slow_pathway = frames[:, idxs, :, :]
        
        slow_pathway = slow_pathway.unsqueeze(0).to(DEVICE)
        fast_pathway = fast_pathway.unsqueeze(0).to(DEVICE)
        
        inp = [slow_pathway, fast_pathway]
        
        with torch.no_grad():
            if DEVICE == 'cuda':
                with torch.amp.autocast('cuda'):
                    out = slowfast_model(inp)  # logits [1, num_classes]
            else:
                out = slowfast_model(inp)
            
            prob = torch.softmax(out, dim=1)[0]
            
            top5 = torch.topk(prob, 5)
            top_idx = top5.indices.cpu().tolist()
            top_prob = top5.values.cpu().tolist()
            
            # Kinetics-400 라벨 로드
            VIOLENCE_KEYWORDS = [
                "fight", "fighting", "punch", "hit", "kick", "attack",
                "shoot", "shooting", "gun", "weapon", "stab", "strangle",
                "choke", "beat", "assault", "violence"
            ]
            
            labels = []
            num_classes = 400
            from pathlib import Path
            import os
            
            current_file = Path(__file__).resolve()
            demo_root = current_file.parent.parent
            label_paths = [
                demo_root / "VIDEO_IM" / "scripts" / "kinetics_400_labels.txt",
                current_file.parent / "kinetics_400_labels.txt",
                Path(r"C:\Users\psw20\OneDrive\바탕 화면\PSW\한국항공대학교_3-2\무하유\DEMO\VIDEO_IM\scripts\kinetics_400_labels.txt"),
            ]
            
            label_path = None
            for path in label_paths:
                if path.exists():
                    label_path = path
                    if verbose:
                        print(f"  [라벨 파일 찾음] {label_path}")
                    break
            
            if label_path is None:
                if verbose:
                    error_msg = f"[경고] kinetics_400_labels.txt 파일을 찾을 수 없습니다!\n시도한 경로:"
                    for p in label_paths:
                        error_msg += f"\n  - {p}"
                    print(error_msg)
                return 0.0
            
            if label_path and label_path.exists():
                try:
                    with open(label_path, "r", encoding="utf-8") as f:
                        for line in f:
                            name = line.strip()
                            if name:
                                labels.append(name)
                except Exception as e:
                    if verbose:
                        print(f"  [라벨 파일 로드 오류] {e}")
                    labels = []
            
            if len(labels) != num_classes:
                if verbose:
                    print(f"  [라벨 개수 불일치] {len(labels)}개 → 'class_i' 형식으로 대체")
                labels = [f"class_{i}" for i in range(num_classes)]
            
            # violence 키워드 매칭하여 최대 prob 반환
            violence_hint = 0.0
            for idx, p in zip(top_idx, top_prob):
                label = labels[idx].lower() if idx < len(labels) else f"class_{idx}"
                p_float = float(p)
                
                if any(kw in label for kw in VIOLENCE_KEYWORDS):
                    violence_hint = max(violence_hint, p_float)
        
        return float(np.clip(violence_hint, 0.0, 1.0))
    
    except Exception as e:
        if verbose:
            print(f"  [SlowFast violence 계산 오류] {e}")
        return 0.0


def extract_frames_safe(video_path: str) -> Tuple[List[Image.Image], List[torch.Tensor]]:
    """비디오에서 32프레임 균등 샘플링"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return [], []
    
    indices = np.linspace(0, total_frames - 1, FRAME_SAMPLE, dtype=int)
    
    frames_pil = []
    frame_tensors = []
    
    for idx in indices:
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret and frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frames_pil.append(frame_pil)
                
                frame_tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1) / 255.0
                frame_tensor = T.Resize((256, 256))(frame_tensor)
                frame_tensors.append(frame_tensor)
        except Exception as e:
            continue
    
    cap.release()
    
    while len(frame_tensors) < FRAME_SAMPLE:
        frame_tensors.extend(frame_tensors[:min(len(frame_tensors), FRAME_SAMPLE - len(frame_tensors))])
        frames_pil.extend(frames_pil[:min(len(frames_pil), FRAME_SAMPLE - len(frames_pil))])
    
    frame_tensors = frame_tensors[:FRAME_SAMPLE]
    frames_pil = frames_pil[:FRAME_SAMPLE]
    
    return frames_pil, frame_tensors


def extract_slowfast_features(slowfast_model, frame_tensors: List[torch.Tensor]) -> torch.Tensor:
    """
    SlowFast 모델로 비디오 행동 특징 추출
    
    Args:
        slowfast_model: SlowFast 모델
        frame_tensors: 프레임 텐서 리스트 (32개)
        
    Returns:
        features: SlowFast 특징 벡터 (400차원)
    """
    try:
        mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1)
        std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1)
        frame_tensors_normalized = [(f - mean) / std for f in frame_tensors]
        
        fast_pathway = torch.stack(frame_tensors_normalized).unsqueeze(0).to(DEVICE, non_blocking=True)
        fast_pathway = fast_pathway.permute(0, 2, 1, 3, 4)
        
        # Slow pathway: 8개 프레임 샘플링 (인덱스 범위 체크)
        num_frames = len(frame_tensors_normalized)
        if num_frames > 0:
            slow_indices = torch.linspace(0, num_frames - 1, min(8, num_frames)).long()
            slow_tensors = [frame_tensors_normalized[i] for i in slow_indices if i < num_frames]
            if len(slow_tensors) == 0:
                slow_tensors = frame_tensors_normalized[:1]  # 최소 1개는 필요
            slow_pathway = torch.stack(slow_tensors).unsqueeze(0).to(DEVICE, non_blocking=True)
            slow_pathway = slow_pathway.permute(0, 2, 1, 3, 4)
        else:
            # 프레임이 없으면 fast_pathway와 동일하게
            slow_pathway = fast_pathway
        
        with torch.no_grad():
            if DEVICE == 'cuda':
                with torch.amp.autocast('cuda'):
                    features = slowfast_model([slow_pathway, fast_pathway])
            else:
                features = slowfast_model([slow_pathway, fast_pathway])
            features = features.squeeze()
        
        return features
    
    except Exception as e:
        # SlowFast 오류는 조용히 처리 (평가 시 출력 방지)
        return torch.zeros(SLOWFAST_DIM, device=DEVICE)


def predict_video(video_path: str, video_model, yolo_model, slowfast_model,
                  clip_model, clip_preprocess, threshold: float = 0.63, verbose: bool = True,
                  clip_processor_im=None, clip_model_im=None,
                  vit_processor_im=None, vit_model_im=None) -> Dict:
    """
    비디오에 대한 유해 콘텐츠 탐지 (Fusion 방식)
    
    Fusion 구조:
    - CLIP (harmful/benign 프롬프트 비교): 가중치 0.8
    - ViT (jaranohaal/vit-base-violence-detection): 가중치 0.1
    - SlowFast R101 (Kinetics-400 행동 인식): 가중치 0.1
    - 임계값: 0.63
    """
    try:
        if video_path is None:
            return {
                "is_harmful": False,
                "confidence": 0.0,
                "detected_objects": [],
                "detected_behaviors": [],
                "error": "비디오 파일을 불러올 수 없습니다."
            }
        
        frames_pil, frame_tensors = extract_frames_safe(video_path)
        
        if len(frames_pil) == 0:
            return {
                "is_harmful": False,
                "confidence": 0.0,
                "detected_objects": [],
                "detected_behaviors": [],
                "error": "비디오에서 프레임을 추출할 수 없습니다."
            }
        
        frame_np_list = [np.array(frame_pil) for frame_pil in frames_pil]
        yolo_results = yolo_model(frame_np_list, verbose=False, device=DEVICE, imgsz=640, conf=0.25)
        
        all_object_counts = {}
        if isinstance(yolo_results, list):
            for result in yolo_results:
                _, obj_counts = extract_yolo_features([result])
                for obj, count in obj_counts.items():
                    all_object_counts[obj] = all_object_counts.get(obj, 0) + count
        else:
            _, obj_counts = extract_yolo_features([yolo_results])
            for obj, count in obj_counts.items():
                all_object_counts[obj] = all_object_counts.get(obj, 0) + count
        
        detected_objects = [obj for obj in HARMFUL_OBJECTS if all_object_counts.get(obj, 0) > 0]
        
        clip_score = compute_clip_violence_score_im(
            frames_pil, clip_model, clip_preprocess, 
            clip_processor_im=clip_processor_im, clip_model_im=clip_model_im, 
            verbose=verbose
        )
        
        vit_score = compute_vit_violence_score_im(
            frames_pil, 
            vit_processor_im=vit_processor_im, vit_model_im=vit_model_im, 
            verbose=verbose
        )
        
        slowfast_score = compute_slowfast_violence_score_im(slowfast_model, frame_tensors, verbose=verbose)
        
        if verbose:
            print(f"  [Fusion 점수] CLIP: {clip_score:.4f}, ViT: {vit_score:.4f}, SlowFast: {slowfast_score:.4f}")
        
        fusion_weights = {
            "clip": 0.8,
            "vit": 0.1,
            "slowfast": 0.1,
        }
        
        confidence = (
            fusion_weights["clip"] * clip_score +
            fusion_weights["vit"] * vit_score +
            fusion_weights["slowfast"] * slowfast_score
        )
        
        if verbose:
            print(f"  [Fusion 결과] confidence: {confidence:.4f}, threshold: {threshold:.4f}")
        
        is_harmful = confidence >= threshold
        
        inferred_categories = infer_behavior_from_objects(all_object_counts)
        detected_behaviors = inferred_categories.copy()
        
        if confidence >= 0.5:
            if 'violence' not in detected_behaviors:
                detected_behaviors.append('violence')
        
        return {
            "is_harmful": is_harmful,
            "confidence": confidence,
            "detected_objects": detected_objects,
            "detected_behaviors": detected_behaviors
        }
    
    except Exception as e:
        print(f"비디오 추론 오류: {e}")
        import traceback
        traceback.print_exc()
        return {
            "is_harmful": False,
            "confidence": 0.0,
            "detected_objects": [],
            "detected_behaviors": [],
            "error": str(e)
        }
