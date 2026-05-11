# 유해 콘텐츠 탐지 시스템 - Demo 버전

> 팀 프로젝트에서 개발된 모델을 활용한 웹 데모 인터페이스 구현  
> 평가 데이터셋(1,200개)에 대한 추가 분석 결과를 포함하여 웹 데모에 통합

딥러닝 기반 이미지·비디오 유해 콘텐츠 자동 탐지 시스템의 배포용 데모입니다.

## 프로젝트 개요

YOLOv8, CLIP, SlowFast 등 최신 딥러닝 모델을 활용하여 이미지와 비디오에서 유해 콘텐츠를 자동으로 탐지하는 시스템입니다.

### 웹 데모 화면

<img src="harmful_content_demo/examples/muhayu/project_demo.png" alt="Hazard Killer 데모" width="600">

### 주요 기능

- **이미지 분석**: 유해 객체 및 행동 탐지
- **비디오 분석**: 시간적 패턴을 고려한 유해 콘텐츠 탐지
- **카테고리 기반**: 9개 카테고리, 8개 행동으로 세분화
- **실시간 분석**: 웹 인터페이스에서 즉시 결과 확인

### 탐지 항목

**유해 객체 (20종, 카테고리 기반)**
- **무기류 (12종)**: knife, dagger, machete, sword, axe, gun, pistol, rifle, shotgun, machine_gun, grenade, bomb
- **음주 (2종)**: wine glass, beer
- **흡연 (2종)**: cigarette, lighter
- **약물 (1종)**: syringe
- **혈액/상처 (3종)**: blood, injury, wound

**유해 행동 (8종)**
- violence (폭력 행위)
- alcohol (음주 행위)
- smoking (흡연 행위)
- drugs (약물 사용)
- blood (혈액/상처)
- threat (위협적 행동)
- sexual (성적 콘텐츠)
- dangerous (위험행동)

## 빠른 시작

### 1. 데모 폴더로 이동

```bash
cd harmful_content_demo
```

### 2. 가상 환경 생성 및 활성화

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. PyTorch 설치 (GPU 사용 시)

```bash
# CUDA 12.1용 PyTorch 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. 패키지 설치

```bash
pip install -r requirements.txt
```

### 5. 모델 가중치 확인

모델 가중치 파일들은 `harmful_content_demo/weights` 폴더에 있어야 합니다:
- `image_model_best.pth` - 박상원 이미지 분류 모델 가중치 (final_model11 기반, 540차원)
- `video_model_best.pth` - 더미 파일 (임영재 비디오 모델은 fusion 방식으로 별도 모델 파일 불필요)
- `yolov8n.pt` - YOLOv8 모델 가중치

### 6. 웹 앱 실행

```bash
# harmful_content_demo 폴더에서 실행
python app.py
```

**접속 방법:**
- 로컬 접속: `http://localhost:7860`
- 외부 접속: `config.py`에서 `GRADIO_SHARE = True`로 설정 시 공개 링크 자동 생성

### 7. 모델 평가 (선택사항)

```bash
# harmful_content_demo 폴더에서 실행
python evaluate_category.py
```

**상세한 설치 및 사용 방법은 `harmful_content_demo/README.md`를 참고하세요.**


## 아키텍처

### 이미지 분류 모델 

1. **YOLOv8**: 객체 탐지 (20차원 특징) - 무기, 음주, 흡연, 약물, 혈액/상처 객체 탐지
2. **CLIP**: 이미지 맥락 이해 (512차원)
3. **행동 인식**: CLIP 기반 Zero-shot 행동 점수 (8차원) - 카테고리별 여러 프롬프트 지원
4. **특징 결합**: YOLO(20) + CLIP(512) + 행동(8) = 540차원
5. **차원 축소**: 540차원 → 256차원
6. **MLP 분류기**: 최종 유해 확률 출력
7. **임계값**: 체크포인트에서 자동 로드 (기본값: 0.4)

### 비디오 분류 모델

1. **32프레임 균등 샘플링**: 비디오에서 32개 프레임 균등 추출 (np.linspace 사용)
2. **CLIP (가중치 0.8)**: 
   - 모델: openai/clip-vit-base-patch32 (transformers)
   - harmful/benign 프롬프트 비교로 violence score 계산
   - temperature: 2.0
   - 점수: p95 (95th percentile) 사용
3. **ViT (가중치 0.1)**: 
   - 모델: jaranohaal/vit-base-violence-detection
   - Violent 클래스(class 1) 확률 사용
   - 점수: p95 (95th percentile) 사용
4. **SlowFast R101 (가중치 0.1)**: 
   - 모델: slowfast_r101 (Kinetics-400)
   - violence 관련 키워드로 행동 탐지
   - 점수: max violence_hint 사용
5. **Fusion**: 가중치 결합으로 최종 violence_prob 계산
   - `confidence = 0.8 * clip_score + 0.1 * vit_score + 0.1 * slowfast_score`
6. **임계값**: 0.63 기준으로 유해/안전 분류

## 프로젝트 구조

```
DEMO/
├── harmful_content_demo/      # 메인 데모 애플리케이션
│   ├── app.py                # Gradio 웹 인터페이스 메인 파일
│   ├── evaluate_category.py  # 모델 평가 스크립트 (카테고리별 상세 분석)
│   ├── models.py             # 모델 클래스 정의
│   ├── inference.py          # 추론 함수 (이미지/비디오 유해 콘텐츠 탐지)
│   ├── config.py             # 설정 파일 (경로, 하이퍼파라미터, 디바이스)
│   ├── requirements.txt      # Python 패키지 의존성 목록
│   ├── examples/             # 예제 이미지/비디오 파일
│   ├── weights/              # 모델 가중치 파일
│   │   ├── image_model_best.pth  # 이미지 분류 모델 (540차원)
│   │   ├── video_model_best.pth  # 더미 파일 (Fusion 방식 사용으로 불필요)
│   │   └── yolov8n.pt            # YOLOv8 객체 탐지 모델
│   └── README.md             # 상세 문서
├── IMAGE_PARK/               # 박상원 이미지 모델 (참고용)
│   └── ...                   # 이미지 모델 관련 파일들
├── VIDEO_IM/                 # 임영재 비디오 모델 (참고용)
│   ├── scripts/              # 비디오 모델 스크립트
│   └── ...                   # 비디오 모델 관련 파일들
├── paper/                    # SW중심대학 산학R&D 프로젝트 결과보고서
│   ├── README.md             # 결과보고서 개요 및 이미지
│   └── img/                  # 결과보고서 페이지 이미지 (PDF 변환)
└── README.md                 # 이 파일 (전체 프로젝트 개요)
```

**주요 사용 폴더**: `harmful_content_demo/` - 실제 데모 실행은 이 폴더에서 진행됩니다.

## 주요 특징

- **멀티모달 접근**: 이미지와 비디오 각각에 최적화된 모델 사용
- **카테고리 기반 구조**: 9개 유해 카테고리, 8개 행동 카테고리로 세분화된 탐지
- **Zero-shot Learning**: CLIP을 활용한 프롬프트 기반 행동 인식
- **Fusion 방식**: 비디오 모델에서 CLIP, ViT, SlowFast를 가중치 결합하여 사용
- **상세한 평가**: 카테고리별 성능 분석 및 모델 비교 기능 제공

## 모델 및 웹 데모

### 모델 개발 과정

모델 개발 및 실험 과정은 팀 프로젝트로 진행되었으며, 별도의 레포지토리에서 관리됩니다.

**모델 발전 과정 및 실험 기록**: [https://github.com/psw204/harmful-detect-muhayu](https://github.com/psw204/harmful-detect-muhayu)

해당 레포지토리에는 다음 내용이 포함되어 있습니다:
- 각 팀원의 모델 실험 과정 (안지산, 박상원, 임영재)
- 모델 구조 발전 과정 (final_model1 ~ final_model11)
- 데이터 수집 및 라벨링 도구
- 학습 스크립트 및 평가 코드
- 최종 모델 선정 과정 및 성능 비교

### 웹 데모 인터페이스

본 레포지토리는 팀 프로젝트에서 개발된 최종 모델을 활용하여 웹 데모 인터페이스를 구현한 것입니다.

**사용 모델:**
- **이미지 모델**: final_model11 (YOLO + CLIP + 행동인식, 540차원)
- **비디오 모델**:  Fusion 방식 (CLIP + ViT + SlowFast)

**웹 데모 특징:**
- 반응형 디자인 (PC/모바일 최적화)
- 다크모드 지원
- 대기열 시스템 (동시 접속 안정성)
- 평가 데이터셋(1,200개) 분석 결과 통합

## 폴더 설명

- **`harmful_content_demo/`**: 메인 데모 애플리케이션 폴더
  - 웹 인터페이스 실행 및 모델 평가는 이 폴더에서 진행
  - 상세한 사용 방법은 `harmful_content_demo/README.md` 참고

- **`IMAGE_PARK/`**: 박상원 이미지 모델 폴더 (참고용)
  - 이미지 모델 개발 과정 및 실험 코드

- **`VIDEO_IM/`**: 임영재 비디오 모델 폴더 (참고용)
  - 비디오 모델 개발 과정 및 Fusion 방식 구현 코드

## 참고사항

- 이 시스템은 학습 데이터 기반으로 작동하며, 100% 정확도를 보장하지 않습니다.
- 실제 판단은 전문가의 검토를 권장합니다.
- **이미지 모델**: final_model11 기반 모델 사용 (540차원 입력)
- **비디오 모델**: Fusion 방식 사용 (CLIP 0.8 + ViT 0.1 + SlowFast 0.1), 32프레임 균등 샘플링
- 모델 가중치 파일은 `harmful_content_demo/weights` 폴더에 있어야 합니다.
- 상세한 설치 및 사용 방법은 `harmful_content_demo/README.md`를 참고하세요.

## 관련 문서

SW중심대학 산학R&D프로젝트 결과보고서를 확인하려면 [`paper`](paper/) 폴더를 참고하세요.
