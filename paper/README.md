# Multimodal Harmful Content Detection Pipeline  
## SW중심대학 산학 R&D 프로젝트 결과보고서

본 저장소는 **2025년도 SW중심대학 산학 R&D 프로젝트**의 결과물로,
이미지 및 비디오 기반 **유해 콘텐츠 자동 탐지 시스템(Hazard Killer)**의
설계, 구현, 평가 내용을 정리한 결과보고서를 포함합니다.

---

## Project Overview

온라인 플랫폼에서 증가하는 폭력, 무기, 혈액/상해, 약물, 성적 콘텐츠,
위협 및 위험행동과 같은 유해 콘텐츠를 자동으로 탐지하기 위해
**모달별 특성을 반영한 멀티모달 데이터 파이프라인**을 설계·구현하였습니다.

본 프로젝트는 **산학 협력(무하유)** 캡스톤 과제로 수행되었으며,
실서비스 적용을 고려한 공정한 모델 비교 및 독립 평가 데이터를 기반으로
시스템 성능을 검증하였습니다.

---

## System Architecture

- **Image Pipeline**
  - YOLOv8 기반 유해 객체 탐지
  - CLIP 기반 장면 문맥 추정
  - 규칙 기반 행동 추론 및 특징 결합

- **Video Pipeline**
  - 32-frame 균등 샘플링 기반 입력 표준화
  - CLIP / ViT / SlowFast 점수 late fusion
  - 실서비스 안정성을 고려한 임계값 설계

---

## Evaluation Summary

- Training: Public datasets (HOD, COCO Safe, RWF-2000, RLVS)
- Evaluation: Independently collected & labeled dataset (1,221 samples)
- Performance:
  - Image F1-score: **0.6912**
  - Video F1-score: **0.5813**

---

## Report Pages

아래는 최종 결과보고서의 전체 페이지 이미지입니다.

<img src="img/SW중심대학 산학R&D프로젝트 결과보고서 (박상원)_page-0001.jpg" width="600">

<img src="img/SW중심대학 산학R&D프로젝트 결과보고서 (박상원)_page-0002.jpg" width="600">

<img src="img/SW중심대학 산학R&D프로젝트 결과보고서 (박상원)_page-0003.jpg" width="600">

<img src="img/SW중심대학 산학R&D프로젝트 결과보고서 (박상원)_page-0004.jpg" width="600">

<img src="img/SW중심대학 산학R&D프로젝트 결과보고서 (박상원)_page-0005.jpg" width="600">

<img src="img/SW중심대학 산학R&D프로젝트 결과보고서 (박상원)_page-0006.jpg" width="600">

<img src="img/SW중심대학 산학R&D프로젝트 결과보고서 (박상원)_page-0007.jpg" width="600">

<img src="img/SW중심대학 산학R&D프로젝트 결과보고서 (박상원)_page-0008.jpg" width="600">

<img src="img/SW중심대학 산학R&D프로젝트 결과보고서 (박상원)_page-0009.jpg" width="600">

<img src="img/SW중심대학 산학R&D프로젝트 결과보고서 (박상원)_page-0010.jpg" width="600">

<img src="img/SW중심대학 산학R&D프로젝트 결과보고서 (박상원)_page-0011.jpg" width="600">

---

## Contributions

### **박상원**

* 프로젝트 전반의 **연구 아이디어 제안 및 방향 설정**을 주도하였으며, 해당 아이디어를 기반으로 본 프로젝트를 최종 진행함
* 유해 콘텐츠 **데이터 카테고리 정의 및 일반 라벨링 가이드라인 설계**, 관련 **Python 기반 데이터 처리 및 검증 코드 구현**
* **이미지 및 비디오 모델 구현과 비교 실험을 수행**하였으며, 이미지 모델을 최종 시스템 구성 요소로 선정
* **GitHub 저장소 전반 관리**, 파일 및 디렉토리 구조 설계, 모든 **README 문서 작성**
* 프로젝트 **데모 시스템 전체 구현** 및 최종 발표에서 단일 데모 시스템을 사용하여 시연 수행
* 기존 논문 초안을 **전면 수정·재구성하여 최종 원고 작성**
* 프로젝트 **홍보 배너 콘텐츠 문안 최종 작성**

### **안지산**

* **비디오 기반 유해 콘텐츠 탐지 모델 구현**
* 프로젝트 홍보용 **배너 이미지 제작**, 특히 파이프라인 구조 및 연구 방법 관련 시각 자료 설계
* 논문 및 홍보 배너 **콘텐츠 초안 작성**

### **임영재**

* **이미지 및 비디오 기반 유해 콘텐츠 탐지 모델 구현**
* 제안한 **비디오 모델이 최종 시스템 구성 요소로 채택됨**

---

## Related Links

- Experiment Code: https://github.com/psw204/harmful-detect-muhayu  
- Demo Repository: https://github.com/psw204/harmful-detect-muhayu-demo  

---

본 연구는 2025년도 과학기술정보통신부 및 정보통신기획평가원의  
**SW중심대학사업** 지원을 받아 수행되었습니다.
