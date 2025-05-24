# 🤖 Sign_Translator

## 📌 개요
**딥러닝의 시계열 데이터 모델링을 보강한 수화 인식 연구**

---

## 📸 데이터 수집

### 사용 기술
- **MediaPipe Hands**  
  Google의 MediaPipe 모델을 이용하여 손가락 21개 랜드마크(x, y, z)를 실시간으로 추출

  <img src="https://github.com/user-attachments/assets/6110e708-73ef-4c0f-a703-d71e272d67ea" width="250"/>

---

## 🧠 AI 모델 구축 과정

### 1. LSTM AutoEncoder 기반 특징 추출
- **입력**: 손 랜드마크 시퀀스 (프레임 단위)
- **출력**: latent feature 추출 (Encoder) 

### 2. 제스처 단위 HMM 학습
- 특징 시퀀스를 HMM에 Gaussian 분포 기반으로 학습
- 라이브러리: `hmmlearn`  

### 3. 제스처별 HMM 모델 통합 (Ergodic HMM)
- 모든 상태 간 전이 가능하도록 구성하여 다중 제스처 인식
- 임계치 모델로 활용

### 4. 슬라이딩 윈도우 기반 인식
- 입력 시퀀스를 일정 구간 단위로 분할
- 각 구간에 대해 제스처별 Likelihood 산출 → **임계치 모델**보다 높을 경우 해당 구간 제스처로 결정  

---

### 🧪 Result
- 9개의 제스처에 대해 무작위로 3개 이상의 동작을 진행하여 인식률 Test 진행  

  <img src="https://github.com/user-attachments/assets/38b343d0-d5ef-4381-b9b1-9ccd4bb5f6e6" width="500"/>  
  <img src="https://github.com/user-attachments/assets/e45555cb-a722-48aa-ac0c-d66b09b0b485" width="500"/>

---

## 🌐 Web
**수어 번역 AI 모델을 탑재한 웹사이트 구현**

### 🔐 [Login]  
<img src="https://github.com/user-attachments/assets/e29fd2f0-241f-4f43-ab6f-b67d7e902d0c" width="400"/>

---

### 📝 [Translate]  
<img src="https://github.com/user-attachments/assets/38aca8b6-ec9d-4e4b-bd86-50d62028e3fb" width="400"/>

---

### 🎙️ [STT]  
<img src="https://github.com/user-attachments/assets/6aca2763-fc2e-46b6-beff-82aca84d850b" width="400"/>

---

### 🙋 [My Page]  
<img src="https://github.com/user-attachments/assets/2c83bf7a-db20-4344-9351-a2b2b9fcbea5" width="400"/>

---
