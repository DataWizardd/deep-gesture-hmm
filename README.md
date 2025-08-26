## 📌 개요
**딥러닝과 은닉 마르코프 모델을 연계한 연속적인 손 제스처 적출에 관한 연구**

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


  <img width="395" height="222" alt="image" src="https://github.com/user-attachments/assets/c5a58d94-2316-4273-809c-a8a00c0a6712" />

  <img width="547" height="185" alt="image" src="https://github.com/user-attachments/assets/6e2bc746-fb7e-485d-b0d8-db07cb0d20dd" />


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
