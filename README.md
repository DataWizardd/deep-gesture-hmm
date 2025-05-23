# 🤖 Sign_Translator

## 📌 개요
딥러닝의 시계열 데이터 모델링을 보강한 수화 인식기 개발

---

## 📸 데이터 수집

### 사용 기술
- **MediaPipe Hands**  
  ![MediaPipe](./img/mediapipe.png)
  
  Google의 MediaPipe 모델을 이용하여 손가락 21개 랜드마크(x, y, z)를 실시간으로 추출합니다.
  ![image](https://github.com/user-attachments/assets/6110e708-73ef-4c0f-a703-d71e272d67ea)

---

## 🧠 AI 모델 구축 과정

### 1. LSTM AutoEncoder 기반 특징 추출
- **입력**: 손 랜드마크 시퀀스 (프레임 단위)
- **출력**: Bottleneck(잠재 특징)  
  ![AutoEncoder](./img/autoencoder.png)

### 2. 제스처 단위 HMM 학습
- 특징 시퀀스를 HMM에 Gaussian 분포 기반으로 학습
- 라이브러리: `hmmlearn`  
  ![hmmlearn](./img/hmmlearn.png)

### 3. 제스처별 HMM 모델 통합 (Ergodic HMM)
- 모든 상태 간 전이 가능하도록 구성하여 다중 제스처 인식  
  ![Ergodic Model](./img/hmm_transition.png)

### 4. 슬라이딩 윈도우 기반 인식
- 입력 시퀀스를 일정 구간 단위로 분할
- 각 구간에 대해 모든 HMM 점수 산출 → **최대 점수 모델**을 예측 결과로 결정  
  ![Sliding Window](./img/sliding_window.png)

### Result
- 9개의 제스처에 대해 무작위로 3개 이상의 동작을 진행하여 인식률 Test 진행
 ![image](https://github.com/user-attachments/assets/38b343d0-d5ef-4381-b9b1-9ccd4bb5f6e6)
<img width="546" alt="image" src="https://github.com/user-attachments/assets/e45555cb-a722-48aa-ac0c-d66b09b0b485" />

---
### Web
- 수어 번역 AI 모델을 탑재한 웹사이트 구현
  
[Login]
---
![image](https://github.com/user-attachments/assets/e29fd2f0-241f-4f43-ab6f-b67d7e902d0c)
----
[Translate]
---
![image](https://github.com/user-attachments/assets/38aca8b6-ec9d-4e4b-bd86-50d62028e3fb)
---
[STT]
---
![image](https://github.com/user-attachments/assets/6aca2763-fc2e-46b6-beff-82aca84d850b)
---
[My Page]
---
![image](https://github.com/user-attachments/assets/2c83bf7a-db20-4344-9351-a2b2b9fcbea5)
---
