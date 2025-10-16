# 1) 가상환경 생성/활성화
conda create -n gesture-rt python=3.10 -y
conda activate gesture-rt

# 2) 의존성 설치
pip install -r requirements.txt

# 3) 실행 : 프로젝트 루트에서
python -m main_5