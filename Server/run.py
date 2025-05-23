import uvicorn
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

if __name__ == "__main__":
    # 개발 환경 설정
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5000"))
    reload = os.getenv("ENV", "development") == "development"

    # 서버 실행
    uvicorn.run(
        "app:app", host=host, port=port, reload=reload, workers=1 if reload else 4
    )
