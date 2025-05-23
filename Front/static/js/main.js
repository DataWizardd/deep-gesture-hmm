let video = document.getElementById('video');
let canvas = document.getElementById('canvas');
let startBtn = document.getElementById('startBtn');
let stopBtn = document.getElementById('stopBtn');
let resultDiv = document.getElementById('result');
let stream = null;
let isRecording = false;

// 카메라 스트림 시작
async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
    } catch (err) {
        console.error('카메라 접근 오류:', err);
        resultDiv.textContent = '카메라 접근에 실패했습니다.';
    }
}

// 카메라 스트림 중지
function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
    }
}

// 수어 인식 및 번역
async function recognizeSign() {
    if (!stream) return;
    
    
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    
    resultDiv.textContent = '수어 인식 중...';
    
    // 다음 프레임을 위해 재귀 호출
    requestAnimationFrame(recognizeSign);
}

// 이벤트 리스너
startBtn.addEventListener('click', () => {
    startCamera();
    startBtn.disabled = true;
    stopBtn.disabled = false;
    recognizeSign();
});

stopBtn.addEventListener('click', () => {
    stopCamera();
    startBtn.disabled = false;
    stopBtn.disabled = true;
    resultDiv.textContent = '';
});

// 초기 상태 설정
stopBtn.disabled = true;

document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('video');
    const cameraContainer = document.querySelector('.camera-container');
    const cameraStatus = document.querySelector('.camera-status');
    const translationResult = document.querySelector('.translation-result .alert');

    cameraContainer.addEventListener('click', async () => {
        if (!isRecording) {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ video: true });
                video.srcObject = stream;
                isRecording = true;
                cameraStatus.innerHTML = '<i class="fas fa-stop"></i><span>카메라를 정지하려면 화면을 클릭하세요</span>';
                translationResult.textContent = '수어를 인식 중입니다...';
            } catch (err) {
                console.error('카메라 접근 오류:', err);
                translationResult.textContent = '카메라 접근이 거부되었습니다. 카메라 권한을 확인해주세요.';
            }
        } else {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                video.srcObject = null;
                stream = null;
            }
            isRecording = false;
            cameraStatus.innerHTML = '<i class="fas fa-video"></i><span>카메라를 시작하려면 화면을 클릭하세요</span>';
            translationResult.textContent = '카메라를 시작하여 수어를 번역하세요.';
        }
    });
}); 