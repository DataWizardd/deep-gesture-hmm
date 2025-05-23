let startRecordingBtn = document.getElementById('startRecording');
let stopRecordingBtn = document.getElementById('stopRecording');
let resultDiv = document.getElementById('result');
let mediaRecorder = null;
let audioChunks = [];

// 음성 인식(To-Be)
async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        
        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };
        
        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            const formData = new FormData();
            formData.append('audio', audioBlob);
            
            try {
                const response = await fetch('/api/speech-to-text', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                resultDiv.textContent = data.text;
            } catch (err) {
                console.error('음성 인식 오류:', err);
                resultDiv.textContent = '음성 인식에 실패했습니다.';
            }
        };
        
        mediaRecorder.start();
        startRecordingBtn.disabled = true;
        stopRecordingBtn.disabled = false;
        resultDiv.textContent = '녹음 중...';
    } catch (err) {
        console.error('마이크 접근 오류:', err);
        resultDiv.textContent = '마이크 접근에 실패했습니다.';
    }
}

// 음성 인식 중지
function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        startRecordingBtn.disabled = false;
        stopRecordingBtn.disabled = true;
    }
}

// 이벤트 리스너
startRecordingBtn.addEventListener('click', startRecording);
stopRecordingBtn.addEventListener('click', stopRecording);

// 초기 상태 설정
stopRecordingBtn.disabled = true; 