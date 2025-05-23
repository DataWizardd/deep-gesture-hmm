let addPhraseForm = document.getElementById('addPhraseForm');
let phraseInput = document.getElementById('phraseInput');
let phraseList = document.getElementById('phraseList');

// 문구 추가
async function addPhrase(event) {
    event.preventDefault();
    
    const phrase = phraseInput.value.trim();
    if (!phrase) return;
    
    try {
        const response = await fetch('/api/phrases', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ phrase })
        });
        
        if (response.ok) {
            phraseInput.value = '';
            loadPhrases();
        }
    } catch (err) {
        console.error('문구 추가 오류:', err);
    }
}

// 문구 삭제
async function deletePhrase(id) {
    try {
        const response = await fetch(`/api/phrases/${id}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            loadPhrases();
        }
    } catch (err) {
        console.error('문구 삭제 오류:', err);
    }
}

// 문구 목록 로드
async function loadPhrases() {
    try {
        const response = await fetch('/api/phrases');
        const phrases = await response.json();
        
        phraseList.innerHTML = '';
        phrases.forEach(phrase => {
            const li = document.createElement('li');
            li.className = 'list-group-item';
            li.innerHTML = `
                <span>${phrase.text}</span>
                <span class="delete-btn" onclick="deletePhrase('${phrase.id}')">삭제</span>
            `;
            phraseList.appendChild(li);
        });
    } catch (err) {
        console.error('문구 목록 로드 오류:', err);
    }
}

// 이벤트 리스너
addPhraseForm.addEventListener('submit', addPhrase);

// 초기 문구 목록 로드
loadPhrases(); 