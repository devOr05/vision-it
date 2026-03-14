const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const status = document.getElementById('status');
const countNum = document.getElementById('count-num');
const labelsContainer = document.createElement('div'); // Dummy container to avoid errors
labelsContainer.id = 'labels-container';
const toggleCamBtn = document.getElementById('toggle-camera');

// UI Elements
const openSettings = document.getElementById('open-settings');
const closeSettings = document.getElementById('close-settings');
const settingsDrawer = document.getElementById('settings-drawer');
const confSlider = document.getElementById('conf-slider');
const confVal = document.getElementById('conf-val');
const speechToggle = document.getElementById('speech-toggle');
const countModeToggle = document.getElementById('count-mode-toggle');
const captureBtn = document.getElementById('capture-btn');

// Training UI
const trainModeBtn = document.getElementById('train-mode-btn');
const saveModelBtn = document.getElementById('save-model-btn');
const resetTraining = document.getElementById('reset-training');
const aiModeText = document.getElementById('ai-mode-text');
const customPredBox = document.getElementById('custom-prediction');
const customLabel = document.getElementById('custom-label');

const sampleCounters = [
    document.getElementById('samples-0'),
    document.getElementById('samples-1'),
    document.getElementById('samples-2')
];
const galleries = [
    document.getElementById('gallery-0'),
    document.getElementById('gallery-1'),
    document.getElementById('gallery-2')
];

let detectionHistory = []; // For PDF report
let lastLogTime = 0;
let lastLoggedClass = -1;

let model;
let featureExtractor; // MobileNet
let classifier; // KNN Classifier
let currentFacingMode = 'environment';
let confidenceThreshold = 0.6;
let isSpeechEnabled = false;
let isTrainingMode = false;
let isCountingMode = false;
let lastSpoken = "";
let lastSpokenTime = 0;

// Demo Mode & Telegram Config
let isDemoMode = true;
let targetClass = 'cell phone';
let telegramToken = localStorage.getItem('tg_token') || '8535485891:AAEvAOiKwef-PlGffxwcJubUKYuB819sd90';
let telegramChatId = localStorage.getItem('tg_chatid') || '1577936762';
let isNotifyingTelegram = false;
let lastTelegramTargetTime = 0;
let telegramCooldown = 30000; // 30s between photos
let targetDetectionStartTime = 0;
let detectionRequiredTime = 800; // 0.8s of stable detection
let detectionCounter = 0; // Sequential ID for detections
let classSampleCounts = [0, 0, 0];
let classThumbnails = [[], [], []];

// Camera Handling
async function setupCamera() {
    status.innerText = 'Conectando...';
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
    }

    const constraints = {
        video: {
            facingMode: currentFacingMode,
            width: { ideal: 1280 },
            height: { ideal: 720 }
        },
        audio: false
    };

    try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = stream;

        return new Promise((resolve, reject) => {
            const onReady = () => {
                video.play().then(() => resolve(video)).catch(reject);
            };
            if (video.readyState >= 3) onReady();
            else video.onloadeddata = onReady;

            // Timeout safety for camera
            setTimeout(() => reject('Timeout de cámara'), 5000);
        });
    } catch (err) {
        console.error(err);
        if (err.name === 'NotAllowedError') status.innerText = 'Cámara bloqueada (permiso denegado)';
        else status.innerText = 'Error de Cámara';
        throw err;
    }
}

// Memory & Persistence
async function saveModel() {
    const dataset = classifier.getClassifierDataset();
    const saveBtn = document.getElementById('save-model-btn');

    if (Object.keys(dataset).length === 0) {
        status.innerText = '⚠️ Capture muestras antes de guardar';
        status.style.color = '#fbbf24';
        return;
    }

    try {
        saveBtn.innerText = 'Guardando...';
        saveBtn.disabled = true;

        const readableDataset = {};
        Object.keys(dataset).forEach((key) => {
            const data = dataset[key].dataSync();
            readableDataset[key] = Array.from(data);
        });

        const json = JSON.stringify({
            dataset: readableDataset,
            counts: classSampleCounts,
            thumbnails: classThumbnails.map(arr => arr.slice(-3))
        });

        localStorage.setItem('vision_it_model_v3', json);

        status.innerText = '✅ Aprendizaje guardado en memoria';
        status.style.color = '#2dd4bf';
        saveBtn.innerText = 'Guardado con éxito';
        saveBtn.style.background = '#059669';

        setTimeout(() => {
            saveBtn.innerText = 'Guardar Aprendizaje';
            saveBtn.disabled = false;
            saveBtn.style.background = '';
            status.style.color = '';
            settingsDrawer.classList.remove('active');
        }, 3000);

    } catch (err) {
        console.error('Error al guardar:', err);
        status.innerText = '❌ Error: Memoria llena o bloqueada';
        saveBtn.innerText = 'Reintentar Guardar';
        saveBtn.disabled = false;
    }
}

function loadModel() {
    const json = localStorage.getItem('vision_it_model_v3');
    if (!json) return;

    try {
        const data = JSON.parse(json);
        const dataset = data.dataset;
        classSampleCounts = data.counts || [0, 0, 0];
        classThumbnails = data.thumbnails || [[], [], []];

        const tensors = {};
        Object.keys(dataset).forEach((key) => {
            tensors[key] = tf.tensor2d(dataset[key], [dataset[key].length / 1024, 1024]);
        });

        classifier.setClassifierDataset(tensors);
        updateUIFeedback();
        status.innerText = '📦 Aprendizaje restaurado (IA Lista)';
    } catch (err) {
        console.error('Error loading model:', err);
        status.innerText = 'Error al cargar memoria previa';
    }
}

function updateUIFeedback() {
    classSampleCounts.forEach((count, i) => {
        if (sampleCounters[i]) sampleCounters[i].innerText = count;

        if (galleries[i]) {
            galleries[i].innerHTML = '';
            classThumbnails[i].forEach(src => {
                const img = document.createElement('img');
                img.src = src;
                img.className = 'sample-thumbnail';
                galleries[i].appendChild(img);
            });
        }
    });
}

window.resetClass = (id) => {
    if (confirm(`¿Reiniciar aprendizaje para esta clase?`)) {
        try {
            classifier.clearClass(id);
        } catch (e) { }
        classSampleCounts[id] = 0;
        classThumbnails[id] = [];
        updateUIFeedback();
        status.innerText = 'Clase reiniciada';
    }
};

// Model Loading
async function loadModels() {
    try {
        const [cocoRes, mobRes] = await Promise.all([
            cocoSsd.load(),
            mobilenet.load({ version: 2, alpha: 1.0 })
        ]);

        model = cocoRes;
        featureExtractor = mobRes;
        classifier = knnClassifier.create();

        loadModel();
    } catch (err) {
        console.error(err);
        throw err;
    }
}

function speak(text) {
    if (!isSpeechEnabled) return;
    const now = Date.now();
    if (now - lastSpokenTime < 4000 && text === lastSpoken) return;

    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = 'es-ES';
    u.rate = 1.1;
    window.speechSynthesis.speak(u);
    lastSpoken = text;
    lastSpokenTime = now;
}

function logDetection(label, classIndex) {
    const now = Date.now();
    if (now - lastLogTime < 2000 && classIndex === lastLoggedClass) return;

    const timeStr = new Date().toLocaleTimeString('es-ES', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const code = 'VIT-' + Math.random().toString(36).substr(2, 6).toUpperCase();
    const statusText = classIndex === 2 ? 'FONDO' : 'OK';
    const statusClass = classIndex === 2 ? '' : 'status-ok';

    detectionHistory.push({
        time: timeStr,
        label: label,
        status: statusText,
        code: code
    });

    const body = document.getElementById('log-body');
    if (body) {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${timeStr}</td>
            <td><strong>${label}</strong></td>
            <td><span class="${statusClass}">${statusText}</span></td>
            <td><code style="font-size:0.7rem">${code}</code></td>
        `;
        body.prepend(row);
        if (body.children.length > 50) body.lastChild.remove();
    }

    lastLogTime = now;
    lastLoggedClass = classIndex;
}

function addEventToFeed(label, sentStatus = 'pending') {
    const feed = document.getElementById('event-list');
    if (!feed) return;

    detectionCounter++;
    const currentId = detectionCounter;

    // Remove empty message if exists
    const emptyMsg = feed.querySelector('.empty-feed-msg');
    if (emptyMsg) emptyMsg.remove();

    const card = document.createElement('div');
    card.className = 'event-card';

    const now = new Date();
    const dateStr = now.toLocaleDateString();
    const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

    const statusText = sentStatus === 'sent' ? 'ENVIADO ✅' : 'DETECTADO';
    const statusClass = sentStatus === 'sent' ? 'sent' : 'pending';

    card.innerHTML = `
        <div class="event-info">
            <span class="event-title">${label} #${currentId}</span>
            <span class="event-time">${dateStr} - ${timeStr}</span>
        </div>
        <div class="event-status ${statusClass}">${statusText}</div>
    `;

    feed.prepend(card);
    if (feed.children.length > 10) feed.lastChild.remove();
}

async function sendTelegramPhoto(imageData, label, score) {
    if (!telegramToken || !telegramChatId || isNotifyingTelegram) return;

    const now = Date.now();
    if (now - lastTelegramTargetTime < telegramCooldown) return;

    isNotifyingTelegram = true;
    const statusMsg = document.getElementById('overlay-status');
    statusMsg.innerText = '📤 Enviando a Telegram...';
    statusMsg.style.color = '#38bdf8';

    try {
        const blob = await (await fetch(imageData)).blob();
        const formData = new FormData();
        formData.append('chat_id', telegramChatId);
        formData.append('photo', blob, 'deteccion.jpg');
        formData.append('caption', `🚀 Visión IT - Objeto Detectado\n\n📦 Producto: ${label}\n🎯 Precisión: ${Math.round(score * 100)}%\n⏰ Hora: ${new Date().toLocaleTimeString()}\n#VisionIT #Demo`);

        const response = await fetch(`https://api.telegram.org/bot${telegramToken}/sendPhoto`, {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            statusMsg.innerText = '✅ Imagen enviada';
            statusMsg.style.color = '#2dd4bf';
            lastTelegramTargetTime = now;
            addEventToFeed(label, 'sent');
        } else {
            statusMsg.innerText = '❌ Error API Telegram';
            statusMsg.style.color = '#ef4444';
        }
    } catch (err) {
        console.error('Telegram Error:', err);
        statusMsg.innerText = '⚠️ Error de Red';
    } finally {
        setTimeout(() => {
            isNotifyingTelegram = false;
            if (statusMsg.innerText.includes('enviada')) {
                statusMsg.innerText = 'A la espera de nueva detección...';
            }
        }, 3000);
    }
}

async function generatePDF() {
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();

    const batchNum = document.getElementById('batch-number').value || 'S/N';
    const batchDesc = document.getElementById('batch-desc').value || 'Sin descripción';

    doc.setFontSize(22);
    doc.setTextColor(56, 189, 248);
    doc.text('REPORTE DE PRODUCCIÓN - VISIÓN IT', 14, 20);

    doc.setFontSize(10);
    doc.setTextColor(100);
    doc.text(`Fecha: ${new Date().toLocaleDateString()} | Generado por: Vision IT IA`, 14, 28);

    doc.setFontSize(12);
    doc.setTextColor(0);
    doc.text(`Nº de Lote: ${batchNum}`, 14, 40);
    doc.text(`Descripción: ${batchDesc}`, 14, 47);

    const name0 = document.getElementById('name-class-0').value || 'Objeto 1';
    const name1 = document.getElementById('name-class-1').value || 'Objeto 2';
    const counts = [0, 0];
    detectionHistory.forEach(d => {
        if (d.label === name0) counts[0]++;
        if (d.label === name1) counts[1]++;
    });

    doc.autoTable({
        startY: 55,
        head: [['Categoría', 'Cantidad Detectada', 'Estado']],
        body: [
            [name0, counts[0], 'Completado'],
            [name1, counts[1], 'Completado'],
            ['TOTAL PRODUCCIÓN', counts[0] + counts[1], 'FINALIZADO']
        ],
        theme: 'striped',
        headStyles: { fillStyle: [56, 189, 248] }
    });

    const tableData = detectionHistory.map(d => [d.time, d.label, d.status, d.code]);

    doc.autoTable({
        startY: doc.lastAutoTable.finalY + 20,
        head: [['Hora', 'Elemento', 'Estado', 'Código Verificación']],
        body: tableData,
        theme: 'grid'
    });

    doc.save(`reporte-produccion-${batchNum}-${Date.now()}.pdf`);
}

function drawGrid(ctx) {
    if (!isCountingMode) return;
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.2)';
    ctx.lineWidth = 1;
    const cols = 4;
    const rows = 4;
    const cw = canvas.width / cols;
    const ch = canvas.height / rows;

    for (let i = 1; i < cols; i++) {
        ctx.beginPath(); ctx.moveTo(i * cw, 0); ctx.lineTo(i * cw, canvas.height); ctx.stroke();
    }
    for (let i = 1; i < rows; i++) {
        ctx.beginPath(); ctx.moveTo(0, i * ch); ctx.lineTo(canvas.width, i * ch); ctx.stroke();
    }
}

async function detect() {
    if (!model) {
        requestAnimationFrame(detect);
        return;
    }

    const ctx = canvas.getContext('2d');
    if (video.videoWidth) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const predictions = await model.detect(video);
    const filtered = predictions.filter(p => p.score >= confidenceThreshold);

    // --- DEMO MODE TARGETING ---
    if (isDemoMode) {
        const targets = filtered.filter(p => p.class === targetClass);
        const overlay = document.getElementById('detection-overlay');

        if (targets.length > 0) {
            const bestTarget = targets.reduce((prev, current) => (prev.score > current.score) ? prev : current);
            document.getElementById('overlay-product').innerText = bestTarget.class;
            document.getElementById('overlay-conf').innerText = `${Math.round(bestTarget.score * 100)}%`;
            overlay.classList.remove('hidden');

            if (targetDetectionStartTime === 0) targetDetectionStartTime = Date.now();

            if (Date.now() - targetDetectionStartTime > detectionRequiredTime) {
                if (!isNotifyingTelegram && (Date.now() - lastTelegramTargetTime > telegramCooldown)) {
                    const screenshot = captureForTelegram();
                    sendTelegramPhoto(screenshot, bestTarget.class, bestTarget.score);
                }
            }

            const [x, y, w, h] = bestTarget.bbox;
            ctx.strokeStyle = '#2dd4bf';
            ctx.lineWidth = 5;
            ctx.strokeRect(x, y, w, h);
        } else {
            overlay.classList.add('hidden');
            targetDetectionStartTime = 0;
        }
    } else {
        filtered.forEach(prediction => {
            const [x, y, width, height] = prediction.bbox;
            ctx.strokeStyle = '#38bdf8';
            ctx.lineWidth = 3;
            ctx.strokeRect(x, y, width, height);
        });
    }

    requestAnimationFrame(detect);
}

async function addExample(classId) {
    const img = tf.browser.fromPixels(video);
    const resized = tf.image.resizeBilinear(img, [224, 224]);
    const activation = featureExtractor.infer(resized, 'conv_preds');
    classifier.addExample(activation, classId);
    img.dispose();
    resized.dispose();

    classSampleCounts[classId]++;
    const thumbCanvas = document.createElement('canvas');
    thumbCanvas.width = 100;
    thumbCanvas.height = 100;
    const tCtx = thumbCanvas.getContext('2d');
    const size = Math.min(video.videoWidth, video.videoHeight);
    const startX = (video.videoWidth - size) / 2;
    const startY = (video.videoHeight - size) / 2;
    tCtx.drawImage(video, startX, startY, size, size, 0, 0, 100, 100);

    const base64 = thumbCanvas.toDataURL('image/jpeg', 0.6);
    classThumbnails[classId].push(base64);
    if (classThumbnails[classId].length > 8) classThumbnails[classId].shift();
    updateUIFeedback();
}

// Event Listeners
[0, 1, 2].forEach(id => {
    const btn = document.getElementById(`add-class-${id}`);
    if (!btn) return;
    let interval;
    const start = (e) => {
        if (e.cancelable) e.preventDefault();
        btn.classList.add('recording');
        addExample(id);
        interval = setInterval(() => addExample(id), 200);
    };
    const end = () => {
        clearInterval(interval);
        btn.classList.remove('recording');
        status.innerText = 'Fotos procesadas';
    };
    btn.addEventListener('mousedown', start);
    btn.addEventListener('mouseup', end);
    btn.addEventListener('mouseleave', end);
    btn.addEventListener('touchstart', start, { passive: false });
    btn.addEventListener('touchend', end);
});

openSettings.addEventListener('click', () => settingsDrawer.classList.add('active'));
closeSettings.addEventListener('click', () => settingsDrawer.classList.remove('active'));

confSlider.addEventListener('input', (e) => {
    confidenceThreshold = e.target.value / 100;
    confVal.innerText = e.target.value;
});

// Basic settings still reachable via drawer if needed, but UI options removed for Beta baseline
if (typeof speechToggle !== 'undefined' && speechToggle) speechToggle.addEventListener('change', (e) => isSpeechEnabled = e.target.checked);
if (typeof countModeToggle !== 'undefined' && countModeToggle) countModeToggle.addEventListener('change', (e) => isCountingMode = e.target.checked);

trainModeBtn.addEventListener('click', () => {
    isTrainingMode = !isTrainingMode;
    trainModeBtn.innerText = isTrainingMode ? 'Desactivar IA Personalizada' : 'Activar IA Personalizada';
    aiModeText.innerText = isTrainingMode ? 'Modo: APRENDIZAJE / CONTEO' : 'Modo: Estándar (Inspección General)';
    if (!isTrainingMode) {
        customPredBox.classList.add('hidden');
    }
});

document.getElementById('main-train-toggle').addEventListener('click', () => {
    const panel = document.getElementById('main-training-panel');
    if (panel) panel.classList.toggle('hidden');
    const isActive = panel && !panel.classList.contains('hidden');
    document.getElementById('main-train-toggle').classList.toggle('recording', isActive);
});

document.getElementById('generate-pdf-btn').addEventListener('click', generatePDF);
saveModelBtn.addEventListener('click', saveModel);

resetTraining.addEventListener('click', () => {
    if (confirm('¿Reiniciar toda la base de datos de aprendizaje?')) {
        classifier.clearAllClasses();
        localStorage.removeItem('vision_it_model_v3');
        classSampleCounts = [0, 0, 0];
        classThumbnails = [[], [], []];
        updateUIFeedback();
        status.innerText = 'Memoria borrada';
    }
});

captureBtn.addEventListener('click', () => {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    const tCtx = tempCanvas.getContext('2d');
    tCtx.drawImage(video, 0, 0);
    tCtx.drawImage(canvas, 0, 0);
    const link = document.createElement('a');
    link.download = `captura-it-${Date.now()}.png`;
    link.href = tempCanvas.toDataURL('image/png');
    link.click();
});

function captureForTelegram() {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    const tCtx = tempCanvas.getContext('2d');
    tCtx.drawImage(video, 0, 0);
    tCtx.drawImage(canvas, 0, 0);
    return tempCanvas.toDataURL('image/jpeg', 0.8);
}

// Settings Listeners
document.getElementById('tg-token').value = telegramToken;
document.getElementById('tg-chatid').value = telegramChatId;

document.getElementById('tg-token').addEventListener('input', (e) => {
    telegramToken = e.target.value;
    localStorage.setItem('tg_token', telegramToken);
});
document.getElementById('tg-chatid').addEventListener('input', (e) => {
    telegramChatId = e.target.value;
    localStorage.setItem('tg_chatid', telegramChatId);
});

document.querySelectorAll('.btn-edit-setting').forEach(btn => {
    btn.addEventListener('click', (e) => {
        const input = e.target.closest('.settings-input-group').querySelector('.settings-input');
        input.focus();
    });
});
document.getElementById('target-class-select').addEventListener('change', (e) => targetClass = e.target.value);
document.getElementById('demo-mode-toggle').addEventListener('change', (e) => {
    isDemoMode = e.target.checked;
    document.querySelector('.dashboard-layout').classList.toggle('demo-mode', isDemoMode);
    const panel = document.querySelector('.info-panel');
    if (isDemoMode) panel.classList.add('demo-hidden');
    else panel.classList.remove('demo-hidden');
});

document.getElementById('toggle-advanced-ui').addEventListener('click', () => {
    const panel = document.querySelector('.info-panel');
    panel.classList.toggle('demo-hidden');
});

toggleCamBtn.addEventListener('click', async () => {
    currentFacingMode = (currentFacingMode === 'user') ? 'environment' : 'user';
    await setupCamera();
});

// Final Initialization
(async () => {
    status.innerText = 'Inicializando IA y Cámara...';
    try {
        // Parallel load
        const [camVideo, _] = await Promise.all([
            setupCamera(),
            loadModels()
        ]);

        status.innerText = 'IA y Cámara Listas';
        status.style.color = '#2dd4bf';
        detect();
    } catch (err) {
        console.error('Core Init Failed:', err);
        if (!status.innerText.includes('Cámara')) {
            status.innerText = 'Error de conexión / Modelos';
        }
        status.style.color = '#ef4444';
    }
})();
