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

let detectionHistory = []; // For PDF report
let lastLogTime = 0;
let lastLoggedClass = -1;

let model;
let modelType = 'blazeface';
let currentFacingMode = 'user';
let confidenceThreshold = 0.50;
let isSpeechEnabled = false;
let lastSpoken = "";
let lastSpokenTime = 0;

// Telegram & Detection Config
let telegramToken = localStorage.getItem('tg_token') || '8535485891:AAEvAOiKwef-PlGffxwcJubUKYuB819sd90';
let telegramChatId = localStorage.getItem('tg_chatid') || '1577936762';
let isNotifyingTelegram = false;
let lastTelegramTargetTime = 0;
let telegramCooldown = 30000; // 30s between photos
let targetDetectionStartTime = 0;
let detectionCounter = 0; // Sequential ID for detections

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
// Persistence Logic removed per Regla de Oro (Training is disabled/deleted)
function loadModel() { }
function updateUIFeedback() { }

// Model Loading (BlazeFace high-speed with WebGL acceleration)
async function loadModels() {
    try {
        if (window.tf) {
            try {
                await tf.setBackend('webgl');
                await tf.ready();
            } catch (backendErr) {
                console.warn('WebGL initialization warning, using default:', backendErr);
            }
        }
        if (typeof blazeface !== 'undefined') {
            model = await blazeface.load();
            modelType = 'blazeface';
            console.log('BlazeFace model loaded successfully with backend:', window.tf ? tf.getBackend() : 'auto');
        } else if (typeof cocoSsd !== 'undefined') {
            model = await cocoSsd.load({ base: 'lite_mobilenet_v2' });
            modelType = 'cocossd';
            console.log('COCO-SSD fallback loaded');
        } else {
            throw new Error('No detection model found');
        }
    } catch (err) {
        console.warn('Error loading BlazeFace, trying COCO-SSD fallback:', err);
        if (typeof cocoSsd !== 'undefined') {
            model = await cocoSsd.load({ base: 'lite_mobilenet_v2' });
            modelType = 'cocossd';
        } else {
            throw err;
        }
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

function logDetection(label) {
    const timeStr = new Date().toLocaleTimeString('es-ES', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const dateStr = new Date().toLocaleDateString('es-ES');
    detectionHistory.push({ time: timeStr, date: dateStr, label: label });
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
        formData.append('caption', `🚀 Visión IT - Rostro Detectado\n\n👤 Detección: ${label}\n🎯 Precisión: ${Math.round(score * 100)}%\n⏰ Hora: ${new Date().toLocaleTimeString()}\n#VisionIT #FaceDetection`);

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

// Grid Analysis removed per Regla de Oro

let lastDetectionTime = 0;
const isMobileDevice = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
// BlazeFace is ultra-optimized: 50-80ms provides instant, smooth 20+ FPS real-time tracking
const DETECTION_THROTTLE_MS = isMobileDevice ? 80 : 40;

async function detect() {
    if (!model) {
        requestAnimationFrame(detect);
        return;
    }

    const now = Date.now();
    if (now - lastDetectionTime < DETECTION_THROTTLE_MS) {
        requestAnimationFrame(detect);
        return;
    }
    lastDetectionTime = now;

    const ctx = canvas.getContext('2d');
    if (video.videoWidth && (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight)) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Guard: don't run if video isn't ready
    if (video.readyState < 2 || !video.videoWidth) {
        requestAnimationFrame(detect);
        return;
    }

    const minScore = Math.max(confidenceThreshold, 0.20);
    const stableTimeNeeded = 100; // Ultra-fast response (<100ms) for immediate feedback
    const overlay = document.getElementById('detection-overlay');

    let detectedFaces = [];

    try {
        if (modelType === 'blazeface') {
            const predictions = await model.estimateFaces(video, false);
            detectedFaces = predictions
                .map(p => {
                    const score = Array.isArray(p.probability) ? p.probability[0] : (p.probability || 0.95);
                    const [x1, y1] = p.topLeft;
                    const [x2, y2] = p.bottomRight;
                    return {
                        x: x1,
                        y: y1,
                        w: Math.max(x2 - x1, 10),
                        h: Math.max(y2 - y1, 10),
                        score: score,
                        landmarks: p.landmarks
                    };
                })
                .filter(f => f.score >= minScore);
        } else {
            // Fallback COCO-SSD for person
            const predictions = await model.detect(video, 10, 0.1);
            detectedFaces = predictions
                .filter(p => p.class === 'person' && p.score >= minScore)
                .map(p => {
                    const [x, y, w, h] = p.bbox;
                    return {
                        x: x + w * 0.15,
                        y: y,
                        w: w * 0.7,
                        h: Math.min(h * 0.45, w * 0.9),
                        score: p.score
                    };
                });
        }
    } catch (err) {
        console.warn('Inference error:', err);
    }

    if (detectedFaces.length > 0) {
        const bestTarget = detectedFaces.reduce((prev, current) => (prev.score > current.score) ? prev : current);
        const { x, y, w, h } = bestTarget;

        // Draw futuristic cyber bounding box
        ctx.strokeStyle = '#2dd4bf';
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w, h);

        // Tech corner accents
        const cornerLen = Math.min(w, h) * 0.22;
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 5;

        // Top-Left corner
        ctx.beginPath();
        ctx.moveTo(x, y + cornerLen);
        ctx.lineTo(x, y);
        ctx.lineTo(x + cornerLen, y);
        ctx.stroke();

        // Top-Right corner
        ctx.beginPath();
        ctx.moveTo(x + w - cornerLen, y);
        ctx.lineTo(x + w, y);
        ctx.lineTo(x + w, y + cornerLen);
        ctx.stroke();

        // Bottom-Left corner
        ctx.beginPath();
        ctx.moveTo(x, y + h - cornerLen);
        ctx.lineTo(x, y + h);
        ctx.lineTo(x + cornerLen, y + h);
        ctx.stroke();

        // Bottom-Right corner
        ctx.beginPath();
        ctx.moveTo(x + w - cornerLen, y + h);
        ctx.lineTo(x + w, y + h);
        ctx.lineTo(x + w, y + h - cornerLen);
        ctx.stroke();

        // Draw facial landmarks (eyes, nose, mouth points)
        if (bestTarget.landmarks && bestTarget.landmarks.length) {
            ctx.fillStyle = '#38bdf8';
            for (let i = 0; i < bestTarget.landmarks.length; i++) {
                const [lx, ly] = bestTarget.landmarks[i];
                ctx.beginPath();
                ctx.arc(lx, ly, 3, 0, 2 * Math.PI);
                ctx.fill();
            }
        }

        // Cyber HUD tag above box
        ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
        ctx.fillRect(x, Math.max(0, y - 28), 160, 24);
        ctx.fillStyle = '#2dd4bf';
        ctx.font = 'bold 12px Outfit, sans-serif';
        ctx.fillText(`ROSTRO ${Math.round(bestTarget.score * 100)}%`, x + 8, Math.max(16, y - 11));

        // Show Overlay
        if (overlay) {
            document.getElementById('overlay-product').innerText = "ROSTRO HUMANO";
            document.getElementById('overlay-conf').innerText = `${Math.round(bestTarget.score * 100)}%`;
            overlay.classList.remove('hidden');
        }

        if (targetDetectionStartTime === 0) targetDetectionStartTime = Date.now();

        if (now - targetDetectionStartTime > stableTimeNeeded) {
            if (now - lastLogTime > 4000) { 
                lastLogTime = now;
                logDetection("Rostro Humano");
                addEventToFeed("Rostro", 'sent');
            }

            if (!isNotifyingTelegram && (now - lastTelegramTargetTime > telegramCooldown)) {
                const screenshot = captureForTelegram();
                sendTelegramPhoto(screenshot, "Rostro Humano", bestTarget.score);
            }
        }
    } else {
        if (overlay) overlay.classList.add('hidden');
        targetDetectionStartTime = 0;
    }

    requestAnimationFrame(detect);
}

// Essential UI listeners only
openSettings.addEventListener('click', () => settingsDrawer.classList.add('active'));
closeSettings.addEventListener('click', () => settingsDrawer.classList.remove('active'));

confSlider.addEventListener('input', (e) => {
    confidenceThreshold = e.target.value / 100;
    confVal.innerText = e.target.value;
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

async function generatePDFReport() {
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();
    const exportBtn = document.getElementById('export-pdf-btn');

    try {
        exportBtn.innerText = 'Generando...';
        exportBtn.disabled = true;

        doc.setFontSize(22);
        doc.setTextColor(45, 212, 191);
        doc.text('VISIÓN IT - REPORTE DE DETECCIONES', 14, 20);

        doc.setFontSize(10);
        doc.setTextColor(100);
        doc.text(`Generado el: ${new Date().toLocaleString('es-ES')}`, 14, 28);

        const tableData = detectionHistory.map(d => [d.date, d.time, d.label, 'OK']);

        doc.autoTable({
            startY: 40,
            head: [['Fecha', 'Hora', 'Detección', 'Estado']],
            body: tableData,
            theme: 'grid',
            headStyles: { fillStyle: [45, 212, 191] }
        });

        const pdfBlob = doc.output('blob');
        const pdfUrl = URL.createObjectURL(pdfBlob);

        // Download local copy
        const link = document.createElement('a');
        link.href = pdfUrl;
        link.download = `reporte-vision-it-${Date.now()}.pdf`;
        link.click();

        // Send to Telegram
        await sendPDFToTelegram(pdfBlob);

        exportBtn.innerText = 'PDF Enviado ✅';
    } catch (err) {
        console.error('PDF Error:', err);
        exportBtn.innerText = 'Error al enviar';
    } finally {
        setTimeout(() => {
            exportBtn.innerText = 'Generar PDF y Enviar';
            exportBtn.disabled = false;
        }, 3000);
    }
}

async function sendPDFToTelegram(blob) {
    if (!telegramToken || !telegramChatId) return;

    const formData = new FormData();
    formData.append('chat_id', telegramChatId);
    formData.append('document', blob, `reporte-vision-it.pdf`);
    formData.append('caption', '📄 Reporte de detecciones Visión IT');

    const url = `https://api.telegram.org/bot${telegramToken}/sendDocument`;

    try {
        await fetch(url, {
            method: 'POST',
            body: formData
        });
    } catch (err) {
        console.error('Telegram PDF upload failed:', err);
    }
}

// Settings Listeners
// Settings listeners maintained
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

document.getElementById('export-pdf-btn').addEventListener('click', generatePDFReport);

openSettings.addEventListener('click', () => {
    settingsDrawer.classList.add('active');
});

closeSettings.addEventListener('click', () => {
    settingsDrawer.classList.remove('active');
});

confSlider.addEventListener('input', (e) => {
    confidenceThreshold = e.target.value / 100;
    confVal.innerText = e.target.value;
});

toggleCamBtn.addEventListener('click', async () => {
    currentFacingMode = (currentFacingMode === 'user') ? 'environment' : 'user';
    await setupCamera();
    status.innerText = 'Sistema on line de reconocimiento de rostros en tiempo real';
});

// Final Initialization
async function init() {
    status.innerText = 'Cargando Modelo Facial de IA...';
    try {
        await loadModels();
        status.innerText = 'Modelo Listo. Iniciando Cámara...';
        await setupCamera();
        status.innerText = 'Sistema on line de reconocimiento de rostros en tiempo real';
        status.style.color = '#2dd4bf';
        detect();
    } catch (err) {
        console.error('Initialization Error:', err);
        status.style.color = '#ef4444';
        // Detailed error is already set in setupCamera for specific cases
    }
}

init();
