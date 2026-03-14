const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const status = document.getElementById('status');
const labelsContainer = document.createElement('div');
labelsContainer.id = 'labels-container';
const toggleCamBtn = document.getElementById('toggle-camera');

// UI Elements
const openSettings = document.getElementById('open-settings');
const closeSettings = document.getElementById('close-settings');
const settingsDrawer = document.getElementById('settings-drawer');
const confSlider = document.getElementById('conf-slider');
const confVal = document.getElementById('conf-val');

let detectionHistory = [];
let lastLogTime = 0;

let model;
let currentFacingMode = 'environment';
let confidenceThreshold = 0.05; // Very low floor — let the model speak
let isSpeechEnabled = false;
let lastSpoken = '';
let lastSpokenTime = 0;

// Demo Mode & Telegram Config
let isDemoMode = true;
let targetClass = 'cell phone';
let telegramToken = localStorage.getItem('tg_token') || '8535485891:AAEvAOiKwef-PlGffxwcJubUKYuB819sd90';
let telegramChatId = localStorage.getItem('tg_chatid') || '1577936762';
let isNotifyingTelegram = false;
let lastTelegramTargetTime = 0;
let telegramCooldown = 30000;
let targetDetectionStartTime = 0;
let detectionRequiredTime = 400;
let detectionCounter = 0;

// Frame throttle: only run inference every N ms (reduces CPU overload on mobile)
let lastDetectTime = 0;
const DETECT_INTERVAL_MS = 250; // 4fps max for inference — enough for real-time feel

// ─── Camera ───────────────────────────────────────────────────────────────────
async function setupCamera() {
    status.innerText = 'Conectando cámara...';
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(t => t.stop());
    }

    // On mobile, request lower resolution to avoid memory pressure
    const isMobile = /Mobi|Android|iPhone/i.test(navigator.userAgent);
    const constraints = {
        video: {
            facingMode: { ideal: currentFacingMode },
            width: { ideal: isMobile ? 640 : 1280 },
            height: { ideal: isMobile ? 480 : 720 }
        },
        audio: false
    };

    try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = stream;

        return new Promise((resolve, reject) => {
            const onReady = () => {
                video.play()
                    .then(() => resolve(video))
                    .catch(reject);
            };
            if (video.readyState >= 3) {
                onReady();
            } else {
                video.onloadeddata = onReady;
            }
            setTimeout(() => reject(new Error('Timeout de cámara')), 8000);
        });
    } catch (err) {
        console.error('Camera error:', err);
        if (err.name === 'NotAllowedError') {
            status.innerText = '🚫 Cámara bloqueada — permitir en el navegador';
        } else if (err.name === 'NotFoundError') {
            status.innerText = '❌ No se encontró ninguna cámara';
        } else {
            status.innerText = `❌ Error: ${err.message}`;
        }
        throw err;
    }
}

// ─── Model Loading ────────────────────────────────────────────────────────────
// Use lite_mobilenet_v2: faster, lower memory, better compatibility on mobile
async function loadModels() {
    status.innerText = '🧠 Cargando modelo IA...';
    try {
        model = await cocoSsd.load({ base: 'lite_mobilenet_v2' });
        console.log('Model loaded: lite_mobilenet_v2');
    } catch (err) {
        console.error('Model load failed:', err);
        status.innerText = '❌ Error cargando modelo IA';
        throw err;
    }
}

// ─── Speech ───────────────────────────────────────────────────────────────────
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

// ─── Detection Log & Feed ─────────────────────────────────────────────────────
function logDetection(label) {
    const timeStr = new Date().toLocaleTimeString('es-ES', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const dateStr = new Date().toLocaleDateString('es-ES');
    detectionHistory.push({ time: timeStr, date: dateStr, label });
}

function addEventToFeed(label, sentStatus = 'pending') {
    const feed = document.getElementById('event-list');
    if (!feed) return;

    detectionCounter++;
    const currentId = detectionCounter;

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

// ─── Telegram ─────────────────────────────────────────────────────────────────
async function sendTelegramPhoto(imageData, label, score) {
    if (!telegramToken || !telegramChatId || isNotifyingTelegram) return;
    const now = Date.now();
    if (now - lastTelegramTargetTime < telegramCooldown) return;

    isNotifyingTelegram = true;
    const statusMsg = document.getElementById('overlay-status');
    if (statusMsg) { statusMsg.innerText = '📤 Enviando a Telegram...'; statusMsg.style.color = '#38bdf8'; }

    try {
        const blob = await (await fetch(imageData)).blob();
        const formData = new FormData();
        formData.append('chat_id', telegramChatId);
        formData.append('photo', blob, 'deteccion.jpg');
        formData.append('caption', `🚀 Visión IT - Objeto Detectado\n\n📦 Producto: ${label}\n🎯 Precisión: ${Math.round(score * 100)}%\n⏰ Hora: ${new Date().toLocaleTimeString()}\n#VisionIT`);

        const response = await fetch(`https://api.telegram.org/bot${telegramToken}/sendPhoto`, {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            if (statusMsg) { statusMsg.innerText = '✅ Imagen enviada'; statusMsg.style.color = '#2dd4bf'; }
            lastTelegramTargetTime = now;
            addEventToFeed(label, 'sent');
        } else {
            if (statusMsg) { statusMsg.innerText = '❌ Error API Telegram'; statusMsg.style.color = '#ef4444'; }
        }
    } catch (err) {
        console.error('Telegram error:', err);
        if (statusMsg) { statusMsg.innerText = '⚠️ Error de Red'; }
    } finally {
        setTimeout(() => {
            isNotifyingTelegram = false;
            if (statusMsg && statusMsg.innerText.includes('enviada')) {
                statusMsg.innerText = 'Escaneando...';
            }
        }, 3000);
    }
}

// ─── Canvas drawing helpers ───────────────────────────────────────────────────
function drawBox(ctx, x, y, w, h, color, label) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.strokeRect(x, y, w, h);

    // Label background
    const textW = ctx.measureText(label).width + 12;
    const textH = 22;
    ctx.fillStyle = color;
    ctx.fillRect(x, y - textH, textW, textH);
    ctx.fillStyle = '#000';
    ctx.font = 'bold 13px sans-serif';
    ctx.fillText(label, x + 6, y - 6);
}

// ─── Main Detection Loop ──────────────────────────────────────────────────────
async function detect() {
    // Throttle: skip frame if not enough time passed
    const now = Date.now();
    if (now - lastDetectTime < DETECT_INTERVAL_MS) {
        requestAnimationFrame(detect);
        return;
    }

    if (!model) {
        requestAnimationFrame(detect);
        return;
    }

    // Guard: video must be ready and have dimensions
    if (video.readyState < 2 || !video.videoWidth || !video.videoHeight) {
        requestAnimationFrame(detect);
        return;
    }

    lastDetectTime = now;

    // Sync canvas to native video resolution
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
    }

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    let predictions = [];
    try {
        // Use a very low threshold so COCO-SSD returns everything it sees
        predictions = await model.detect(video, undefined, 0.03);
    } catch (err) {
        console.error('detect() error:', err);
        requestAnimationFrame(detect);
        return;
    }

    // Always draw ALL detected objects (helps debug what the model sees)
    // Color: teal for target, blue for everything else
    const overlay = document.getElementById('detection-overlay');
    let foundTarget = false;

    predictions.forEach(p => {
        if (p.score < confidenceThreshold) return; // apply user slider filter for feed/overlay only

        const [x, y, w, h] = p.bbox;
        const isTarget = (p.class === targetClass);
        const color = isTarget ? '#2dd4bf' : 'rgba(56,189,248,0.6)';
        const labelText = `${p.class} ${Math.round(p.score * 100)}%`;
        drawBox(ctx, x, y, w, h, color, labelText);

        if (isTarget) {
            foundTarget = true;
            const conf = Math.round(p.score * 100);
            document.getElementById('overlay-product').innerText = p.class;
            document.getElementById('overlay-conf').innerText = `${conf}%`;

            if (targetDetectionStartTime === 0) targetDetectionStartTime = now;

            if (now - targetDetectionStartTime > detectionRequiredTime) {
                if (now - lastLogTime > 5000) {
                    lastLogTime = now;
                    logDetection(p.class);
                    addEventToFeed(p.class, 'sent');
                }
                if (!isNotifyingTelegram && (now - lastTelegramTargetTime > telegramCooldown)) {
                    const screenshot = captureForTelegram();
                    sendTelegramPhoto(screenshot, p.class, p.score);
                }
            }
        }
    });

    if (foundTarget) {
        overlay.classList.remove('hidden');
    } else {
        overlay.classList.add('hidden');
        targetDetectionStartTime = 0;
    }

    requestAnimationFrame(detect);
}

// ─── Capture for Telegram ─────────────────────────────────────────────────────
function captureForTelegram() {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    const tCtx = tempCanvas.getContext('2d');
    tCtx.drawImage(video, 0, 0);
    tCtx.drawImage(canvas, 0, 0);
    return tempCanvas.toDataURL('image/jpeg', 0.8);
}

// ─── PDF & Telegram Report ────────────────────────────────────────────────────
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
            head: [['Fecha', 'Hora', 'Objeto', 'Estado']],
            body: tableData,
            theme: 'grid',
            headStyles: { fillColor: [45, 212, 191] }
        });

        const pdfBlob = doc.output('blob');
        const pdfUrl = URL.createObjectURL(pdfBlob);
        const link = document.createElement('a');
        link.href = pdfUrl;
        link.download = `reporte-vision-it-${Date.now()}.pdf`;
        link.click();

        await sendPDFToTelegram(pdfBlob);
        exportBtn.innerText = 'PDF Enviado ✅';
    } catch (err) {
        console.error('PDF Error:', err);
        exportBtn.innerText = 'Error al generar';
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
    formData.append('document', blob, 'reporte-vision-it.pdf');
    formData.append('caption', '📄 Reporte de detecciones Visión IT');
    try {
        await fetch(`https://api.telegram.org/bot${telegramToken}/sendDocument`, {
            method: 'POST',
            body: formData
        });
    } catch (err) {
        console.error('Telegram PDF upload failed:', err);
    }
}

// ─── UI Listeners ─────────────────────────────────────────────────────────────
openSettings.addEventListener('click', () => settingsDrawer.classList.add('active'));
closeSettings.addEventListener('click', () => settingsDrawer.classList.remove('active'));

confSlider.addEventListener('input', (e) => {
    confidenceThreshold = e.target.value / 100;
    confVal.innerText = e.target.value;
});

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

toggleCamBtn.addEventListener('click', async () => {
    currentFacingMode = (currentFacingMode === 'user') ? 'environment' : 'user';
    try {
        await setupCamera();
        status.innerText = '📷 Cámara cambiada';
        setTimeout(() => status.innerText = 'Sistema Online', 1500);
    } catch (err) {
        console.error(err);
    }
});

// ─── Init ─────────────────────────────────────────────────────────────────────
async function init() {
    try {
        await loadModels();
        status.innerText = '📷 Iniciando cámara...';
        await setupCamera();
        status.innerText = 'Sistema Online';
        status.style.color = '#2dd4bf';
        detect();
    } catch (err) {
        console.error('Init error:', err);
        status.style.color = '#ef4444';
        // Error message already set inside loadModels / setupCamera
    }
}

init();
