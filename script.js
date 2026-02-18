const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const status = document.getElementById('status');
const countNum = document.getElementById('count-num');
const labelsContainer = document.getElementById('labels-container');
const toggleCamBtn = document.getElementById('toggle-camera');

// UI Elements
const openSettings = document.getElementById('open-settings');
const closeSettings = document.getElementById('close-settings');
const settingsDrawer = document.getElementById('settings-drawer');
const confSlider = document.getElementById('conf-slider');
const confVal = document.getElementById('conf-val');
const speechToggle = document.getElementById('speech-toggle');
const captureBtn = document.getElementById('capture-btn');

// Training UI
const trainModeBtn = document.getElementById('train-mode-btn');
const resetTraining = document.getElementById('reset-training');
const aiModeText = document.getElementById('ai-mode-text');
const customPredBox = document.getElementById('custom-prediction');
const customLabel = document.getElementById('custom-label');

let model;
let featureExtractor; // MobileNet
let classifier; // KNN Classifier
let currentFacingMode = 'environment';
let confidenceThreshold = 0.6;
let isSpeechEnabled = false;
let isTrainingMode = false;
let lastSpoken = "";
let lastSpokenTime = 0;

// Camera Handling
async function setupCamera() {
    status.innerText = 'Buscando cámara...';
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

        return new Promise((resolve) => {
            video.onloadeddata = () => {
                video.play();
                resolve(video);
            };
        });
    } catch (err) {
        status.innerText = 'Sin acceso a cámara';
        console.error(err);
        alert('Por favor activa los permisos de cámara');
    }
}

// Model Loading
async function loadModels() {
    status.innerText = 'Conectando cerebros...';
    try {
        // Load in parallel
        const [cocoRes, mobRes] = await Promise.all([
            cocoSsd.load(),
            mobilenet.load({ version: 2, alpha: 1.0 })
        ]);

        model = cocoRes;
        featureExtractor = mobRes;
        classifier = knnClassifier.create();

        status.innerText = 'IA Lista';
    } catch (err) {
        status.innerText = 'Error al cargar IA';
        console.error(err);
    }
}

function speak(text) {
    if (!isSpeechEnabled) return;
    const now = Date.now();
    if (now - lastSpokenTime < 4000 && text === lastSpoken) return;

    // Cancel previous speech to be more responsive
    window.speechSynthesis.cancel();

    const u = new SpeechSynthesisUtterance(text);
    u.lang = 'es-ES';
    u.rate = 1.1; // Slightly faster for responsiveness
    window.speechSynthesis.speak(u);
    lastSpoken = text;
    lastSpokenTime = now;
}

async function detect() {
    if (!model) {
        requestAnimationFrame(detect);
        return;
    }

    const ctx = canvas.getContext('2d');
    // Actual frame dimensions
    if (video.videoWidth) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (isTrainingMode) {
        // --- CUSTOM CLASSIFICATION MODE ---
        if (classifier.getNumClasses() > 0) {
            const img = tf.browser.fromPixels(video);
            const activation = featureExtractor.infer(img, 'conv_preds');
            const result = await classifier.predictClass(activation);

            const classes = ['Objeto 1', 'Objeto 2', 'NADA/Fondo'];
            const label = classes[result.label];
            const probability = result.confidences[result.label];

            if (probability > 0.6) {
                customLabel.innerText = `${label} (${Math.round(probability * 100)}%)`;
                customPredBox.classList.remove('hidden');

                // Visual Highlight on Canvas
                ctx.strokeStyle = 'var(--accent)';
                ctx.lineWidth = 10;
                ctx.strokeRect(canvas.width * 0.1, canvas.height * 0.1, canvas.width * 0.8, canvas.height * 0.8);

                if (label !== 'NADA/Fondo') {
                    speak(`Esto es ${label}`);
                }
            } else {
                customLabel.innerText = '---';
            }
            img.dispose();
        }
    } else {
        // --- STANDARD DETECTION MODE ---
        const predictions = await model.detect(video);
        const filtered = predictions.filter(p => p.score >= confidenceThreshold);

        countNum.innerText = filtered.length;
        labelsContainer.innerHTML = '';

        filtered.forEach(prediction => {
            const [x, y, width, height] = prediction.bbox;

            // Draw Box
            ctx.strokeStyle = '#38bdf8';
            ctx.lineWidth = 4;
            ctx.strokeRect(x, y, width, height);

            // Create Label Element
            const tag = document.createElement('div');
            tag.className = 'label-tag';
            tag.innerHTML = `<strong>${prediction.class}</strong><span>${Math.round(prediction.score * 100)}%</span>`;
            labelsContainer.appendChild(tag);
        });

        if (filtered.length > 0) {
            const sumNames = [...new Set(filtered.map(f => f.class))].join(' y ');
            speak(`Hay ${filtered.length} objetos: ${sumNames}`);
        }
    }

    requestAnimationFrame(detect);
}

// Training Interactions
async function addExample(classId) {
    if (!isTrainingMode) return;

    // Visual feedback
    const btn = document.getElementById(`add-class-${classId}`);
    btn.classList.add('recording');
    status.innerText = 'Entrenando...';

    const img = tf.browser.fromPixels(video);
    const activation = featureExtractor.infer(img, 'conv_preds');
    classifier.addExample(activation, classId);
    img.dispose();

    setTimeout(() => {
        btn.classList.remove('recording');
        status.innerText = 'IA Lista';
    }, 200);
}

// Event Listeners
[0, 1, 2].forEach(id => {
    const btn = document.getElementById(`add-class-${id}`);

    // Support for both mouse and touch (holding)
    let interval;
    const start = (e) => {
        if (e.cancelable) e.preventDefault();
        addExample(id);
        interval = setInterval(() => addExample(id), 100);
    };
    const end = () => clearInterval(interval);

    btn.addEventListener('mousedown', start);
    btn.addEventListener('mouseup', end);
    btn.addEventListener('mouseleave', end);

    btn.addEventListener('touchstart', start, { passive: false });
    btn.addEventListener('touchend', end);
});

// General UI
openSettings.addEventListener('click', () => settingsDrawer.classList.add('active'));
closeSettings.addEventListener('click', () => settingsDrawer.classList.remove('active'));

confSlider.addEventListener('input', (e) => {
    confidenceThreshold = e.target.value / 100;
    confVal.innerText = e.target.value;
});

speechToggle.addEventListener('change', (e) => isSpeechEnabled = e.target.checked);

trainModeBtn.addEventListener('click', () => {
    isTrainingMode = !isTrainingMode;
    trainModeBtn.innerText = isTrainingMode ? 'Desactivar Entrenamiento' : 'Activar Entrenamiento';
    aiModeText.innerText = isTrainingMode ? 'Modo: APRENDIZAJE' : 'Modo: Estándar (COCO-SSD)';
    customPredBox.classList.toggle('hidden', !isTrainingMode);
});

resetTraining.addEventListener('click', () => {
    classifier.clearAllClasses();
    status.innerText = 'Cerebro Reiniciado';
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

toggleCamBtn.addEventListener('click', async () => {
    currentFacingMode = (currentFacingMode === 'user') ? 'environment' : 'user';
    await setupCamera();
});

// Init
(async () => {
    await setupCamera();
    await loadModels();
    detect();
})();
