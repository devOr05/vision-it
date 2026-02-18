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

        return new Promise((resolve) => {
            video.onloadeddata = () => {
                video.play();
                resolve(video);
            };
        });
    } catch (err) {
        status.innerText = 'Cámara bloqueada';
        console.error(err);
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
        status.innerText = 'IA Activa';
    } catch (err) {
        status.innerText = 'Error de conexión';
        console.error(err);
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

    // --- SHARED DETECTION VALUES ---
    let cocoDetectedCount = 0;
    const name0 = document.getElementById('name-class-0').value || 'Objeto 1';
    const name1 = document.getElementById('name-class-1').value || 'Objeto 2';
    const name2 = document.getElementById('name-class-2').value || 'Fondo';
    const classes = [name0, name1, name2];

    // 1. RUN STANDARD COCO-SSD (Always if not in exclusive counting mode)
    if (!isCountingMode || !isTrainingMode) {
        const predictions = await model.detect(video);
        const filtered = predictions.filter(p => p.score >= confidenceThreshold);
        cocoDetectedCount = filtered.length;
        labelsContainer.innerHTML = '';

        filtered.forEach(prediction => {
            const [x, y, width, height] = prediction.bbox;
            ctx.strokeStyle = '#38bdf8';
            ctx.lineWidth = 3;
            ctx.strokeRect(x, y, width, height);

            const tag = document.createElement('div');
            tag.className = 'label-tag';
            tag.innerHTML = `<strong>${prediction.class}</strong><span>${Math.round(prediction.score * 100)}%</span>`;
            labelsContainer.appendChild(tag);
        });
    }

    // 2. RUN CUSTOM IA (If trained)
    if (isTrainingMode && classifier.getNumClasses() > 0) {
        if (isCountingMode) {
            // --- TILED COUNTING MODE ---
            drawGrid(ctx);
            const cols = 4;
            const rows = 4;
            const tw = video.videoWidth / cols;
            const th = video.videoHeight / rows;
            let totalCount = 0;

            const videoTensor = tf.browser.fromPixels(video);

            for (let y = 0; y < rows; y++) {
                for (let x = 0; x < cols; x++) {
                    const crop = videoTensor.slice([y * th, x * tw, 0], [th, tw, 3]);
                    const resizedCrop = tf.image.resizeBilinear(crop, [224, 224]);
                    const activation = featureExtractor.infer(resizedCrop, 'conv_preds');
                    const result = await classifier.predictClass(activation);

                    if (result.label !== "2" && result.confidences[result.label] > confidenceThreshold) {
                        totalCount++;
                        ctx.fillStyle = 'rgba(45, 212, 191, 0.6)';
                        ctx.beginPath();
                        ctx.arc((x * tw) + tw / 2, (y * th) + th / 2, 15, 0, Math.PI * 2);
                        ctx.fill();

                        ctx.fillStyle = "#fff";
                        ctx.font = "bold 14px sans-serif";
                        ctx.fillText(classes[parseInt(result.label)], (x * tw) + 5, (y * th) + 20);
                    }
                    crop.dispose();
                    resizedCrop.dispose();
                }
            }
            videoTensor.dispose();
            countNum.innerText = totalCount;
            customLabel.innerText = `Total ${name0}/${name1}: ${totalCount}`;
            customPredBox.classList.remove('hidden');
        } else {
            // --- FULL FRAME CLASSIFICATION ---
            const img = tf.browser.fromPixels(video);
            const resized = tf.image.resizeBilinear(img, [224, 224]);
            const activation = featureExtractor.infer(resized, 'conv_preds');
            try {
                const result = await classifier.predictClass(activation);
                const labelIndex = parseInt(result.label);
                const label = classes[labelIndex];
                const prob = result.confidences[result.label];

                if (prob > confidenceThreshold) {
                    customLabel.innerText = `${label} (${Math.round(prob * 100)}%)`;
                    customPredBox.classList.remove('hidden');
                    ctx.strokeStyle = '#2dd4bf';
                    ctx.lineWidth = 12;
                    ctx.strokeRect(canvas.width * 0.02, canvas.height * 0.02, canvas.width * 0.96, canvas.height * 0.96);
                    if (labelIndex !== 2) speak(`Viendo: ${label}`);
                } else {
                    customLabel.innerText = 'Escaneando...';
                }
            } catch (e) { }
            img.dispose();
            resized.dispose();
            countNum.innerText = cocoDetectedCount; // Balance count
        }
    } else {
        countNum.innerText = cocoDetectedCount;
        customPredBox.classList.add('hidden');
    }

    requestAnimationFrame(detect);
}

// Training with Resizing
async function addExample(classId) {
    const img = tf.browser.fromPixels(video);
    const resized = tf.image.resizeBilinear(img, [224, 224]);
    const activation = featureExtractor.infer(resized, 'conv_preds');
    classifier.addExample(activation, classId);
    img.dispose();
    resized.dispose();

    classSampleCounts[classId]++;

    // Thumbnail Capture
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

speechToggle.addEventListener('change', (e) => isSpeechEnabled = e.target.checked);
countModeToggle.addEventListener('change', (e) => isCountingMode = e.target.checked);

trainModeBtn.addEventListener('click', () => {
    isTrainingMode = !isTrainingMode;
    trainModeBtn.innerText = isTrainingMode ? 'Desactivar IA Personalizada' : 'Activar IA Personalizada';
    aiModeText.innerText = isTrainingMode ? 'Modo: APRENDIZAJE / CONTEO' : 'Modo: Estándar (Inspección General)';
    // Force standard count if custom off
    if (!isTrainingMode) {
        labelsContainer.innerHTML = '';
        customPredBox.classList.add('hidden');
    }
});

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

toggleCamBtn.addEventListener('click', async () => {
    currentFacingMode = (currentFacingMode === 'user') ? 'environment' : 'user';
    await setupCamera();
});

// Final Initialization
(async () => {
    await setupCamera();
    await loadModels();
    detect();
})();
