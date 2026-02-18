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

// Model Persistence Logic
function saveModel() {
    const dataset = classifier.getClassifierDataset();
    if (Object.keys(dataset).length === 0) {
        status.innerText = 'Sin datos para guardar';
        return;
    }

    const readableDataset = {};
    Object.keys(dataset).forEach((key) => {
        const data = dataset[key].dataSync();
        readableDataset[key] = Array.from(data);
    });

    const json = JSON.stringify({
        dataset: readableDataset,
        counts: classSampleCounts
    });

    localStorage.setItem('vision_it_model_v2', json);
    status.innerText = 'Aprendizaje guardado correctamente';
}

function loadModel() {
    const json = localStorage.getItem('vision_it_model_v2');
    if (!json) return;

    try {
        const data = JSON.parse(json);
        const dataset = data.dataset;
        classSampleCounts = data.counts || [0, 0, 0];

        const tensors = {};
        Object.keys(dataset).forEach((key) => {
            tensors[key] = tf.tensor2d(dataset[key], [dataset[key].length / 1024, 1024]);
        });

        classifier.setClassifierDataset(tensors);
        updateSampleUI();
        status.innerText = 'Sistema listo (Memoria cargada)';
    } catch (err) {
        console.error('Error al cargar modelo:', err);
    }
}

function updateSampleUI() {
    classSampleCounts.forEach((count, i) => {
        sampleCounters[i].innerText = count;
    });
}

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
        status.innerText = 'Sistema Activo';
    } catch (err) {
        status.innerText = 'Fallo en conexión';
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

// DRAW GRID FOR COUNTING MODE
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

    if (isTrainingMode && classifier.getNumClasses() > 0) {
        const name0 = document.getElementById('name-class-0').value || 'Objeto 1';
        const name1 = document.getElementById('name-class-1').value || 'Objeto 2';
        const name2 = document.getElementById('name-class-2').value || 'Fondo';
        const classes = [name0, name1, name2];

        if (isCountingMode) {
            // --- TILED COUNTING MODE ---
            drawGrid(ctx);
            const cols = 4;
            const rows = 4;
            const tw = video.videoWidth / cols;
            const th = video.videoHeight / rows;
            let totalCount = 0;

            for (let y = 0; y < rows; y++) {
                for (let x = 0; x < cols; x++) {
                    // Crop frame
                    const crop = tf.browser.fromPixels(video).slice([y * th, x * tw, 0], [th, tw, 3]);
                    const activation = featureExtractor.infer(crop, 'conv_preds');
                    const result = await classifier.predictClass(activation);

                    if (result.label !== 2 && result.confidences[result.label] > confidenceThreshold) {
                        totalCount++;
                        // Draw marker
                        ctx.fillStyle = 'rgba(45, 212, 191, 0.5)';
                        ctx.beginPath();
                        ctx.arc((x * tw) + tw / 2, (y * th) + th / 2, 10, 0, Math.PI * 2);
                        ctx.fill();
                    }
                    crop.dispose();
                }
            }
            countNum.innerText = totalCount;
            customLabel.innerText = `Total ${name0}/${name1}: ${totalCount}`;
            customPredBox.classList.remove('hidden');
        } else {
            // --- FULL FRAME CLASSIFICATION ---
            const img = tf.browser.fromPixels(video);
            const activation = featureExtractor.infer(img, 'conv_preds');
            try {
                const result = await classifier.predictClass(activation);
                const label = classes[result.label];
                const probability = result.confidences[result.label];

                if (probability > confidenceThreshold) {
                    customLabel.innerText = `${label} (${Math.round(probability * 100)}%)`;
                    customPredBox.classList.remove('hidden');
                    ctx.strokeStyle = '#2dd4bf';
                    ctx.lineWidth = 10;
                    ctx.strokeRect(canvas.width * 0.1, canvas.height * 0.1, canvas.width * 0.8, canvas.height * 0.8);
                    if (label !== name2) speak(`Identificado: ${label}`);
                } else {
                    customLabel.innerText = 'Buscando patrón...';
                }
            } catch (e) { }
            img.dispose();
        }
    } else {
        // --- STANDARD COCO-SSD DETECTION ---
        const predictions = await model.detect(video);
        const filtered = predictions.filter(p => p.score >= confidenceThreshold);
        countNum.innerText = filtered.length;
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

        if (filtered.length > 0) {
            const sumNames = [...new Set(filtered.map(f => f.class))].join(' y ');
            speak(`${filtered.length} objetos en panel: ${sumNames}`);
        }
    }

    requestAnimationFrame(detect);
}

// Training Logic
async function addExample(classId) {
    const img = tf.browser.fromPixels(video);
    const activation = featureExtractor.infer(img, 'conv_preds');
    classifier.addExample(activation, classId);
    img.dispose();

    classSampleCounts[classId]++;
    updateSampleUI();
}

// Event Listeners
[0, 1, 2].forEach(id => {
    const btn = document.getElementById(`add-class-${id}`);
    let interval;

    const start = (e) => {
        if (e.cancelable) e.preventDefault();
        btn.classList.add('recording');
        addExample(id);
        interval = setInterval(() => addExample(id), 150);
    };

    const end = () => {
        clearInterval(interval);
        btn.classList.remove('recording');
        status.innerText = 'Muestras añadidas';
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
    customPredBox.classList.toggle('hidden', !isTrainingMode);
});

saveModelBtn.addEventListener('click', saveModel);

resetTraining.addEventListener('click', () => {
    if (confirm('¿Reiniciar toda la base de datos de aprendizaje?')) {
        classifier.clearAllClasses();
        localStorage.removeItem('vision_it_model_v2');
        classSampleCounts = [0, 0, 0];
        updateSampleUI();
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
