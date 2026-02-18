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
const saveModelBtn = document.getElementById('save-model-btn');
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
    status.innerText = 'Conectando cámara...';
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
        shape: dataset[0].shape // Capture first for shape reference
    });

    localStorage.setItem('vision_it_model', json);
    status.innerText = 'Modelo guardado localmente';
}

function loadModel() {
    const json = localStorage.getItem('vision_it_model');
    if (!json) return;

    try {
        const data = JSON.parse(json);
        const dataset = data.dataset;
        const tensors = {};

        Object.keys(dataset).forEach((key) => {
            tensors[key] = tf.tensor2d(dataset[key], [dataset[key].length / 1024, 1024]);
        });

        classifier.setClassifierDataset(tensors);
        status.innerText = 'Modelo previo cargado';
    } catch (err) {
        console.error('Error al cargar modelo:', err);
    }
}

// Model Loading
async function loadModels() {
    status.innerText = 'Conectando...';
    try {
        const [cocoRes, mobRes] = await Promise.all([
            cocoSsd.load(),
            mobilenet.load({ version: 2, alpha: 1.0 })
        ]);

        model = cocoRes;
        featureExtractor = mobRes;
        classifier = knnClassifier.create();

        loadModel(); // Load saved data if exists
        status.innerText = 'Sistema Listo';
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
        const img = tf.browser.fromPixels(video);
        const activation = featureExtractor.infer(img, 'conv_preds');

        try {
            const result = await classifier.predictClass(activation);

            const name0 = document.getElementById('name-class-0').value || 'Objeto 1';
            const name1 = document.getElementById('name-class-1').value || 'Objeto 2';
            const name2 = document.getElementById('name-class-2').value || 'Fondo';
            const classes = [name0, name1, name2];

            const label = classes[result.label];
            const probability = result.confidences[result.label];

            if (probability > 0.6) {
                customLabel.innerText = `${label} (${Math.round(probability * 100)}%)`;
                customPredBox.classList.remove('hidden');

                ctx.strokeStyle = '#2dd4bf';
                ctx.lineWidth = 10;
                ctx.strokeRect(canvas.width * 0.1, canvas.height * 0.1, canvas.width * 0.8, canvas.height * 0.8);

                if (label !== name2) {
                    speak(`Identificado: ${label}`);
                }
            } else {
                customLabel.innerText = 'Analizando...';
            }
        } catch (e) {
            console.warn("Esperando datos de entrenamiento...");
        }
        img.dispose();
    } else {
        const predictions = await model.detect(video);
        const filtered = predictions.filter(p => p.score >= confidenceThreshold);

        countNum.innerText = filtered.length;
        labelsContainer.innerHTML = '';

        filtered.forEach(prediction => {
            const [x, y, width, height] = prediction.bbox;
            ctx.strokeStyle = '#38bdf8';
            ctx.lineWidth = 4;
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
    status.innerText = 'Entrenando...';
    const img = tf.browser.fromPixels(video);
    const activation = featureExtractor.infer(img, 'conv_preds');
    classifier.addExample(activation, classId);
    img.dispose();
}

// Event Listeners
[0, 1, 2].forEach(id => {
    const btn = document.getElementById(`add-class-${id}`);
    let interval;

    const start = (e) => {
        if (e.cancelable) e.preventDefault();
        btn.classList.add('recording');
        addExample(id);
        interval = setInterval(() => addExample(id), 100);
    };

    const end = () => {
        clearInterval(interval);
        btn.classList.remove('recording');
        status.innerText = 'Entrenamiento guardado en sesión';
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

trainModeBtn.addEventListener('click', () => {
    isTrainingMode = !isTrainingMode;
    trainModeBtn.innerText = isTrainingMode ? 'Desactivar IA Personalizada' : 'Activar IA Personalizada';
    aiModeText.innerText = isTrainingMode ? 'Modo: APRENDIZAJE ACTIVO' : 'Modo: Estándar (Inspección General)';
    customPredBox.classList.toggle('hidden', !isTrainingMode);
});

saveModelBtn.addEventListener('click', saveModel);

resetTraining.addEventListener('click', () => {
    classifier.clearAllClasses();
    localStorage.removeItem('vision_it_model');
    status.innerText = 'Base de datos reiniciada';
});

captureBtn.addEventListener('click', () => {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    const tCtx = tempCanvas.getContext('2d');
    tCtx.drawImage(video, 0, 0);
    tCtx.drawImage(canvas, 0, 0);

    const link = document.createElement('a');
    link.download = `inspeccion-it-${Date.now()}.png`;
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
