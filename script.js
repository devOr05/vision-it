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
const trainUI = document.getElementById('train-ui');
const resetTraining = document.getElementById('reset-training');

// Collection UI
const collectModeBtn = document.getElementById('collect-mode-btn');
const collectionUI = document.getElementById('collection-ui');
const objectLabel = document.getElementById('object-label');
const snapRecord = document.getElementById('snap-record');
const sampleCount = document.getElementById('sample-count');

let model;
let featureExtractor; // MobileNet
let classifier; // KNN Classifier
let currentFacingMode = 'environment';
let confidenceThreshold = 0.6;
let isSpeechEnabled = false;
let isCollectionMode = false;
let isTrainingMode = false;
let samples = [];
let lastSpoken = "";
let lastSpokenTime = 0;

async function setupCamera() {
    status.innerText = 'Inuciando cámara...';
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
    }
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: currentFacingMode },
            audio: false
        });
        video.srcObject = stream;
        return new Promise((resolve) => {
            video.onloadedmetadata = () => resolve(video);
        });
    } catch (err) {
        status.innerText = 'Error de cámara';
        console.error(err);
    }
}

async function loadModels() {
    status.innerText = 'Cargando IAs...';
    // Load COCO-SSD for standard detection
    model = await cocoSsd.load();
    // Load MobileNet for feature extraction
    featureExtractor = await mobilenet.load({ version: 2, alpha: 1.0 });
    // Initialize KNN Classifier
    classifier = knnClassifier.create();

    status.innerText = 'Visión IT Activa';
}

function speak(text) {
    if (!isSpeechEnabled) return;
    const now = Date.now();
    if (now - lastSpokenTime < 3000 && text === lastSpoken) return;

    const u = new SpeechSynthesisUtterance(text);
    u.lang = 'es-ES';
    window.speechSynthesis.speak(u);
    lastSpoken = text;
    lastSpokenTime = now;
}

async function detect() {
    if (!model || isCollectionMode) {
        if (!isCollectionMode) requestAnimationFrame(detect);
        return;
    }

    const ctx = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (isTrainingMode) {
        // Transfer Learning / Custom Classification
        const img = tf.browser.fromPixels(video);
        const features = featureExtractor.infer(img, 'conv_preds');

        if (classifier.getNumClasses() > 0) {
            const res = await classifier.predictClass(features);
            const classes = ['Objeto 1', 'Objeto 2', 'Fondo'];
            const label = classes[res.label];
            const prob = res.confidences[res.label];

            if (prob > 0.5) {
                countNum.innerText = "1";
                labelsContainer.innerHTML = `<span class="label-tag" style="background:var(--accent); color:#000">${label} (${Math.round(prob * 100)}%)</span>`;

                // Draw a simple center box for custom objects
                ctx.strokeStyle = 'var(--accent)';
                ctx.lineWidth = 4;
                ctx.strokeRect(canvas.width * 0.25, canvas.height * 0.25, canvas.width * 0.5, canvas.height * 0.5);

                speak(`Detectado ${label}`);
            } else {
                countNum.innerText = "0";
                labelsContainer.innerHTML = '';
            }
        }
        img.dispose();
    } else {
        // Standard COCO-SSD Detection
        const predictions = await model.detect(video);
        const filtered = predictions.filter(p => p.score >= confidenceThreshold);

        countNum.innerText = filtered.length;
        labelsContainer.innerHTML = '';

        filtered.forEach(prediction => {
            ctx.strokeStyle = '#38bdf8';
            ctx.lineWidth = 4;
            ctx.strokeRect(...prediction.bbox);

            ctx.fillStyle = '#38bdf8';
            const textHeight = 25;
            ctx.fillRect(prediction.bbox[0], prediction.bbox[1] - textHeight, ctx.measureText(prediction.class).width + 10, textHeight);

            ctx.fillStyle = '#000';
            ctx.font = 'bold 16px Outfit';
            ctx.fillText(prediction.class, prediction.bbox[0] + 5, prediction.bbox[1] - 7);

            const tag = document.createElement('span');
            tag.className = 'label-tag';
            tag.innerText = `${prediction.class} (${Math.round(prediction.score * 100)}%)`;
            labelsContainer.appendChild(tag);
        });

        if (filtered.length > 0) {
            const names = filtered.map(f => f.class).join(', ');
            speak(`Veo ${names}`);
        }
    }

    requestAnimationFrame(detect);
}

// UI Event Listeners
openSettings.addEventListener('click', () => settingsDrawer.classList.add('active'));
closeSettings.addEventListener('click', () => settingsDrawer.classList.remove('active'));

confSlider.addEventListener('input', (e) => {
    confidenceThreshold = e.target.value / 100;
    confVal.innerText = e.target.value;
});

speechToggle.addEventListener('change', (e) => isSpeechEnabled = e.target.checked);

captureBtn.addEventListener('click', () => {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    const tCtx = tempCanvas.getContext('2d');
    tCtx.drawImage(video, 0, 0);
    tCtx.drawImage(canvas, 0, 0);
    const link = document.createElement('a');
    link.download = `vision-it-${Date.now()}.png`;
    link.href = tempCanvas.toDataURL();
    link.click();
});

// Training Logic
trainModeBtn.addEventListener('click', () => {
    isTrainingMode = !isTrainingMode;
    trainModeBtn.innerText = isTrainingMode ? 'Modo Entrenamiento: ON' : 'Modo Entrenamiento: OFF';
    trainModeBtn.style.borderColor = isTrainingMode ? 'var(--accent)' : 'var(--primary)';
    trainUI.classList.toggle('hidden');
});

[0, 1, 2].forEach(id => {
    const btn = document.getElementById(`add-class-${id}`);
    btn.addEventListener('mousedown', () => addExample(id));
    btn.addEventListener('touchstart', (e) => { e.preventDefault(); addExample(id); });
});

async function addExample(classId) {
    if (!isTrainingMode) return;
    const img = tf.browser.fromPixels(video);
    const features = featureExtractor.infer(img, 'conv_preds');
    classifier.addExample(features, classId);
    img.dispose();
    status.innerText = `Aprendiendo clase ${classId}...`;
    setTimeout(() => status.innerText = 'Entrenamiento guardado', 500);
}

resetTraining.addEventListener('click', () => {
    classifier.clearAllClasses();
    status.innerText = 'Aprendizaje borrado';
});

// Collection Logic (from previous phase)
collectModeBtn.addEventListener('click', () => {
    isCollectionMode = !isCollectionMode;
    collectModeBtn.innerText = isCollectionMode ? 'Modo Recolección: ON' : 'Modo Recolección: OFF';
    collectionUI.classList.toggle('hidden');
    if (!isCollectionMode) detect();
});

snapRecord.addEventListener('click', () => {
    samples.push({ label: objectLabel.value, time: Date.now() });
    sampleCount.innerText = `${samples.length} muestras`;
    snapRecord.style.background = 'white';
    setTimeout(() => snapRecord.style.background = 'var(--accent)', 100);
});

toggleCamBtn.addEventListener('click', async () => {
    currentFacingMode = (currentFacingMode === 'user') ? 'environment' : 'user';
    await setupCamera();
});

async function init() {
    await setupCamera();
    await loadModels();
    detect();
}

init();
