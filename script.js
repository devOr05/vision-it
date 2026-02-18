const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const status = document.getElementById('status');
const countNum = document.getElementById('count-num');
const labelsContainer = document.getElementById('labels-container');
const toggleCamBtn = document.getElementById('toggle-camera');

let model;
let currentFacingMode = 'environment'; // Default to back camera

async function setupCamera() {
    status.innerText = 'Iniciando cámara...';

    // Stop any current stream
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
            video.onloadedmetadata = () => {
                resolve(video);
            };
        });
    } catch (err) {
        console.error('Error al acceder a la cámara:', err);
        status.innerText = 'Error: No se pudo acceder a la cámara.';
    }
}

async function loadModel() {
    status.innerText = 'Cargando modelo de IA...';
    model = await cocoSsd.load();
    status.innerText = '¡Listo para detectar!';
}

async function detect() {
    if (!model) return;

    const predictions = await model.detect(video);

    // Clear canvas
    const ctx = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Update Counter
    countNum.innerText = predictions.length;
    labelsContainer.innerHTML = '';

    predictions.forEach(prediction => {
        // Draw bounding box
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 4;
        ctx.strokeRect(...prediction.bbox);

        // Draw label background
        ctx.fillStyle = '#38bdf8';
        const textWidth = ctx.measureText(prediction.class).width;
        ctx.fillRect(prediction.bbox[0], prediction.bbox[1] - 25, textWidth + 10, 25);

        // Draw text
        ctx.fillStyle = '#000';
        ctx.font = 'bold 16px Outfit';
        ctx.fillText(prediction.class, prediction.bbox[0] + 5, prediction.bbox[1] - 7);

        // Add to labels footer
        const tag = document.createElement('span');
        tag.className = 'label-tag';
        tag.innerText = prediction.class;
        labelsContainer.appendChild(tag);
    });

    requestAnimationFrame(detect);
}

toggleCamBtn.addEventListener('click', async () => {
    currentFacingMode = (currentFacingMode === 'user') ? 'environment' : 'user';
    await setupCamera();
});

async function init() {
    await setupCamera();
    await loadModel();
    detect();
}

init();
