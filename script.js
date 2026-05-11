import {
  PoseLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest";

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const mode = document.body.dataset.mode || "art";

const COLS = 60;
const ROWS = 40;
const MAX_INTENSITY = 1;
const INTENSITY_DECAY = 0.5; // controlla quanto scuriscono le celle ad ogni frame
const INTENSITY_BOOST = 0.9; // controlla quanto si accendono le celle quando vengono attivate
const VIDEO_FEED_IS_MIRRORED = true; // set to true if your camera preview is mirrored
const MIRROR_X = VIDEO_FEED_IS_MIRRORED;
const POSE_MODEL_NAME = "pose_landmarker_full";
const POSE_MODEL_URL = `https://storage.googleapis.com/mediapipe-models/pose_landmarker/${POSE_MODEL_NAME}/float16/latest/${POSE_MODEL_NAME}.task`;
const USE_VIDEO_CONTOURS = true;
const EDGE_BRIGHTNESS_THRESHOLD = 40;

const cells = Array.from({ length: ROWS }, () => Array(COLS).fill(0));
const analysisCanvas = document.createElement("canvas");
analysisCanvas.width = COLS;
analysisCanvas.height = ROWS;
const analysisCtx = analysisCanvas.getContext("2d");

let poseLandmarker = null;
let lastVideoTime = -1;
let lastPoseResult = null;
let statusText = "Loading script...";

const POSE_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 7],
  [0, 4], [4, 5], [5, 6], [6, 8],
  [9, 10],
  [11, 12], [11, 13], [13, 15],
  [12, 14], [14, 16],
  [15, 17], [16, 18],
  [15, 19], [19, 21],
  [16, 20], [20, 22],
  [11, 23], [12, 24],
  [23, 24], [23, 25], [24, 26],
  [25, 27], [26, 28],
  [27, 29], [28, 30],
  [29, 31], [30, 32]
];

function resizeCanvas() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
}

function drawGrid() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const squareSize = Math.floor(
    Math.min(canvas.width / COLS, canvas.height / ROWS)
  );

  const gridW = squareSize * COLS;
  const gridH = squareSize * ROWS;

  const offsetX = (canvas.width - gridW) / 2;
  const offsetY = (canvas.height - gridH) / 2;

  const gap = 2;

  for (let y = 0; y < ROWS; y++) {
    for (let x = 0; x < COLS; x++) {
      const intensity = cells[y][x];

      ctx.fillStyle =
        intensity > 0.01
          ? `rgba(255, 220, 120, ${Math.max(0.4, intensity)})`
          : "rgba(55, 55, 55, 0.5)";

      ctx.fillRect(
        offsetX + x * squareSize,
        offsetY + y * squareSize,
        squareSize - gap,
        squareSize - gap
      );
    }
  }

  ctx.fillStyle = "white";
  ctx.font = "18px Arial";
  ctx.fillText(statusText, 20, 30);
}

function fadeCells() {
  for (let y = 0; y < ROWS; y++) {
    for (let x = 0; x < COLS; x++) {
      cells[y][x] *= INTENSITY_DECAY;
    }
  }
}

function activateCell(normX, normY) {
  const x = Math.floor(normX * COLS);
  const y = Math.floor(normY * ROWS);

  if (x >= 0 && x < COLS && y >= 0 && y < ROWS) {
    cells[y][x] = Math.min(MAX_INTENSITY, cells[y][x] + INTENSITY_BOOST);
  }
}

function updateCellsFromVideoContours() {
  if (!video || video.readyState < 2) {
    return;
  }

  analysisCtx.save();
  if (VIDEO_FEED_IS_MIRRORED) {
    analysisCtx.setTransform(-1, 0, 0, 1, COLS, 0);
  } else {
    analysisCtx.setTransform(1, 0, 0, 1, 0, 0);
  }
  analysisCtx.drawImage(video, 0, 0, COLS, ROWS);
  analysisCtx.restore();

  const imageData = analysisCtx.getImageData(0, 0, COLS, ROWS);
  const data = imageData.data;

  const getLuma = (x, y) => {
    x = Math.max(0, Math.min(COLS - 1, x));
    y = Math.max(0, Math.min(ROWS - 1, y));
    const index = (y * COLS + x) * 4;
    const r = data[index];
    const g = data[index + 1];
    const b = data[index + 2];
    return 0.299 * r + 0.587 * g + 0.114 * b;
  };

  for (let y = 0; y < ROWS; y++) {
    for (let x = 0; x < COLS; x++) {
      const left = getLuma(x - 1, y);
      const right = getLuma(x + 1, y);
      const top = getLuma(x, y - 1);
      const bottom = getLuma(x, y + 1);
      const dx = right - left;
      const dy = bottom - top;
      const edge = Math.min(1, Math.sqrt(dx * dx + dy * dy) / EDGE_BRIGHTNESS_THRESHOLD);
      const intensity = Math.max(cells[y][x], edge);
      cells[y][x] = Math.min(MAX_INTENSITY, intensity);
    }
  }
}

function drawDebug() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const videoRatio = video.videoWidth / video.videoHeight;
  const canvasRatio = canvas.width / canvas.height;
  let drawWidth, drawHeight;

  if (videoRatio > canvasRatio) {
    drawWidth = canvas.width;
    drawHeight = drawWidth / videoRatio;
  } else {
    drawHeight = canvas.height;
    drawWidth = drawHeight * videoRatio;
  }

  const offsetX = (canvas.width - drawWidth) / 2;
  const offsetY = (canvas.height - drawHeight) / 2;

  ctx.save();
  if (VIDEO_FEED_IS_MIRRORED) {
    ctx.translate(offsetX + drawWidth, offsetY);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, drawWidth, drawHeight);
  } else {
    ctx.drawImage(video, offsetX, offsetY, drawWidth, drawHeight);
  }
  ctx.restore();

  if (lastPoseResult && lastPoseResult.landmarks && lastPoseResult.landmarks.length) {
    const landmarks = lastPoseResult.landmarks[0];
    ctx.strokeStyle = "rgba(0, 255, 255, 0.8)";
    ctx.fillStyle = "rgba(255, 0, 0, 0.9)";
    ctx.lineWidth = 2;

    for (const [a, b] of POSE_CONNECTIONS) {
      const pointA = landmarks[a];
      const pointB = landmarks[b];
      if (!pointA || !pointB) continue;
      const xA = offsetX + pointA.x * drawWidth;
      const yA = offsetY + pointA.y * drawHeight;
      const xB = offsetX + pointB.x * drawWidth;
      const yB = offsetY + pointB.y * drawHeight;
      ctx.beginPath();
      ctx.moveTo(xA, yA);
      ctx.lineTo(xB, yB);
      ctx.stroke();
    }

    for (const point of landmarks) {
      const x = offsetX + point.x * drawWidth;
      const y = offsetY + point.y * drawHeight;
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  ctx.fillStyle = "white";
  ctx.font = "18px Arial";
  ctx.fillText(statusText, 20, 30);
}

async function setupCamera() {
  statusText = "Requesting webcam...";
  drawGrid();

  const stream = await navigator.mediaDevices.getUserMedia({
    video: true
  });

  video.srcObject = stream;

  await new Promise(resolve => {
    video.onloadedmetadata = () => resolve();
  });

  await video.play();

  statusText = "Webcam ready";
}

async function setupPose() {
  statusText = "Loading MediaPipe...";
  drawGrid();

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );

  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: POSE_MODEL_URL
    },
    runningMode: "VIDEO",
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5
  });

  statusText = "MediaPipe ready — move in front of camera";
}

function processPose(result) {
  lastPoseResult = result;

  if (!result || !result.landmarks || result.landmarks.length === 0) {
    statusText = "MediaPipe ready — no body detected";
    return;
  }

  statusText = "Body detected";

  const landmarks = result.landmarks[0];

  for (const point of landmarks) {
    const x = MIRROR_X ? 1 - point.x : point.x;
    activateCell(x, point.y);
  }

  // attiva anche il centro come punto di riferimento visivo
  activateCell(0.5, 0.5);
}

function loop() {
  fadeCells();

  if (poseLandmarker && video.readyState >= 2) {
    if (video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;

      const result = poseLandmarker.detectForVideo(
        video,
        performance.now()
      );

      processPose(result);
    }
  }

  if (mode === "art") {
    if (USE_VIDEO_CONTOURS) {
      updateCellsFromVideoContours();
    }
    drawGrid();
  } else if (mode === "debug") {
    drawDebug();
  }

  requestAnimationFrame(loop);
}

async function main() {
  resizeCanvas();
  window.addEventListener("resize", resizeCanvas);

  drawGrid();

  try {
    await setupCamera();
    await setupPose();
    loop();
  } catch (error) {
    statusText = "ERROR: " + error.message;
    drawGrid();
    console.error(error);
  }
}

main();