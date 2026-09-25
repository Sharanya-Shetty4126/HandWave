
import {
    FilesetResolver,
    HandLandmarker
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/+esm";


// --------------------------------------------------
// DOM ELEMENTS
// --------------------------------------------------

const video = document.getElementById("video");

const overlayCanvas = document.getElementById("overlayCanvas");
const drawingCanvas = document.getElementById("drawingCanvas");

const overlayCtx = overlayCanvas.getContext("2d");
const drawingCtx = drawingCanvas.getContext("2d");

const startCameraBtn = document.getElementById("startCameraBtn");
const stopCameraBtn = document.getElementById("stopCameraBtn");

const clearBtn = document.getElementById("clearBtn");
const undoBtn = document.getElementById("undoBtn");
const saveBtn = document.getElementById("saveBtn");

const penColor = document.getElementById("penColor");
const penSize = document.getElementById("penSize");
const penSizeValue = document.getElementById("penSizeValue");

const connectionDot = document.getElementById("connectionDot");
const connectionText = document.getElementById("connectionText");

const gestureText = document.getElementById("gestureText");
const handCount = document.getElementById("handCount");
const gestureStatus = document.getElementById("gestureStatus");
const strokeCount = document.getElementById("strokeCount");

const loadingMessage = document.getElementById("loadingMessage");
const statusMessage = document.getElementById("statusMessage");


// --------------------------------------------------
// APPLICATION STATE
// --------------------------------------------------

let handLandmarker = null;
let cameraStream = null;
let animationId = null;

let isCameraRunning = false;
let isModelReady = false;

let lastVideoTime = -1;

let previousPoint = null;
let isDrawing = false;

let currentColor = penColor.value;
let currentPenSize = Number(penSize.value);

let strokes = [];


// --------------------------------------------------
// MEDIAPIPE MODEL CONFIGURATION
// --------------------------------------------------

const WASM_PATH =
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm";

const MODEL_PATH =
    "https://storage.googleapis.com/mediapipe-models/" +
    "hand_landmarker/hand_landmarker/float16/1/" +
    "hand_landmarker.task";


// --------------------------------------------------
// UI HELPERS
// --------------------------------------------------

function setStatus(message) {
    statusMessage.textContent = message;
}

function setConnectionStatus(message, type = "") {
    connectionText.textContent = message;

    connectionDot.classList.remove("active", "error");

    if (type) {
        connectionDot.classList.add(type);
    }
}

function setGestureStatus(message, drawing = false) {
    gestureText.textContent = message;
    gestureStatus.textContent = drawing ? "Drawing" : "Inactive";
}

function updateStrokeCount() {
    strokeCount.textContent = strokes.length;
}


// --------------------------------------------------
// CANVAS SETUP
// --------------------------------------------------

function setupCanvas() {
    /*
     * A fixed internal resolution is used for the first prototype.
     * CSS makes the canvas responsive on the page.
     */
    overlayCanvas.width = 1280;
    overlayCanvas.height = 720;

    drawingCanvas.width = 1280;
    drawingCanvas.height = 720;

    drawingCtx.lineCap = "round";
    drawingCtx.lineJoin = "round";
}

function clearCanvas() {
    drawingCtx.clearRect(
        0,
        0,
        drawingCanvas.width,
        drawingCanvas.height
    );

    strokes = [];
    previousPoint = null;

    updateStrokeCount();

    setStatus("Whiteboard cleared.");
}


// --------------------------------------------------
// LOAD HAND LANDMARKER
// --------------------------------------------------

async function createHandLandmarker() {
    setStatus("Loading hand detection model...");
    setConnectionStatus("Loading model");

    try {
        const vision = await FilesetResolver.forVisionTasks(
            WASM_PATH
        );

        handLandmarker = await HandLandmarker.createFromOptions(
            vision,
            {
                baseOptions: {
                    modelAssetPath: MODEL_PATH
                },

                runningMode: "VIDEO",

                numHands: 1,

                minHandDetectionConfidence: 0.6,

                minHandPresenceConfidence: 0.6,

                minTrackingConfidence: 0.6
            }
        );

        isModelReady = true;

        setStatus(
            "Model ready. Press Start camera."
        );

        setConnectionStatus("Model ready", "active");

    } catch (error) {
        console.error("Model loading error:", error);

        setStatus(
            "Could not load the hand detection model. " +
            "Check your internet connection and browser console."
        );

        setConnectionStatus("Model error", "error");
    }
}


// --------------------------------------------------
// CAMERA MANAGEMENT
// --------------------------------------------------

async function startCamera() {
    if (!isModelReady) {
        setStatus("Please wait for the model to finish loading.");
        return;
    }

    if (isCameraRunning) {
        return;
    }

    if (!navigator.mediaDevices ||
        !navigator.mediaDevices.getUserMedia) {

        setStatus(
            "Camera access is not available in this browser."
        );

        return;
    }

    try {
        setStatus("Requesting camera permission...");

        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: {
                    ideal: 1280
                },

                height: {
                    ideal: 720
                },

                facingMode: "user"
            },

            audio: false
        });

        video.srcObject = cameraStream;

        await video.play();

        isCameraRunning = true;

        lastVideoTime = -1;
        previousPoint = null;

        startCameraBtn.disabled = true;
        stopCameraBtn.disabled = false;

        loadingMessage.style.display = "none";

        setConnectionStatus("Camera running", "active");
        setStatus("Camera started. Raise your index finger to draw.");

        renderLoop();

    } catch (error) {
        console.error("Camera error:", error);

        setConnectionStatus("Camera error", "error");

        if (error.name === "NotAllowedError") {
            setStatus(
                "Camera permission was denied. " +
                "Allow camera access and try again."
            );
        } else {
            setStatus(
                "Could not start the camera. " +
                "Check that your camera is available."
            );
        }
    }
}

function stopCamera() {
    isCameraRunning = false;

    if (animationId !== null) {
        cancelAnimationFrame(animationId);
        animationId = null;
    }

    if (cameraStream) {
        cameraStream.getTracks().forEach(track => {
            track.stop();
        });

        cameraStream = null;
    }

    video.srcObject = null;

    previousPoint = null;
    isDrawing = false;

    overlayCtx.clearRect(
        0,
        0,
        overlayCanvas.width,
        overlayCanvas.height
    );

    handCount.textContent = "0";

    setGestureStatus("Waiting", false);

    startCameraBtn.disabled = false;
    stopCameraBtn.disabled = true;

    loadingMessage.style.display = "flex";

    setConnectionStatus("Camera stopped");
    setStatus("Camera stopped.");
}


// --------------------------------------------------
// HAND LANDMARK HELPERS
// --------------------------------------------------

function isIndexFingerExtended(landmarks) {
    /*
     * This is a simple first-version heuristic.
     *
     * Landmark 8  = index fingertip
     * Landmark 6  = index finger PIP joint
     *
     * In a typical upright palm view, a smaller y-coordinate
     * means a point is higher in the image.
     *
     * This rule is not reliable for every hand orientation.
     */
    const indexTip = landmarks[8];
    const indexPip = landmarks[6];

    return indexTip.y < indexPip.y;
}

function isDrawingGesture(landmarks) {
    return isIndexFingerExtended(landmarks);
}

function getMirroredCanvasPoint(landmark) {
    /*
     * The camera preview is mirrored using CSS.
     * We mirror x here too, so the virtual pen follows
     * the movement naturally.
     */
    return {
        x: (1 - landmark.x) * drawingCanvas.width,
        y: landmark.y * drawingCanvas.height
    };
}


// --------------------------------------------------
// DRAWING FUNCTIONS
// --------------------------------------------------

let currentStroke = null;

function drawLine(from, to) {
    drawingCtx.beginPath();

    drawingCtx.moveTo(from.x, from.y);
    drawingCtx.lineTo(to.x, to.y);

    drawingCtx.strokeStyle = currentColor;
    drawingCtx.lineWidth = currentPenSize;

    drawingCtx.lineCap = "round";
    drawingCtx.lineJoin = "round";

    drawingCtx.stroke();
}

function addStrokePoint(point) {
    if (currentStroke === null) {
        currentStroke = {
            points: [point],
            color: currentColor,
            size: currentPenSize
        };

        previousPoint = point;
        return;
    }

    if (previousPoint !== null) {
        drawLine(previousPoint, point);
    }

    currentStroke.points.push(point);
    previousPoint = point;
}

function stopCurrentStroke() {
    if (currentStroke !== null) {
        if (currentStroke.points.length >= 2) {
            strokes.push(currentStroke);
            updateStrokeCount();
        }

        currentStroke = null;
    }

    previousPoint = null;
    isDrawing = false;
}

// --------------------------------------------------
// LANDMARK VISUALISATION
// --------------------------------------------------

function drawHandLandmarks(landmarks) {
    const width = overlayCanvas.width;
    const height = overlayCanvas.height;

    /*
     * These are the standard 21 hand-landmark connections.
     * Each pair identifies two landmarks that should be joined.
     */
    const connections = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],

        [0, 5],
        [5, 6],
        [6, 7],
        [7, 8],

        [5, 9],
        [9, 10],
        [10, 11],
        [11, 12],

        [9, 13],
        [13, 14],
        [14, 15],
        [15, 16],

        [13, 17],
        [17, 18],
        [18, 19],
        [19, 20],

        [0, 17]
    ];

    overlayCtx.strokeStyle = "rgba(49, 92, 69, 0.85)";
    overlayCtx.lineWidth = 3;

    for (const [startIndex, endIndex] of connections) {
        const start = landmarks[startIndex];
        const end = landmarks[endIndex];

        const x1 = (1 - start.x) * width;
        const y1 = start.y * height;

        const x2 = (1 - end.x) * width;
        const y2 = end.y * height;

        overlayCtx.beginPath();
        overlayCtx.moveTo(x1, y1);
        overlayCtx.lineTo(x2, y2);
        overlayCtx.stroke();
    }

    for (const landmark of landmarks) {
        const x = (1 - landmark.x) * width;
        const y = landmark.y * height;

        overlayCtx.beginPath();
        overlayCtx.arc(x, y, 5, 0, Math.PI * 2);

        overlayCtx.fillStyle = "#315C45";
        overlayCtx.fill();
    }

    /*
     * Highlight the index fingertip.
     * Landmark 8 is the index fingertip.
     */
    const indexTip = landmarks[8];

    const tipX = (1 - indexTip.x) * width;
    const tipY = indexTip.y * height;

    overlayCtx.beginPath();
    overlayCtx.arc(tipX, tipY, 10, 0, Math.PI * 2);

    overlayCtx.strokeStyle = "#D6A343";
    overlayCtx.lineWidth = 4;
    overlayCtx.stroke();
}


// --------------------------------------------------
// PROCESS ONE VIDEO FRAME
// --------------------------------------------------

function processFrame() {
    if (!handLandmarker || !isCameraRunning) {
        return;
    }

    if (video.readyState < 2) {
        return;
    }

    /*
     * Avoid processing the exact same video frame repeatedly.
     */
    if (video.currentTime === lastVideoTime) {
        return;
    }

    lastVideoTime = video.currentTime;

    const results = handLandmarker.detectForVideo(
        video,
        performance.now()
    );

    overlayCtx.clearRect(
        0,
        0,
        overlayCanvas.width,
        overlayCanvas.height
    );

    const hands = results.landmarks || [];

    handCount.textContent = String(hands.length);

    if (hands.length === 0) {
        stopCurrentStroke();
        setGestureStatus("No hand detected", false);
        return;
    }

    const landmarks = hands[0];

    drawHandLandmarks(landmarks);

    const drawingGesture = isDrawingGesture(landmarks);

    if (drawingGesture) {
        const point = getMirroredCanvasPoint(landmarks[8]);

        if (!isDrawing) {
            isDrawing = true;
            previousPoint = point;
        } else {
            addStrokePoint(point);
        }

        setGestureStatus("Drawing", true);

    } else {
        stopCurrentStroke();
        setGestureStatus("Pen lifted", false);
    }
}


// --------------------------------------------------
// ANIMATION LOOP
// --------------------------------------------------

function renderLoop() {
    if (!isCameraRunning) {
        return;
    }

    processFrame();

    animationId = requestAnimationFrame(renderLoop);
}


// --------------------------------------------------
// SAVE AND UNDO
// --------------------------------------------------

function redrawAllStrokes() {
    drawingCtx.clearRect(
        0,
        0,
        drawingCanvas.width,
        drawingCanvas.height
    );

    for (const stroke of strokes) {
        if (stroke.points.length < 2) {
            continue;
        }

        drawingCtx.beginPath();

        drawingCtx.moveTo(
            stroke.points[0].x,
            stroke.points[0].y
        );

        for (let i = 1; i < stroke.points.length; i++) {
            drawingCtx.lineTo(
                stroke.points[i].x,
                stroke.points[i].y
            );
        }

        drawingCtx.strokeStyle = stroke.color;
        drawingCtx.lineWidth = stroke.size;
        drawingCtx.lineCap = "round";
        drawingCtx.lineJoin = "round";

        drawingCtx.stroke();
    }
}

function saveWhiteboard() {
    const exportCanvas = document.createElement("canvas");

    exportCanvas.width = drawingCanvas.width;
    exportCanvas.height = drawingCanvas.height;

    const exportCtx = exportCanvas.getContext("2d");

    exportCtx.fillStyle = "#ffffff";

    exportCtx.fillRect(
        0,
        0,
        exportCanvas.width,
        exportCanvas.height
    );

    exportCtx.drawImage(
        drawingCanvas,
        0,
        0
    );

    const link = document.createElement("a");

    link.download = "PhantomPen-whiteboard.png";
    link.href = exportCanvas.toDataURL("image/png");

    link.click();

    setStatus("Whiteboard exported as PNG.");
}


// --------------------------------------------------
// TOOLBAR EVENT LISTENERS
// --------------------------------------------------

penColor.addEventListener("input", () => {
    currentColor = penColor.value;
});

penSize.addEventListener("input", () => {
    currentPenSize = Number(penSize.value);
    penSizeValue.textContent = String(currentPenSize);
});

clearBtn.addEventListener("click", () => {
    clearCanvas();
});

undoBtn.addEventListener("click", () => {
    if (strokes.length === 0) {
        setStatus("There are no strokes to undo.");
        return;
    }

    strokes.pop();

    redrawAllStrokes();
    updateStrokeCount();

    setStatus("Last stroke removed.");
});

saveBtn.addEventListener("click", () => {
    saveWhiteboard();
});

startCameraBtn.addEventListener("click", () => {
    startCamera();
});

stopCameraBtn.addEventListener("click", () => {
    stopCamera();
});


// --------------------------------------------------
// INITIALISE APPLICATION
// --------------------------------------------------

setupCanvas();

updateStrokeCount();

createHandLandmarker();