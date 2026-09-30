import {
    FilesetResolver,
    HandLandmarker
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/+esm";


// ==================================================
// 1. DOM ELEMENTS
// ==================================================

const video = document.getElementById("video");
const overlayCanvas = document.getElementById("overlayCanvas");
const drawingCanvas = document.getElementById("drawingCanvas");

const overlayCtx = overlayCanvas.getContext("2d");
const drawingCtx = drawingCanvas.getContext("2d");

const boardEl = document.getElementById("boardWrapper");

const startCameraBtn = document.getElementById("startCameraBtn");
const stopCameraBtn = document.getElementById("stopCameraBtn");

const clearBtn = document.getElementById("clearBtn");
const undoBtn = document.getElementById("undoBtn");
const redoBtn = document.getElementById("redoBtn");
const saveBtn = document.getElementById("saveBtn");
const fullscreenBtn = document.getElementById("fullscreenBtn");

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


const capturePenBtn = document.getElementById("capturePenBtn");
const captureOverlay = document.getElementById("captureOverlay");
const captureBox = document.getElementById("captureBox");

// Pen-mode UI
const modeHandBtn = document.getElementById("modeHandBtn");
const modePenBtn = document.getElementById("modePenBtn");
const penToolbar = document.getElementById("penToolbar");
const calibrateBtn = document.getElementById("calibrateBtn");
const penDownBtn = document.getElementById("penDownBtn");
const penSwatch = document.getElementById("penSwatch");
const calibrationHint = document.getElementById("calibrationHint");
const gestureHint = document.getElementById("gestureHint");


// ==================================================
// 2. CONSTANTS
// ==================================================

const WASM_PATH =
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm";

const MODEL_PATH =
    "https://storage.googleapis.com/mediapipe-models/" +
    "hand_landmarker/hand_landmarker/float16/1/" +
    "hand_landmarker.task";


// --- Gesture vocabulary ---

const GESTURE = {
    DRAW: "draw",
    ERASE: "erase",
    PAUSE: "pause",
    NONE: "none"
};

const GESTURE_LABEL = {
    [GESTURE.DRAW]: "☝️ Draw",
    [GESTURE.ERASE]: "🖐️ Erase",
    [GESTURE.PAUSE]: "✊ Pause",
    [GESTURE.NONE]: "Not recognised"
};

const GESTURE_HOLD_FRAMES = 3;


// --- Drawing constants ---

const SMOOTHING_FACTOR = 0.35;
const MIN_MOVEMENT = 2;

const ERASER_MIN_SIZE = 30;
const ERASER_SIZE_MULTIPLIER = 3;


// --- Tool configuration ---
//
// Only tools that PAINT pixels live here. The eraser
// removes pixels (destination-out), so it has its own
// code path.

const tools = {
    pen: { name: "Pen", opacity: 1, compositeOperation: "source-over" },
    marker: { name: "Marker", opacity: 1, compositeOperation: "source-over" },
    highlighter: { name: "Highlighter", opacity: 0.3, compositeOperation: "source-over" }
};


// ==================================================
// 3. APPLICATION STATE
// ==================================================

let handLandmarker = null;
let cameraStream = null;
let animationId = null;

let isCameraRunning = false;
let isModelReady = false;
let lastVideoTime = -1;

// Gesture state
let activeGesture = GESTURE.NONE;
let candidateGesture = GESTURE.NONE;
let candidateFrames = 0;

// Drawing state
let previousPoint = null;
let smoothedPoint = null;

// Action history (see §14 for the model)
let actions = [];
let redoStack = [];
let currentAction = null;

// Tool settings
let currentColor = penColor.value;
let currentPenSize = Number(penSize.value);


// ==================================================
// 4. UI HELPERS
// ==================================================

function setStatus(message) {
    statusMessage.textContent = message;
}

function setConnectionStatus(message, type = "") {
    connectionText.textContent = message;
    connectionDot.classList.remove("active", "error");
    if (type) connectionDot.classList.add(type);
}

function setGestureStatus(message, activity = "Inactive") {
    gestureText.textContent = message;
    gestureStatus.textContent = activity;
}

function updateHistoryUI() {
    strokeCount.textContent = String(actions.length);
    undoBtn.disabled = actions.length === 0;
    redoBtn.disabled = redoStack.length === 0;
}


// ==================================================
// 5. GEOMETRY + SMOOTHING
// ==================================================

function distance(a, b) {
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    return Math.sqrt(dx * dx + dy * dy);
}


// new_smoothed = old_smoothed + (raw - old_smoothed) * α
//
// α close to 1 → follows input tightly but jitters.
// α close to 0 → smooth but lags.
// 0.35 is a good middle ground at 30–60 fps.

function smoothPoint(point) {

    if (smoothedPoint === null) {
        smoothedPoint = { x: point.x, y: point.y };
        return { x: smoothedPoint.x, y: smoothedPoint.y };
    }

    smoothedPoint.x += (point.x - smoothedPoint.x) * SMOOTHING_FACTOR;
    smoothedPoint.y += (point.y - smoothedPoint.y) * SMOOTHING_FACTOR;

    return { x: smoothedPoint.x, y: smoothedPoint.y };
}


// ==================================================
// 6. CANVAS SETUP
// ==================================================

function setupCanvas() {

    /*
     * Stable internal coordinate system (1280×720).
     * CSS controls the visual size.
     *
     * We deliberately never touch width/height again,
     * because assigning those clears the canvas.
     */

    overlayCanvas.width = 1280;
    overlayCanvas.height = 720;

    drawingCanvas.width = 1280;
    drawingCanvas.height = 720;

    drawingCtx.lineCap = "round";
    drawingCtx.lineJoin = "round";
}

function resizeCanvasesToVideo() {
    const width = video.videoWidth;
    const height = video.videoHeight;
    if (!width || !height) return;
    boardEl.style.aspectRatio = `${width} / ${height}`;
}


// ==================================================
// 7. FULLSCREEN
// ==================================================

async function toggleFullscreen() {

    try {

        if (!document.fullscreenElement) {
            await boardEl.requestFullscreen();
            boardEl.classList.add("fullscreen");
            fullscreenBtn.textContent = "Exit fullscreen";
        } else {
            await document.exitFullscreen();
            boardEl.classList.remove("fullscreen");
            fullscreenBtn.textContent = "Fullscreen";
        }

    } catch (error) {
        console.error("Fullscreen error:", error);
        setStatus("Fullscreen mode could not be activated.");
    }
}

document.addEventListener("fullscreenchange", () => {

    if (document.fullscreenElement !== null) {
        fullscreenBtn.textContent = "Exit fullscreen";
    } else {
        fullscreenBtn.textContent = "Fullscreen";
        boardEl.classList.remove("fullscreen");
    }
});


// ==================================================
// 8. CLEAR
// ==================================================

function clearCanvas() {

    drawingCtx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);

    actions = [];
    redoStack = [];
    currentAction = null;

    previousPoint = null;
    smoothedPoint = null;

    updateHistoryUI();
    setStatus("Whiteboard cleared.");
}


// ==================================================
// 9. MODEL LOADING
// ==================================================

async function createHandLandmarker() {

    setStatus("Loading hand detection model...");
    setConnectionStatus("Loading model");

    try {

        const vision = await FilesetResolver.forVisionTasks(WASM_PATH);

        handLandmarker = await HandLandmarker.createFromOptions(
            vision,
            {
                baseOptions: { modelAssetPath: MODEL_PATH },
                runningMode: "VIDEO",
                numHands: 1,
                minHandDetectionConfidence: 0.6,
                minHandPresenceConfidence: 0.6,
                minTrackingConfidence: 0.6
            }
        );

        isModelReady = true;

        setStatus("Model ready. Press Start camera.");
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


// ==================================================
// 10. CAMERA
// ==================================================

async function startCamera() {

    if (!isModelReady) {
        setStatus("Please wait for the model to finish loading.");
        return;
    }

    if (isCameraRunning) return;

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setStatus("Camera access is not available in this browser.");
        return;
    }

    try {

        setStatus("Requesting camera permission...");

        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: "user"
            },
            audio: false
        });

        video.srcObject = cameraStream;
        await video.play();

        resizeCanvasesToVideo();

        isCameraRunning = true;
        lastVideoTime = -1;

        previousPoint = null;
        smoothedPoint = null;

        resetGestureState();

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
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }

    video.srcObject = null;

    stopCurrentAction();

    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    handCount.textContent = "0";

    resetGestureState();
    setGestureStatus("Waiting", "Inactive");
    endCalibration();

    startCameraBtn.disabled = false;
    stopCameraBtn.disabled = true;

    loadingMessage.style.display = "flex";

    setConnectionStatus("Camera stopped");
    setStatus("Camera stopped.");
}


// ==================================================
// 11. GESTURE CLASSIFIER (shared by both modes)
// ==================================================
//
// A finger is "extended" when its tip is farther from
// the wrist than its PIP joint. This is rotation-
// invariant, unlike the naive tip.y < pip.y test.

function isFingerExtended(landmarks, tipIndex, pipIndex) {

    const wrist = landmarks[0];
    const tip = landmarks[tipIndex];
    const pip = landmarks[pipIndex];

    const tipDistance = distance(wrist, tip);
    const pipDistance = distance(wrist, pip);

    return tipDistance > pipDistance * 1.05;
}

function getFingerStates(landmarks) {
    return {
        index:  isFingerExtended(landmarks, 8, 6),
        middle: isFingerExtended(landmarks, 12, 10),
        ring:   isFingerExtended(landmarks, 16, 14),
        pinky:  isFingerExtended(landmarks, 20, 18)
    };
}

function classifyGesture(landmarks) {

    const fingers = getFingerStates(landmarks);

    const extendedCount = [
        fingers.index,
        fingers.middle,
        fingers.ring,
        fingers.pinky
    ].filter(Boolean).length;

    // ☝️ index only → DRAW

    if (
        fingers.index &&
        !fingers.middle &&
        !fingers.ring &&
        !fingers.pinky
    ) {
        return GESTURE.DRAW;
    }

    // 🖐️ everything extended → ERASE

    if (extendedCount === 4) {
        return GESTURE.ERASE;
    }

    // ✊ nothing extended → PAUSE

    if (extendedCount === 0) {
        return GESTURE.PAUSE;
    }

    return GESTURE.NONE;
}


// Raw gestures flicker frame-to-frame. Accept a new
// gesture only after it persists for GESTURE_HOLD_FRAMES.

function stabiliseGesture(rawGesture) {

    if (rawGesture === candidateGesture) {
        candidateFrames += 1;
    } else {
        candidateGesture = rawGesture;
        candidateFrames = 1;
    }

    if (
        candidateFrames >= GESTURE_HOLD_FRAMES &&
        candidateGesture !== activeGesture
    ) {
        stopCurrentAction();
        activeGesture = candidateGesture;
        setStatus(`Gesture: ${GESTURE_LABEL[activeGesture]}`);
    }

    return activeGesture;
}

function resetGestureState() {
    candidateGesture = GESTURE.NONE;
    candidateFrames = 0;
    activeGesture = GESTURE.NONE;
}


// ==================================================
// 12. COORDINATE MAPPING
// ==================================================
//
// <video> is CSS-mirrored with scaleX(-1) so it feels
// natural. Mirror landmark X too so drawn ink lands
// under the fingertip.

function getMirroredCanvasPoint(landmark) {
    return {
        x: (1 - landmark.x) * drawingCanvas.width,
        y: landmark.y * drawingCanvas.height
    };
}


// ==================================================
// 13. LOW-LEVEL DRAWING
// ==================================================

function drawStrokeSegment(from, to, action) {

    const tool = tools[action.tool] || tools.pen;

    drawingCtx.save();

    drawingCtx.globalAlpha = tool.opacity;
    drawingCtx.globalCompositeOperation = tool.compositeOperation;

    drawingCtx.strokeStyle = action.color;
    drawingCtx.lineWidth = action.size;
    drawingCtx.lineCap = "round";
    drawingCtx.lineJoin = "round";

    drawingCtx.beginPath();
    drawingCtx.moveTo(from.x, from.y);
    drawingCtx.lineTo(to.x, to.y);
    drawingCtx.stroke();

    drawingCtx.restore();
}

function drawEraseSegment(from, to, size) {

    // destination-out → whatever we paint becomes a
    // "hole"; the video underneath shows through.

    drawingCtx.save();

    drawingCtx.globalAlpha = 1;
    drawingCtx.globalCompositeOperation = "destination-out";

    drawingCtx.strokeStyle = "#000000";
    drawingCtx.lineWidth = size;
    drawingCtx.lineCap = "round";
    drawingCtx.lineJoin = "round";

    drawingCtx.beginPath();
    drawingCtx.moveTo(from.x, from.y);
    drawingCtx.lineTo(to.x, to.y);
    drawingCtx.stroke();

    drawingCtx.restore();
}

function drawDot(point, action) {

    drawingCtx.save();

    if (action.type === "erase") {

        drawingCtx.globalCompositeOperation = "destination-out";
        drawingCtx.globalAlpha = 1;
        drawingCtx.fillStyle = "#000000";

        drawingCtx.beginPath();
        drawingCtx.arc(point.x, point.y, action.size / 2, 0, Math.PI * 2);
        drawingCtx.fill();

    } else {

        const tool = tools[action.tool] || tools.pen;

        drawingCtx.globalAlpha = tool.opacity;
        drawingCtx.globalCompositeOperation = tool.compositeOperation;
        drawingCtx.fillStyle = action.color;

        drawingCtx.beginPath();
        drawingCtx.arc(point.x, point.y, action.size / 2, 0, Math.PI * 2);
        drawingCtx.fill();
    }

    drawingCtx.restore();
}


// ==================================================
// 14. ACTION BUILDING
// ==================================================
//
// Actions are the unit of history:
//   { type:"stroke", tool, color, size, points:[...] }
//   { type:"erase",               size, points:[...] }
//
// Undo/redo replay the full list in order (see §21).

function getEraserSize() {
    return Math.max(
        currentPenSize * ERASER_SIZE_MULTIPLIER,
        ERASER_MIN_SIZE
    );
}

function appendPoint(point) {

    if (currentAction === null) return;

    // First point of the action → drop a dot.

    if (previousPoint === null) {
        previousPoint = point;
        currentAction.points.push(point);
        drawDot(point, currentAction);
        return;
    }

    // Ignore micro-jitter.

    if (distance(previousPoint, point) < MIN_MOVEMENT) return;

    if (currentAction.type === "erase") {
        drawEraseSegment(previousPoint, point, currentAction.size);
    } else {
        drawStrokeSegment(previousPoint, point, currentAction);
    }

    currentAction.points.push(point);
    previousPoint = point;
}

function commitCurrentAction() {

    if (currentAction === null) return;

    if (currentAction.points.length > 0) {

        actions.push(currentAction);

        // A new action invalidates the redo branch.

        redoStack = [];

        updateHistoryUI();
    }

    currentAction = null;
}

function stopCurrentAction() {
    commitCurrentAction();
    previousPoint = null;
    smoothedPoint = null;
}


// ==================================================
// 15. ACTION HANDLERS (shared entry points)
// ==================================================

function handleDrawAt(canvasPoint) {

    const point = smoothPoint(canvasPoint);

    if (currentAction === null || currentAction.type !== "stroke") {

        commitCurrentAction();

        currentAction = {
            type: "stroke",
            tool: "pen",
            color: currentColor,
            size: currentPenSize,
            points: []
        };

        previousPoint = null;
    }

    appendPoint(point);
}

function handleEraseAt(canvasPoint, size) {

    const point = smoothPoint(canvasPoint);

    if (currentAction === null || currentAction.type !== "erase") {

        commitCurrentAction();

        currentAction = {
            type: "erase",
            size: size,
            points: []
        };

        previousPoint = null;
    }

    appendPoint(point);
}


// ==================================================
// 16. PEN MODE
// ==================================================
//
// MediaPipe's hand model can't see pens, and general
// object detectors have no "pen" class. So the user
// clicks the pen's coloured tip once. Every frame we
// find the largest blob of that colour, compute its
// principal axis, and take the end farthest from the
// hand as the tip.
//
//   pen visible + pen down → DRAW
//   free hand open palm     → ERASE (hand-sized eraser)
//   free hand fist          → LIFT pen
//   Space / button          → toggle pen down / lifted

const TW = 320, TH = 180, MIN_PEN_AREA = 12;

const trackCtx = Object.assign(
    document.createElement("canvas"),
    { width: TW, height: TH }
).getContext("2d", { willReadFrequently: true });

const maskBuf = new Uint8Array(TW * TH);
const seenBuf = new Uint8Array(TW * TH);
const queue = new Int32Array(TW * TH);

let drawMode = "hand";
let penDown = true;
let penTarget = null;
let isCalibrating = false;


function rgbToHsv(r, g, b) {

    r /= 255; g /= 255; b /= 255;

    const mx = Math.max(r, g, b);
    const d = mx - Math.min(r, g, b);

    let h = 0;

    if (d) {
        if (mx === r) h = ((g - b) / d + 6) % 6;
        else if (mx === g) h = (b - r) / d + 2;
        else h = (r - g) / d + 4;
        h /= 6;
    }

    return [h, mx ? d / mx : 0, mx];
}


function calibratePen(nx, ny) {

    // Sample a small patch around the click.
    // nx/ny are normalised 0–1 coords (already mirrored).

    trackCtx.drawImage(video, 0, 0, TW, TH);

    const cx = Math.round(nx * TW);
    const cy = Math.round(ny * TH);

    const d = trackCtx.getImageData(
        Math.max(0, cx - 3),
        Math.max(0, cy - 3),
        7, 7
    ).data;

    let r = 0, g = 0, b = 0;
    const n = d.length / 4;

    for (let i = 0; i < d.length; i += 4) {
        r += d[i]; g += d[i + 1]; b += d[i + 2];
    }

    r /= n; g /= n; b /= n;

    const [h, s, v] = rgbToHsv(r, g, b);

    penTarget = { h, s, v };

    penSwatch.style.background = `rgb(${r | 0},${g | 0},${b | 0})`;

    setStatus(
        s < 0.25
            ? "Pen calibrated. Tip: a colourful cap or sticker " +
              "tracks far more reliably than a black or white pen."
            : "Pen calibrated. Hold it up and draw!"
    );
}


function detectPen(hands) {

    // Downscale to a small working canvas for speed.

    trackCtx.drawImage(video, 0, 0, TW, TH);
    const d = trackCtx.getImageData(0, 0, TW, TH).data;

    const { h: th, s: ts, v: tv } = penTarget;
    const colourful = ts > 0.25 && tv > 0.2;

    maskBuf.fill(0);
    seenBuf.fill(0);

    // --- Build a mask of matching pixels ---

    for (let p = 0, i = 0; p < TW * TH; p++, i += 4) {

        const r = d[i], g = d[i + 1], b = d[i + 2];

        const mx = Math.max(r, g, b);
        const df = mx - Math.min(r, g, b);

        const v = mx / 255;
        const s = mx ? df / mx : 0;

        if (colourful) {

            if (!df || s < ts * 0.5 || v < 0.15 || Math.abs(v - tv) > 0.4) continue;

            let h;
            if (mx === r) h = ((g - b) / df + 6) % 6;
            else if (mx === g) h = (b - r) / df + 2;
            else h = (r - g) / df + 4;
            h /= 6;

            const dh = Math.abs(h - th);

            if (Math.min(dh, 1 - dh) <= 0.06) maskBuf[p] = 1;

        } else if (s < 0.3 && Math.abs(v - tv) < 0.15) {
            maskBuf[p] = 1;
        }
    }

    // --- Find the largest connected blob (BFS flood fill) ---

    let best = null, used = 0;

    for (let p = 0; p < TW * TH; p++) {

        if (!maskBuf[p] || seenBuf[p]) continue;

        const start = used;
        queue[used++] = p;
        seenBuf[p] = 1;

        for (let q = start; q < used; q++) {

            const c = queue[q];
            const x = c % TW;
            const y = (c / TW) | 0;

            if (x > 0 && maskBuf[c - 1] && !seenBuf[c - 1]) { seenBuf[c - 1] = 1; queue[used++] = c - 1; }
            if (x < TW - 1 && maskBuf[c + 1] && !seenBuf[c + 1]) { seenBuf[c + 1] = 1; queue[used++] = c + 1; }
            if (y > 0 && maskBuf[c - TW] && !seenBuf[c - TW]) { seenBuf[c - TW] = 1; queue[used++] = c - TW; }
            if (y < TH - 1 && maskBuf[c + TW] && !seenBuf[c + TW]) { seenBuf[c + TW] = 1; queue[used++] = c + TW; }
        }

        if (!best || used - start > best.len) best = { start, len: used - start };
    }

    if (!best || best.len < MIN_PEN_AREA) return null;

    // --- Blob centroid ---

    const s0 = best.start, e0 = s0 + best.len, n = best.len;
    let sx = 0, sy = 0;

    for (let k = s0; k < e0; k++) {
        sx += queue[k] % TW;
        sy += (queue[k] / TW) | 0;
    }

    const cx = sx / n, cy = sy / n;

    // --- Principal axis of the blob (PCA) = pen direction ---

    let a = 0, b2 = 0, c2 = 0;

    for (let k = s0; k < e0; k++) {
        const dx = queue[k] % TW - cx;
        const dy = ((queue[k] / TW) | 0) - cy;
        a += dx * dx;
        b2 += dy * dy;
        c2 += dx * dy;
    }

    a /= n; b2 /= n; c2 /= n;

    const ang = 0.5 * Math.atan2(2 * c2, a - b2);
    const ax = Math.cos(ang), ay = Math.sin(ang);

    const disc = Math.sqrt(((a - b2) / 2) ** 2 + c2 * c2);
    const l1 = (a + b2) / 2 + disc;
    const l2 = Math.max((a + b2) / 2 - disc, 0.01);

    let tx = cx, ty = cy;

    // Only use the "far end" heuristic when the blob is
    // elongated enough to be a pen rather than a blob.

    if (Math.sqrt(l1 / l2) > 1.6) {

        // Reference point = wrist of nearest hand.

        let ref = null, bestD = 0.22;

        for (const lm of hands) {
            for (const p of lm) {
                const dd = Math.hypot(p.x - cx / TW, p.y - cy / TH);
                if (dd < bestD) {
                    bestD = dd;
                    ref = { x: lm[0].x * TW, y: lm[0].y * TH };
                }
            }
        }

        const dir = ref
            ? ((cx - ref.x) * ax + (cy - ref.y) * ay >= 0 ? 1 : -1)
            : (ay >= 0 ? -1 : 1);

        // Project every blob pixel on the axis; take the far end.

        let lo = 1e9, hi = -1e9;

        for (let k = s0; k < e0; k++) {
            const t = ((queue[k] % TW - cx) * ax +
                      (((queue[k] / TW) | 0) - cy) * ay) * dir;
            if (t < lo) lo = t;
            if (t > hi) hi = t;
        }

        let px = 0, py = 0, pn = 0;

        for (let k = s0; k < e0; k++) {
            const x = queue[k] % TW;
            const y = (queue[k] / TW) | 0;
            const t = ((x - cx) * ax + (y - cy) * ay) * dir;
            if (t >= hi - 0.15 * (hi - lo)) { px += x; py += y; pn++; }
        }

        if (pn) { tx = px / pn; ty = py / pn; }
    }

    return {
        nx: cx / TW,
        ny: cy / TH,
        point: {
            x: (1 - tx / TW) * drawingCanvas.width,
            y: (ty / TH) * drawingCanvas.height
        }
    };
}


function isPenHand(lm, pen) {
    return lm.some(p => Math.hypot(p.x - pen.nx, p.y - pen.ny) < 0.22);
}

function getPalmCenter(lm) {

    let x = 0, y = 0;

    for (const i of [0, 5, 9, 13, 17]) {
        x += lm[i].x;
        y += lm[i].y;
    }

    return {
        x: (1 - x / 5) * drawingCanvas.width,
        y: (y / 5) * drawingCanvas.height
    };
}

function getPalmEraserSize(lm) {
    return Math.max(
        ERASER_MIN_SIZE,
        distance(lm[0], lm[9]) * drawingCanvas.width * 0.8
    );
}

function drawPenCursor(pt) {

    overlayCtx.save();

    overlayCtx.strokeStyle = currentColor;
    overlayCtx.lineWidth = 3;
    overlayCtx.globalAlpha = penDown ? 1 : 0.4;

    overlayCtx.beginPath();
    overlayCtx.arc(pt.x, pt.y, currentPenSize / 2 + 8, 0, Math.PI * 2);

    overlayCtx.moveTo(pt.x - 16, pt.y);
    overlayCtx.lineTo(pt.x + 16, pt.y);
    overlayCtx.moveTo(pt.x, pt.y - 16);
    overlayCtx.lineTo(pt.x, pt.y + 16);

    overlayCtx.stroke();
    overlayCtx.restore();
}


function processPenMode(hands) {

    hands.forEach(lm => drawHandLandmarks(lm));

    const pen = penTarget ? detectPen(hands) : null;

    let eraseHand = null, lift = false;

    for (const lm of hands) {

        if (pen && isPenHand(lm, pen)) continue;   // the pen-holding hand

        const g = classifyGesture(lm);

        if (g === GESTURE.ERASE) eraseHand = lm;
        else if (g === GESTURE.PAUSE) lift = true;
    }

    const raw = eraseHand ? GESTURE.ERASE
        : lift ? GESTURE.PAUSE
        : (pen && penDown) ? GESTURE.DRAW
        : GESTURE.NONE;

    const gesture = stabiliseGesture(raw);

    if (gesture === GESTURE.ERASE && eraseHand) {

        const palm = getPalmCenter(eraseHand);

        handleEraseAt(palm, getPalmEraserSize(eraseHand));
        drawEraserCursor(palm);
        setGestureStatus("🖐️ Erase", "Erasing");

    } else if (gesture === GESTURE.DRAW && pen) {

        handleDrawAt(pen.point);
        setGestureStatus("🖊️ Pen drawing", "Drawing");

    } else {

        setGestureStatus(
            !penTarget ? "Calibrate your pen"
            : !pen ? "Pen not found"
            : penDown ? "Pen ready" : "Pen lifted",
            "Inactive"
        );
    }

    if (pen) drawPenCursor(pen.point);
}


// ==================================================
// 17. MODE SWITCHING
// ==================================================

function setMode(mode) {

    drawMode = mode;

    stopCurrentAction();
    resetGestureState();

    modeHandBtn.classList.toggle("active", mode === "hand");
    modePenBtn.classList.toggle("active", mode === "pen");
    modeHandBtn.setAttribute("aria-checked", String(mode === "hand"));
    modePenBtn.setAttribute("aria-checked", String(mode === "pen"));

    penToolbar.hidden = mode !== "pen";

    gestureHint.textContent = mode === "pen"
        ? "🖊️ Pen draws · 🖐️ Palm erases · ✊ Free fist lifts"
        : "☝️ Draw · 🖐️ Erase · ✊ Pause";

    // Pen mode needs 2 hands (pen + eraser/lift).
    // Hand mode needs only 1.

    if (handLandmarker) {
        try {
            handLandmarker.setOptions({ numHands: mode === "pen" ? 2 : 1 });
        } catch (error) {
            console.warn("setOptions failed:", error);
        }
    }

    setStatus(mode === "pen"
        ? "Pen mode. Calibrate your pen, then draw with its tip."
        : "Hand mode. Raise your index finger to draw.");
}

function togglePenDown() {
    penDown = !penDown;
    penDownBtn.textContent = penDown ? "Pen: down" : "Pen: lifted";
}

function endCalibration() {
    isCalibrating = false;
    calibrationHint.hidden = true;
    boardEl.classList.remove("calibrating");
}


modeHandBtn.addEventListener("click", () => setMode("hand"));
modePenBtn.addEventListener("click", () => setMode("pen"));
penDownBtn.addEventListener("click", togglePenDown);


calibrateBtn.addEventListener("click", () => {

    if (!isCameraRunning) {
        setStatus("Start the camera first, then calibrate your pen.");
        return;
    }

    isCalibrating = true;
    calibrationHint.hidden = false;
    boardEl.classList.add("calibrating");

    setStatus("Click the coloured tip of your pen in the video.");
});


boardEl.addEventListener("click", (event) => {

    if (!isCalibrating) return;

    const rect = video.getBoundingClientRect();

    calibratePen(
        1 - (event.clientX - rect.left) / rect.width,   // video is mirrored
        (event.clientY - rect.top) / rect.height
    );

    endCalibration();
});


// Quick-colour swatches

document.querySelectorAll(".swatch").forEach(btn => {
    btn.addEventListener("click", () => {
        penColor.value = btn.dataset.color;
        currentColor = btn.dataset.color;
    });
});


// Escape cancels calibration; Space toggles pen in pen mode.

document.addEventListener("keydown", (event) => {

    if (event.key === "Escape" && isCalibrating) {
        endCalibration();
        return;
    }

    if (
        event.code === "Space" &&
        drawMode === "pen" &&
        !/^(BUTTON|INPUT)$/.test(event.target.tagName)
    ) {
        event.preventDefault();
        togglePenDown();
    }
});


// ==================================================
// 18. HAND-MODE GESTURE HANDLER
// ==================================================

function handleGesture(gesture, landmarks) {

    if (gesture === GESTURE.DRAW) {

        handleDrawAt(getMirroredCanvasPoint(landmarks[8]));
        setGestureStatus(GESTURE_LABEL[GESTURE.DRAW], "Drawing");

    } else if (gesture === GESTURE.ERASE) {

        handleEraseAt(
            getPalmCenter(landmarks),
            getPalmEraserSize(landmarks)
        );

        drawEraserCursor(getPalmCenter(landmarks));
        setGestureStatus(GESTURE_LABEL[GESTURE.ERASE], "Erasing");

    } else if (gesture === GESTURE.PAUSE) {

        setGestureStatus(GESTURE_LABEL[GESTURE.PAUSE], "Inactive");

    } else {

        setGestureStatus(GESTURE_LABEL[GESTURE.NONE], "Inactive");
    }
}


// ==================================================
// 19. LANDMARK VISUALISATION
// ==================================================

const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [17, 18], [18, 19], [19, 20],
    [0, 17]
];

function drawHandLandmarks(landmarks) {

    const width = overlayCanvas.width;
    const height = overlayCanvas.height;

    overlayCtx.save();

    overlayCtx.strokeStyle = "rgba(49, 92, 69, 0.85)";
    overlayCtx.lineWidth = 3;

    for (const [startIndex, endIndex] of HAND_CONNECTIONS) {

        const start = landmarks[startIndex];
        const end = landmarks[endIndex];

        overlayCtx.beginPath();
        overlayCtx.moveTo((1 - start.x) * width, start.y * height);
        overlayCtx.lineTo((1 - end.x) * width, end.y * height);
        overlayCtx.stroke();
    }

    for (const landmark of landmarks) {

        overlayCtx.beginPath();
        overlayCtx.arc(
            (1 - landmark.x) * width,
            landmark.y * height,
            5, 0, Math.PI * 2
        );

        overlayCtx.fillStyle = "#315C45";
        overlayCtx.fill();
    }

    // Highlight the index fingertip (landmark 8).

    const indexTip = landmarks[8];

    overlayCtx.beginPath();
    overlayCtx.arc(
        (1 - indexTip.x) * width,
        indexTip.y * height,
        10, 0, Math.PI * 2
    );

    overlayCtx.strokeStyle = "#D6A343";
    overlayCtx.lineWidth = 4;
    overlayCtx.stroke();

    overlayCtx.restore();
}

function drawEraserCursor(point) {

    const radius = (
        currentAction && currentAction.type === "erase"
            ? currentAction.size
            : getEraserSize()
    ) / 2;

    overlayCtx.save();

    overlayCtx.beginPath();
    overlayCtx.arc(point.x, point.y, radius, 0, Math.PI * 2);

    overlayCtx.strokeStyle = "rgba(181, 74, 74, 0.9)";
    overlayCtx.lineWidth = 3;
    overlayCtx.setLineDash([10, 8]);
    overlayCtx.stroke();

    overlayCtx.restore();
}


// ==================================================
// 20. FRAME PROCESSING
// ==================================================

function processFrame() {

    if (!handLandmarker || !isCameraRunning) return;
    if (video.readyState < 2) return;

    // Skip identical video frames.

    if (video.currentTime === lastVideoTime) return;
    lastVideoTime = video.currentTime;

    const results = handLandmarker.detectForVideo(video, performance.now());

    // Clear only the overlay — never the drawing layer.

    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    const hands = results.landmarks || [];
    handCount.textContent = String(hands.length);


    // Pen mode has its own pipeline.

    if (drawMode === "pen") {
        processPenMode(hands);
        return;
    }


    // ---------- Hand mode ----------

    if (hands.length === 0) {

        resetGestureState();
        stopCurrentAction();

        setGestureStatus("No hand detected", "Inactive");
        return;
    }

    const landmarks = hands[0];

    drawHandLandmarks(landmarks);

    const rawGesture = classifyGesture(landmarks);
    const gesture = stabiliseGesture(rawGesture);

    handleGesture(gesture, landmarks);
}

function renderLoop() {

    if (!isCameraRunning) return;

    processFrame();

    animationId = requestAnimationFrame(renderLoop);
}


// ==================================================
// 21. HISTORY REPLAY
// ==================================================
//
// Undo/Redo both work by:
//   1. wipe the drawing canvas
//   2. replay every action in `actions`, in order
//
// Because erase actions replay in the correct
// position, erased pixels stay erased and un-erased
// pixels come back. This is what makes undo work
// across erasing.

function replayAction(action) {

    if (action.points.length === 0) return;

    if (action.points.length === 1) {
        drawDot(action.points[0], action);
        return;
    }

    if (action.type === "erase") {

        drawingCtx.save();

        drawingCtx.globalAlpha = 1;
        drawingCtx.globalCompositeOperation = "destination-out";

        drawingCtx.strokeStyle = "#000000";
        drawingCtx.lineWidth = action.size;
        drawingCtx.lineCap = "round";
        drawingCtx.lineJoin = "round";

        drawingCtx.beginPath();
        drawingCtx.moveTo(action.points[0].x, action.points[0].y);

        for (let i = 1; i < action.points.length; i++) {
            drawingCtx.lineTo(action.points[i].x, action.points[i].y);
        }

        drawingCtx.stroke();
        drawingCtx.restore();

        return;
    }

    const tool = tools[action.tool] || tools.pen;

    drawingCtx.save();

    drawingCtx.globalAlpha = tool.opacity;
    drawingCtx.globalCompositeOperation = tool.compositeOperation;

    drawingCtx.strokeStyle = action.color;
    drawingCtx.lineWidth = action.size;
    drawingCtx.lineCap = "round";
    drawingCtx.lineJoin = "round";

    drawingCtx.beginPath();
    drawingCtx.moveTo(action.points[0].x, action.points[0].y);

    for (let i = 1; i < action.points.length; i++) {
        drawingCtx.lineTo(action.points[i].x, action.points[i].y);
    }

    drawingCtx.stroke();
    drawingCtx.restore();
}

function redrawAll() {

    drawingCtx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);

    for (const action of actions) {
        replayAction(action);
    }
}


// ==================================================
// 22. UNDO / REDO
// ==================================================

function undo() {

    stopCurrentAction();

    if (actions.length === 0) {
        setStatus("There is nothing to undo.");
        updateHistoryUI();
        return;
    }

    redoStack.push(actions.pop());

    redrawAll();
    updateHistoryUI();
    setStatus("Undo.");
}

function redo() {

    if (redoStack.length === 0) {
        setStatus("There is nothing to redo.");
        return;
    }

    stopCurrentAction();

    actions.push(redoStack.pop());

    redrawAll();
    updateHistoryUI();
    setStatus("Redo.");
}


// ==================================================
// 23. SAVE
// ==================================================

function saveWhiteboard() {

    const exportCanvas = document.createElement("canvas");

    exportCanvas.width = drawingCanvas.width;
    exportCanvas.height = drawingCanvas.height;

    const exportCtx = exportCanvas.getContext("2d");

    exportCtx.fillStyle = "#ffffff";
    exportCtx.fillRect(0, 0, exportCanvas.width, exportCanvas.height);

    exportCtx.drawImage(drawingCanvas, 0, 0);

    const link = document.createElement("a");

    link.download = "HandWave-whiteboard.png";
    link.href = exportCanvas.toDataURL("image/png");
    link.click();

    setStatus("Whiteboard exported as PNG.");
}


// ==================================================
// 24. TOOLBAR EVENT LISTENERS
// ==================================================

penColor.addEventListener("input", () => {
    currentColor = penColor.value;
});

penSize.addEventListener("input", () => {
    currentPenSize = Number(penSize.value);
    penSizeValue.textContent = String(currentPenSize);
});

clearBtn.addEventListener("click", clearCanvas);
undoBtn.addEventListener("click", undo);
redoBtn.addEventListener("click", redo);
saveBtn.addEventListener("click", saveWhiteboard);
startCameraBtn.addEventListener("click", startCamera);
stopCameraBtn.addEventListener("click", stopCamera);
fullscreenBtn.addEventListener("click", toggleFullscreen);


// ==================================================
// 25. KEYBOARD SHORTCUTS (history)
// ==================================================
//
//   Ctrl/Cmd + Z          → Undo
//   Ctrl/Cmd + Shift + Z  → Redo
//   Ctrl/Cmd + Y          → Redo (Windows)

document.addEventListener("keydown", (event) => {

    const modifier = event.ctrlKey || event.metaKey;
    if (!modifier) return;

    const key = event.key.toLowerCase();

    if (key === "z" && !event.shiftKey) {
        event.preventDefault();
        undo();
        return;
    }

    if ((key === "z" && event.shiftKey) || key === "y") {
        event.preventDefault();
        redo();
    }
});


// ==================================================
// 26. INITIALISE
// ==================================================

setupCanvas();

// Set initial UI state to match `drawMode = "hand"`.

modeHandBtn.classList.add("active");
modePenBtn.classList.remove("active");
penToolbar.hidden = true;

updateHistoryUI();

createHandLandmarker();