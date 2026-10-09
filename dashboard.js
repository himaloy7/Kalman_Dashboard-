// ==================================================================
// GLOBAL SIMULATION STATE
// ==================================================================
let isRunning = true;
let frameCounter = 0;
const historyWindowSize = 50;

let timelineLabels = Array.from({length: historyWindowSize}, (_, i) => -historyWindowSize + i);
let truePositionData = Array(historyWindowSize).fill(0);
let estimatedPositionData = Array(historyWindowSize).fill(0);
let covarianceData = Array(historyWindowSize).fill(0.1);
let noiseRData = Array(historyWindowSize).fill(0.2);

// ==================================================================
// DOM HANDLES
// ==================================================================
const jitterSlider = document.getElementById('jitter-slider');
const swaySlider = document.getElementById('sway-slider');
const simToggleBtn = document.getElementById('sim-toggle-btn');
const resetBtn = document.getElementById('reset-btn');
const addRisBtn = document.getElementById('add-ris-btn');
const addBlockerBtn = document.getElementById('add-blocker-btn');
const clearSceneBtn = document.getElementById('clear-scene-btn');
const windowSizeSelect = document.getElementById('window-size-select');
const bandwidthSlider = document.getElementById('bandwidth-slider');

// ==================================================================
// SESSION ID — isolates EKF state when multiple browsers connect.
// ==================================================================
function getSessionId() {
    let sid = sessionStorage.getItem('kalman_session_id');
    if (!sid) {
        sid = 'sid_' + Math.random().toString(36).slice(2) + '_' + Date.now();
        sessionStorage.setItem('kalman_session_id', sid);
    }
    return sid;
}

// ==================================================================
// SCENE STATE
// Canvas coordinate space; 5 px = 1 m of airspace.
// ==================================================================
const PIXELS_PER_METER = 25;

const scene = {
    base:    { x: 60,  y: 0, radius: 22, fixed: true,  label: 'BASE STN' },
    drone:   { x: 0,   y: 0, radius: 16, draggable: true, label: 'AERIAL NODE' },
    ris:     [ { x: 0, y: 0, width: 80, height: 15, draggable: true, label: 'RIS SURFACE ARRAY' } ],
    blockers: [],
    absorption_zones: []
};

// Blocker defaults applied to every new blocker.
function makeBlocker(cx, cy) {
    return {
        x: cx, y: cy,
        radius: 22,
        draggable: true,
        // Motion
        motion: 'static',       // 'static' | 'sweep' | 'random_walk' | 'orbit_base' | 'orbit_drone'
        speed_mps: 1.2,
        axis: 'x',              // for 'sweep'
        sweep_min: 100,         // px bounds for sweep
        sweep_max: 500,
        sweep_dir: 1,
        phase: Math.random() * 2 * Math.PI,   // for orbit modes
        orbit_radius: 120,
        // Physics mode
        mode: 'occluder'        // 'occluder' | 'reflector'
    };
}

// Build an irregular polygon around a center. Vertex count between
// 8 and 12, radius jittered ±30% around the base ellipse.
function makeAbsorptionPolygon(cx, cy, rx, ry) {
    const n = 8 + Math.floor(Math.random() * 5);   // 8..12 vertices
    const verts = [];
    for (let i = 0; i < n; i++) {
        const theta = (i / n) * 2 * Math.PI;
        const jitter = 0.7 + Math.random() * 0.6;   // 0.7..1.3
        verts.push({
            x: cx + Math.cos(theta) * rx * jitter,
            y: cy + Math.sin(theta) * ry * jitter
        });
    }
    return verts;
}

function makeAbsorptionZone(cx, cy) {
    const rx = 40 + Math.random() * 30;
    const ry = 30 + Math.random() * 30;
    return {
        cx, cy, rx, ry,
        vertices: makeAbsorptionPolygon(cx, cy, rx, ry),
        humidity_pct: 60,
        temp_k: 293.15,
        // Motion
        motion: 'static',
        speed_mps: 0.5,
        axis: 'x',
        sweep_min: 100,
        sweep_max: 500,
        sweep_dir: 1,
        phase: Math.random() * 2 * Math.PI,
        orbit_radius: 150,
        draggable: true
    };
}

let sceneInitialized = false;
let dragging = null;
let dragOffset = { x: 0, y: 0 };

function initSceneIfNeeded(w, h) {
    if (sceneInitialized) return;
    scene.base.y = h - 70;
    scene.base.x = 60;
    scene.drone.x = w - 100;
    scene.drone.centerY = h / 2;
    scene.drone.y = h / 2;
    scene.ris[0].x = w / 2;
    scene.ris[0].y = 40;
    sceneInitialized = true;
}

// ==================================================================
// GEOMETRY HELPERS
// ==================================================================
function dist(ax, ay, bx, by) {
    return Math.hypot(bx - ax, by - ay);
}

// Segment-circle intersection (returns true if line segment a->b
// passes within r of point c).
function segmentIntersectsCircle(ax, ay, bx, by, cx, cy, r) {
    const dx = bx - ax, dy = by - ay;
    const lenSq = dx*dx + dy*dy;
    if (lenSq === 0) return dist(ax, ay, cx, cy) <= r;
    let t = ((cx - ax) * dx + (cy - ay) * dy) / lenSq;
    t = Math.max(0, Math.min(1, t));
    const px = ax + t * dx, py = ay + t * dy;
    return dist(px, py, cx, cy) <= r;
}

function segmentBlocked(ax, ay, bx, by, blockers) {
    for (const b of blockers) {
        // Reflector-mode blockers do not occlude — they redirect.
        if (b.mode === 'reflector') continue;
        if (segmentIntersectsCircle(ax, ay, bx, by, b.x, b.y, b.radius)) {
            return true;
        }
    }
    return false;
}

// Returns the first reflector-mode blocker the LOS path hits, or null.
function firstReflectorOnPath(ax, ay, bx, by, blockers) {
    for (const b of blockers) {
        if (b.mode !== 'reflector') continue;
        if (segmentIntersectsCircle(ax, ay, bx, by, b.x, b.y, b.radius)) {
            return b;
        }
    }
    return null;
}

// Approximate polygon-segment overlap length in pixels.
// Mirrors the Python path_length_in_polygon function; both use the
// same uniform-sampling scheme so the frontend and backend agree.
function pathLengthInPolygon(ax, ay, bx, by, vertices, samples=40) {
    const totalLen = Math.hypot(bx - ax, by - ay);
    if (totalLen <= 0) return 0;
    let insideCount = 0;
    for (let k = 0; k < samples; k++) {
        const t = (k + 0.5) / samples;
        const px = ax + t * (bx - ax);
        const py = ay + t * (by - ay);
        if (pointInPolygon(px, py, vertices)) insideCount++;
    }
    return (insideCount / samples) * totalLen;
}

function pointInPolygon(px, py, vertices) {
    const n = vertices.length;
    let inside = false;
    let j = n - 1;
    for (let i = 0; i < n; i++) {
        const xi = vertices[i].x, yi = vertices[i].y;
        const xj = vertices[j].x, yj = vertices[j].y;
        if (((yi > py) !== (yj > py)) &&
            (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi)) {
            inside = !inside;
        }
        j = i;
    }
    return inside;
}

// Total extra absorption (dB) picked up by a segment passing through
// any absorption zone. Uses the reduced P.676 coefficient computed
// per-zone. Same alpha formula as kalman.py :: itu_p676_reduced_alpha_db_per_km().
function absorptionLossDb(ax, ay, bx, by, zones) {
    let total_db = 0;
    for (const z of zones) {
        const lenPx = pathLengthInPolygon(ax, ay, bx, by, z.vertices);
        if (lenPx <= 0) continue;
        const lenM = lenPx / PIXELS_PER_METER;
        const alpha_db_per_km = reducedP676AlphaDbPerKm(140e9, z.humidity_pct, z.temp_k);
        total_db += alpha_db_per_km * (lenM / 1000.0);
    }
    return total_db;
}

function saturationVaporPressureHpa(temp_k) {
    const tc = temp_k - 273.15;
    return 6.112 * Math.exp((17.62 * tc) / (243.12 + tc));
}

function waterVaporDensityGm3(rh_pct, temp_k) {
    const e_hpa = (rh_pct / 100.0) * saturationVaporPressureHpa(temp_k);
    const e_pa = e_hpa * 100.0;
    return (e_pa * 18.015) / (8.314 * temp_k);
}

// ITU-R P.676-13 gaseous attenuation at a single frequency, dB/km.
// Reduced to 4 lines (2 O2, 2 H2O). Matches full P.676-13 within 5%
// at 140 GHz dry-air, 10% at the 60 GHz O2 band, 20% at the 183 GHz
// H2O line. Under-estimates by 50-80% at 140 GHz with high humidity
// (dropped H2O lines).
// Mirrors kalman.py :: itu_p676_reduced_alpha_db_per_km() exactly.
function reducedP676AlphaDbPerKm(freq_hz, humidity_pct, temp_k) {
    const f = freq_hz / 1e9;
    const T = Math.max(250, Math.min(temp_k, 320));
    const theta = 300.0 / T;

    const e = (humidity_pct / 100.0) * saturationVaporPressureHpa(T);
    const p = Math.max(1013.25 - e, 1.0);

    // Oxygen lines: f0, a1, a2, a3, a4, a5, a6
    const o2Lines = [
        [118.750334, 940.300, 0.010, 16.640, 0.0, -0.439, 0.079],
        [ 60.434778, 2438.000, 0.386, 13.390, 0.0,  6.342, -2.825]
    ];
    // Water vapour lines: f0, b1, b2, b3, b4, b5, b6
    const h2oLines = [
        [ 22.235080, 0.1079, 2.144, 26.38, 8.76, 5.087, 1.00],
        [183.310087, 2.273,  3.668, 29.0,  6.77, 5.022, 2.85]
    ];

    let Npp = 0.0;

    for (const [f0, a1, a2, a3, a4, a5, a6] of o2Lines) {
        const S = a1 * 1e-6 * p * Math.pow(theta, 3) * Math.exp(a2 * (1.0 - theta));
        let df = a3 * 1e-4 * (p * Math.pow(theta, 0.8 - a4) + 1.1 * e * theta);
        df = Math.sqrt(df * df + 2.25e-6);
        const delta = (a5 + a6 * theta) * 1e-4 * (p + e) * Math.pow(theta, 0.8);
        const Fi = (f / f0) * (
            (df - delta * (f0 - f)) / (Math.pow(f0 - f, 2) + df * df) +
            (df - delta * (f0 + f)) / (Math.pow(f0 + f, 2) + df * df)
        );
        Npp += S * Fi;
    }

    for (const [f0, b1, b2, b3, b4, b5, b6] of h2oLines) {
        const S = b1 * 1.0 * e * Math.pow(theta, 3.5) * Math.exp(b2 * (1.0 - theta));
        let df = b3 * 1e-4 * (p * Math.pow(theta, b4) + b5 * e * Math.pow(theta, b6));
        df = 0.535 * df + Math.sqrt(0.217 * df * df +
             (2.1316e-12 * f0 * f0) / theta);
        const Fi = (f / f0) * (
            df / (Math.pow(f0 - f, 2) + df * df) +
            df / (Math.pow(f0 + f, 2) + df * df)
        );
        Npp += S * Fi;
    }

    const d_width = 5.6e-4 * (p + e) * Math.pow(theta, 0.8);
    const Npp_D = f * p * (theta * theta) * (
        6.14e-5 / (d_width * (1.0 + Math.pow(f / d_width, 2))) +
        (1.4e-12 * p * Math.pow(theta, 1.5)) / (1.0 + 1.9e-5 * Math.pow(f, 1.5))
    );
    Npp += Npp_D;

    return Math.max(0, 0.1820 * f * Npp);
}

// ==================================================================
// LINK STATE COMPUTATION
// Returns geometry facts that the backend will use to derive SNR.
// ==================================================================
function computeLinkState() {
    const bx = scene.base.x, by = scene.base.y;
    const dx = scene.drone.x, dy = scene.drone.y;

    const losBlocked = segmentBlocked(bx, by, dx, dy, scene.blockers);
    const losDistPx = dist(bx, by, dx, dy);

    // RIS path: base -> each RIS -> drone. Find the best (unblocked,
    // shortest) RIS path. Multiple RIS contribute array gain.
    let bestRisDistPx = Infinity;
    let bestRisIndex = -1;
    let usableRisCount = 0;

    for (let i = 0; i < scene.ris.length; i++) {
        const r = scene.ris[i];
        const rx = r.x, ry = r.y + r.height / 2;
        const hop1Blocked = segmentBlocked(bx, by, rx, ry, scene.blockers);
        const hop2Blocked = segmentBlocked(rx, ry, dx, dy, scene.blockers);
        if (hop1Blocked || hop2Blocked) continue;
        usableRisCount++;
        const totalPx = dist(bx, by, rx, ry) + dist(rx, ry, dx, dy);
        if (totalPx < bestRisDistPx) {
            bestRisDistPx = totalPx;
            bestRisIndex = i;
        }
    }

    const anyRisUsable = usableRisCount > 0;

    // Split the RIS path into two legs for the bistatic exponent model.
    let bestD1 = 0, bestD2 = 0;
    if (bestRisIndex >= 0) {
        const r = scene.ris[bestRisIndex];
        const rx = r.x, ry = r.y + r.height / 2;
        bestD1 = dist(bx, by, rx, ry) / PIXELS_PER_METER;
        bestD2 = dist(rx, ry, dx, dy) / PIXELS_PER_METER;
    }

    // --- Absorption losses on LOS and on the best RIS path ---
    const absLossLosDb = absorptionLossDb(bx, by, dx, dy, scene.absorption_zones);

    let absLossRisDb = 0;
    if (bestRisIndex >= 0) {
        const r = scene.ris[bestRisIndex];
        const rx = r.x, ry = r.y + r.height / 2;
        absLossRisDb = absorptionLossDb(bx, by, rx, ry, scene.absorption_zones)
                     + absorptionLossDb(rx, ry, dx, dy, scene.absorption_zones);
    }

    // --- Reflector-mode blockers on LOS ---
    // If a reflector-mode blocker sits on the direct LOS segment, the
    // link is not blocked — but it takes a specular reflection penalty.
    // 15 dB is a typical mid-band value for a rough metal surface at
    // 140 GHz; see paper p. 5, "rooftops may have metal structures and
    // air conditioning equipment to create strong specular multipath".
    const REFLECTOR_PENALTY_DB = 15.0;
    const reflectorLOS = firstReflectorOnPath(bx, by, dx, dy, scene.blockers);
    const reflectorLosPenDb = reflectorLOS ? REFLECTOR_PENALTY_DB : 0.0;

    return {
        los_blocked: losBlocked,
        los_distance_m: losDistPx / PIXELS_PER_METER,
        ris_blocked: !anyRisUsable,
        ris_d1_m: bestD1,
        ris_d2_m: bestD2,
        ris_distance_m: isFinite(bestRisDistPx) ? bestRisDistPx / PIXELS_PER_METER : 0,
        ris_count: usableRisCount,
        best_ris_index: bestRisIndex,

        absorption_loss_los_db:      absLossLosDb,
        absorption_loss_ris_db:      absLossRisDb,
        reflector_los_penalty_db:    reflectorLosPenDb,
        reflector_ris_penalty_db:    0.0,
        reflector_blocker:           reflectorLOS
    };
}

// ==================================================================
// MOUSE INTERACTION
// ==================================================================
const canvas = document.getElementById('spatial-canvas');
const ctx = canvas.getContext('2d');

function canvasCoords(e) {
    const rect = canvas.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
}

function hitTest(mx, my) {
    // Order matters: topmost first.
    for (const b of scene.blockers) {
        if (dist(mx, my, b.x, b.y) < b.radius) return b;
    }
    // Absorption zones: point-in-polygon test. These carry cx/cy
    // for the drag anchor.
    for (const z of scene.absorption_zones) {
        if (pointInPolygon(mx, my, z.vertices)) return z;
    }
    for (const r of scene.ris) {
        if (mx >= r.x - r.width/2 && mx <= r.x + r.width/2 &&
            my >= r.y && my <= r.y + r.height) return r;
    }
    if (dist(mx, my, scene.drone.x, scene.drone.y) < scene.drone.radius) return scene.drone;
    return null;
}

canvas.addEventListener('mousedown', (e) => {
    const { x, y } = canvasCoords(e);
    const obj = hitTest(x, y);
    if (!obj || !obj.draggable) return;
    dragging = obj;
    // Absorption zones anchor at (cx, cy); others at (x, y).
    if (scene.absorption_zones.includes(obj)) {
        dragOffset.x = x - obj.cx;
        dragOffset.y = y - obj.cy;
    } else {
        dragOffset.x = x - obj.x;
        dragOffset.y = y - obj.y;
    }
    canvas.classList.add('dragging');
    e.preventDefault();
});

canvas.addEventListener('mousemove', (e) => {
    if (!dragging) {
        const { x, y } = canvasCoords(e);
        canvas.style.cursor = hitTest(x, y) ? 'grab' : 'default';
        return;
    }
    const { x, y } = canvasCoords(e);

    // Absorption zones use cx/cy as their anchor; everything else uses x/y.
    if (scene.absorption_zones.includes(dragging)) {
        dragging.cx = x - dragOffset.x;
        dragging.cy = y - dragOffset.y;
        // Rebuild the polygon around the new center.
        dragging.vertices = makeAbsorptionPolygon(dragging.cx, dragging.cy, dragging.rx, dragging.ry);
    } else if (dragging === scene.drone) {
        dragging.x = x - dragOffset.x;
        dragging.centerY = y - dragOffset.y;
    } else {
        dragging.x = x - dragOffset.x;
        dragging.y = y - dragOffset.y;
    }
});

canvas.addEventListener('mouseup', () => {
    // Nothing to do for the drone: mousemove has been writing
    // centerY the whole time. Just clear the drag state.
    dragging = null;
    canvas.classList.remove('dragging');
});

canvas.addEventListener('mouseleave', () => {
    dragging = null;
    canvas.classList.remove('dragging');
});

// ==================================================================
// SCENE EDIT BUTTONS
// ==================================================================
addRisBtn.addEventListener('click', () => {
    const w = canvas.width || 800;
    const h = canvas.height || 380;
    scene.ris.push({
        x: 150 + Math.random() * (w - 300),
        y: 60 + Math.random() * 80,
        width: 80,
        height: 15,
        draggable: true,
        label: 'RIS'
    });
});

addBlockerBtn.addEventListener('click', () => {
    const w = canvas.width || 800;
    const h = canvas.height || 380;
    const b = makeBlocker(
        200 + Math.random() * (w - 400),
        h / 2 + (Math.random() - 0.5) * 100
    );
    b.sweep_min = 100;
    b.sweep_max = w - 100;
    scene.blockers.push(b);
});

const addAbsorptionBtn = document.getElementById('add-absorption-btn');
if (addAbsorptionBtn) {
    addAbsorptionBtn.addEventListener('click', () => {
        const w = canvas.width || 800;
        const h = canvas.height || 380;
        const z = makeAbsorptionZone(
            200 + Math.random() * (w - 400),
            h / 2 + (Math.random() - 0.5) * 100
        );
        z.sweep_min = 100;
        z.sweep_max = w - 100;
        scene.absorption_zones.push(z);
    });
}

clearSceneBtn.addEventListener('click', () => {
    const firstRis = scene.ris[0];
    scene.ris = [firstRis];
    scene.blockers = [];
    scene.absorption_zones = [];
});

// ==================================================================
// UI LISTENERS
// ==================================================================
jitterSlider.addEventListener('input', (e) => document.getElementById('jitter-val').innerText = parseFloat(e.target.value).toFixed(1) + ' mrad p-p');
swaySlider.addEventListener('input', (e) => document.getElementById('sway-val').innerText = parseFloat(e.target.value).toFixed(1) + ' mrad');

simToggleBtn.addEventListener('click', () => {
    isRunning = !isRunning;
    simToggleBtn.innerText = isRunning ? "Pause Simulation Loop" : "Resume Simulation Loop";
});

resetBtn.addEventListener('click', async () => {
    truePositionData.fill(0);
    estimatedPositionData.fill(0);
    covarianceData.fill(0.1);
    noiseRData.fill(0.2);
    frameCounter = 0;
    chartTracking.update();
    chartCovariance.update();
    try {
        await fetch('/api/kalman-reset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: getSessionId() })
        });
    } catch (e) {
        console.error("[-] EKF reset request failed:", e);
    }
});

// ==================================================================
// CHART.JS INITIALIZATION
// ==================================================================
const ctxTrack = document.getElementById('chart-tracking').getContext('2d');
const chartTracking = new Chart(ctxTrack, {
    type: 'line',
    data: {
        labels: timelineLabels,
        datasets: [
            { label: 'True Angle (millidegrees)', borderColor: '#10b981', data: truePositionData, borderWidth: 2, pointRadius: 0, fill: false },
            { label: 'EKF Estimate (millidegrees)', borderColor: '#38bdf8', data: estimatedPositionData, borderWidth: 2, pointRadius: 0, fill: false }
        ]
    },
    options: {
        responsive: true, maintainAspectRatio: false, animation: false,
                scales: { y: { min: -3000, max: 3000, title: { display: true, text: 'Angle (millidegrees)', color: '#f8fafc' } } }
    }
});

const ctxCov = document.getElementById('chart-covariance').getContext('2d');
const chartCovariance = new Chart(ctxCov, {
    type: 'line',
    data: {
        labels: timelineLabels,
        datasets: [
            { label: 'Error Covariance Vector (P_k)', borderColor: '#ef4444', data: covarianceData, borderWidth: 2, pointRadius: 0, fill: false, yAxisID: 'y' },
            { label: 'Meas Noise Matrix (R_k)',      borderColor: '#eab308', data: noiseRData,     borderWidth: 1.5, pointRadius: 0, fill: false, yAxisID: 'y1' }
        ]
    },
    options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        scales: {
            y:  { type: 'linear', display: true, position: 'left',  min: 0, max: 10, title: { display: true, text: 'P_k Scale' } },
            y1: { type: 'linear', display: true, position: 'right', min: 0, max: 600, grid: { drawOnChartArea: false }, title: { display: true, text: 'R_k Dynamic Magnitude' } }
        }
    }
});

// ------------------------------------------------------------------
// CHART 3: Live SNR decomposition (LOS / RIS / Active)
// ------------------------------------------------------------------
const snrLabels = Array.from({length: historyWindowSize}, (_, i) => -historyWindowSize + i);
const snrLosData   = Array(historyWindowSize).fill(-100);
const snrRisData   = Array(historyWindowSize).fill(-100);
const snrActiveData = Array(historyWindowSize).fill(-100);

const ctxSnr = document.getElementById('chart-snr').getContext('2d');
const chartSnr = new Chart(ctxSnr, {
    type: 'line',
    data: {
        labels: snrLabels,
        datasets: [
            { label: 'LOS SNR',    borderColor: '#10b981', data: snrLosData,    borderWidth: 2, pointRadius: 0 },
            { label: 'RIS SNR',    borderColor: '#a855f7', data: snrRisData,    borderWidth: 2, pointRadius: 0 },
            { label: 'Active SNR', borderColor: '#f59e0b', data: snrActiveData, borderWidth: 2.5, pointRadius: 0 }
        ]
    },
    options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        scales: {
            y: { min: -20, max: 60, title: { display: true, text: 'SNR (dB)', color: '#f8fafc' } }
        },
        plugins: {
            title: { display: true, text: 'Live SNR Decomposition', color: '#f8fafc' },
            legend: { labels: { color: '#f8fafc', font: { size: 10 } } }
        }
    }
});

// ------------------------------------------------------------------
// CHART 4: Live RIS path-loss exponent n_eff
// ------------------------------------------------------------------
const nEffData = Array(historyWindowSize).fill(2.0);

const ctxNeff = document.getElementById('chart-neff').getContext('2d');
const chartNeff = new Chart(ctxNeff, {
    type: 'line',
    data: {
        labels: snrLabels,
        datasets: [
            { label: 'RIS path-loss exponent n_eff',
              borderColor: '#22d3ee', data: nEffData, borderWidth: 2.5, pointRadius: 0 }
        ]
    },
    options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        scales: {
            y: { min: 1.8, max: 4.2,
                 title: { display: true, text: 'n_eff', color: '#f8fafc' },
                 ticks: { stepSize: 0.5 } }
        },
        plugins: {
            title: { display: true, text: 'Bistatic RIS Path-Loss Exponent (paper p. 4)',
                     color: '#f8fafc' }
        }
    }
});

// ==================================================================
// DRAWING
// ==================================================================
function drawScene(targetAngle, innovationY, linkState, blockageProb, activeLink) {
    // Resize the backing store only when the CSS size actually changed.
    const cssW = canvas.offsetWidth;
    const cssH = canvas.offsetHeight;
    if (canvas.width !== cssW || canvas.height !== cssH) {
        canvas.width = cssW;
        canvas.height = cssH;
    }
    const w = canvas.width;
    const h = canvas.height;

    initSceneIfNeeded(w, h);

    const isRisActive = activeLink === "RIS REFLECTED NODE ACTIVE";
    const isDown      = activeLink === "LINK DOWN";

    // Visual Y = user-chosen center + sway offset. The drag handler
    // writes to scene.drone.centerY; sway is layered on top and
    // never overwrites it.
    scene.drone.y = scene.drone.centerY + targetAngle * 0.02;

    ctx.clearRect(0, 0, w, h);

    // --- Base station ---
    ctx.fillStyle = "#475569";
    ctx.fillRect(scene.base.x - 20, scene.base.y - 20, 40, 40);
    ctx.fillStyle = "#f8fafc";
    ctx.font = "bold 10px monospace";
    ctx.fillText(scene.base.label, scene.base.x - 25, scene.base.y + 35);

    // --- Absorption zones (drawn first, under everything else) ---
    for (const z of scene.absorption_zones) {
        ctx.beginPath();
        ctx.moveTo(z.vertices[0].x, z.vertices[0].y);
        for (let k = 1; k < z.vertices.length; k++) {
            ctx.lineTo(z.vertices[k].x, z.vertices[k].y);
        }
        ctx.closePath();
        ctx.fillStyle = "rgba(234, 179, 8, 0.10)";
        ctx.fill();
        ctx.strokeStyle = "rgba(234, 179, 8, 0.55)";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        ctx.stroke();
        ctx.setLineDash([]);

        // Label
        ctx.fillStyle = "rgba(234, 179, 8, 0.85)";
        ctx.font = "bold 10px monospace";
        ctx.fillText(
            `H₂O zone · ${z.humidity_pct}% · ${Math.round(z.temp_k)}K`,
            z.cx - 55, z.cy - z.ry - 6
        );
    }

    // --- Blockers ---
    for (const b of scene.blockers) {
        if (b.mode === 'reflector') {
            // Reflector: magenta dashed circle with "⟳" tag
            ctx.fillStyle = "rgba(168, 85, 247, 0.25)";
            ctx.beginPath();
            ctx.arc(b.x, b.y, b.radius, 0, 2 * Math.PI);
            ctx.fill();
            ctx.strokeStyle = "#a855f7";
            ctx.lineWidth = 2;
            ctx.setLineDash([6, 3]);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = "#e9d5ff";
            ctx.font = "bold 12px monospace";
            ctx.fillText("⟳", b.x - 5, b.y + 4);
        } else {
            // Occluder: solid red circle (existing behavior)
            ctx.fillStyle = "rgba(239, 68, 68, 0.35)";
            ctx.beginPath();
            ctx.arc(b.x, b.y, b.radius, 0, 2 * Math.PI);
            ctx.fill();
            ctx.strokeStyle = "#ef4444";
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }
    }

    // --- RIS surfaces ---
    for (let i = 0; i < scene.ris.length; i++) {
        const r = scene.ris[i];
        const isActive = (i === linkState.best_ris_index) && isRisActive;
        ctx.fillStyle = isActive ? "#22d3ee" : "#a855f7";
        ctx.fillRect(r.x - r.width/2, r.y, r.width, r.height);
        ctx.fillStyle = "#f8fafc";
        ctx.font = "bold 10px monospace";
        ctx.fillText(`RIS ${i}`, r.x - 18, r.y - 4);
    }

    // --- Aerial node ---
    ctx.fillStyle = "#38bdf8";
    ctx.beginPath();
    ctx.arc(scene.drone.x, scene.drone.y, scene.drone.radius, 0, 2 * Math.PI);
    ctx.fill();
    ctx.fillStyle = "#f8fafc";
    ctx.font = "bold 10px monospace";
    ctx.fillText(scene.drone.label, scene.drone.x - 35, scene.drone.y - 22);

    // --- Beam paths ---
    ctx.lineWidth = 2;

    // Always draw the LOS path (dashed grey when blocked,
    // magenta-bent when a reflector is on the path).
    if (linkState.los_blocked) {
        ctx.strokeStyle = "rgba(239, 68, 68, 0.35)";
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(scene.base.x, scene.base.y);
        ctx.lineTo(scene.drone.x, scene.drone.y);
        ctx.stroke();
        ctx.setLineDash([]);
    } else if (linkState.reflector_blocker) {
        // Reflector on the path: draw a bent LOS around the reflector.
        const rb = linkState.reflector_blocker;
        ctx.strokeStyle = "rgba(168, 85, 247, 0.75)";
        ctx.setLineDash([6, 3]);
        ctx.beginPath();
        ctx.moveTo(scene.base.x, scene.base.y);
        ctx.lineTo(rb.x, rb.y);
        ctx.lineTo(scene.drone.x, scene.drone.y);
        ctx.stroke();
        ctx.setLineDash([]);
    } else {
        ctx.strokeStyle = "#38bdf8";
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo(scene.base.x, scene.base.y);
        ctx.lineTo(scene.drone.x, scene.drone.y);
        ctx.stroke();
    }

    // Draw RIS path if it's the active route.
    if (isRisActive && linkState.best_ris_index >= 0) {
        const r = scene.ris[linkState.best_ris_index];
        ctx.strokeStyle = "#10b981";
        ctx.beginPath();
        ctx.moveTo(scene.base.x, scene.base.y);
        ctx.lineTo(r.x, r.y + r.height/2);
        ctx.lineTo(scene.drone.x, scene.drone.y);
        ctx.stroke();
    }

    // Status text on the canvas
    if (isDown) {
        ctx.fillStyle = "#ef4444";
        ctx.font = "bold 14px Arial";
        ctx.fillText("❌ LINK DOWN — NO CLEAR PATH", scene.drone.x - 220, scene.drone.y + 40);
    } else if (isRisActive) {
        ctx.fillStyle = "#f59e0b";
        ctx.font = "bold 14px Arial";
        ctx.fillText("↪ RIS REROUTE ACTIVE", scene.drone.x - 190, scene.drone.y + 40);
    } else {
        ctx.fillStyle = "rgba(16, 185, 129, 0.85)";
        ctx.font = "bold 14px Arial";
        ctx.fillText("✓ DIRECT LINE-OF-SIGHT", scene.drone.x - 190, scene.drone.y + 40);

        ctx.strokeStyle = "rgba(16, 185, 129, 0.5)";
        ctx.beginPath();
        ctx.arc(scene.drone.x, scene.drone.y + (innovationY * 4), 6, 0, 2 * Math.PI);
        ctx.stroke();
    }
}

// ==================================================================
// MOVING SCENE OBJECTS
// ==================================================================
function stepMovingObjects(dtSeconds) {
    const w = canvas.width  || 800;
    const h = canvas.height || 380;

    for (const b of scene.blockers) {
        stepOneObject(b, dtSeconds, w, h);
    }
    for (const z of scene.absorption_zones) {
        // Absorption zones move via their center; the polygon is
        // rebuilt each frame from the new center.
        const oldCx = z.cx, oldCy = z.cy;
        stepOneObject(z, dtSeconds, w, h, /*useCenter=*/true);
        if (z.cx !== oldCx || z.cy !== oldCy) {
            z.vertices = makeAbsorptionPolygon(z.cx, z.cy, z.rx, z.ry);
        }
    }
}

function stepOneObject(obj, dt, w, h, useCenter=false) {
    const px_per_m = PIXELS_PER_METER;
    const v_px_per_s = (obj.speed_mps || 0) * px_per_m;
    const dx = v_px_per_s * dt;

    const getX = () => useCenter ? obj.cx : obj.x;
    const getY = () => useCenter ? obj.cy : obj.y;
    const setX = (v) => { if (useCenter) obj.cx = v; else obj.x = v; };
    const setY = (v) => { if (useCenter) obj.cy = v; else obj.y = v; };

    switch (obj.motion) {
        case 'sweep': {
            const nx = getX() + dx * obj.sweep_dir;
            if (nx < obj.sweep_min) { setX(obj.sweep_min); obj.sweep_dir = 1; }
            else if (nx > obj.sweep_max) { setX(obj.sweep_max); obj.sweep_dir = -1; }
            else setX(nx);
            break;
        }
        case 'random_walk': {
            // Velocity perturbation, capped to keep it local.
            obj._vx = (obj._vx || 0) + (Math.random() - 0.5) * dx;
            obj._vy = (obj._vy || 0) + (Math.random() - 0.5) * dx;
            obj._vx *= 0.9;
            obj._vy *= 0.9;
            let nx = getX() + obj._vx;
            let ny = getY() + obj._vy;
            // Bounce off canvas edges
            if (nx < 20) { nx = 20; obj._vx = Math.abs(obj._vx); }
            if (nx > w - 20) { nx = w - 20; obj._vx = -Math.abs(obj._vx); }
            if (ny < 20) { ny = 20; obj._vy = Math.abs(obj._vy); }
            if (ny > h - 20) { ny = h - 20; obj._vy = -Math.abs(obj._vy); }
            setX(nx); setY(ny);
            break;
        }
        case 'orbit_base': {
            obj.phase += (v_px_per_s / Math.max(obj.orbit_radius, 1)) * dt;
            const cx = scene.base.x + Math.cos(obj.phase) * obj.orbit_radius;
            const cy = scene.base.y + Math.sin(obj.phase) * obj.orbit_radius;
            setX(cx); setY(cy);
            break;
        }
        case 'orbit_drone': {
            obj.phase += (v_px_per_s / Math.max(obj.orbit_radius, 1)) * dt;
            const cx = scene.drone.x + Math.cos(obj.phase) * obj.orbit_radius;
            const cy = scene.drone.y + Math.sin(obj.phase) * obj.orbit_radius;
            setX(cx); setY(cy);
            break;
        }
        case 'static':
        default:
            break;
    }
}

// ==================================================================
// CONTEXT MENU (right-click)
// ==================================================================
const contextMenuEl = document.getElementById('context-menu');
let contextMenuTarget = null;

function openContextMenu(clientX, clientY, obj) {
    contextMenuTarget = obj;
    contextMenuEl.innerHTML = buildMenuHTML(obj);
    contextMenuEl.style.left = clientX + 'px';
    contextMenuEl.style.top  = clientY + 'px';
    contextMenuEl.classList.add('open');
    attachMenuHandlers(obj);
}

function closeContextMenu() {
    contextMenuEl.classList.remove('open');
    contextMenuTarget = null;
}

function buildMenuHTML(obj) {
    // Identify object type
    let kind = 'unknown';
    if (obj === scene.base) kind = 'base';
    else if (obj === scene.drone) kind = 'drone';
    else if (scene.ris.includes(obj)) kind = 'ris';
    else if (scene.blockers.includes(obj)) kind = 'blocker';
    else if (scene.absorption_zones.includes(obj)) kind = 'absorption';

    const isBlocker   = kind === 'blocker';
    const isAbsorp    = kind === 'absorption';

    const infoKey = ({
        base: 'scene_base', drone: 'scene_drone', ris: 'scene_ris',
        blocker: 'scene_blocker', absorption: 'scene_absorption'
    })[kind] || 'spec_table';

    let html = '';
    html += `<div class="ctx-item" data-act="info" data-key="${infoKey}">ⓘ  About this object</div>`;

    if (isBlocker || isAbsorp) {
        const cur = obj.motion || 'static';
        const modes = [
            ['static',      'Static'],
            ['sweep',       'Linear sweep'],
            ['random_walk', 'Random walk'],
            ['orbit_base',  'Orbit base'],
            ['orbit_drone', 'Orbit drone']
        ];
        html += '<div class="ctx-section">Motion</div>';
        for (const [val, label] of modes) {
            const sel = (cur === val) ? ' ctx-selected' : '';
            html += `<div class="ctx-item ctx-radio${sel}" data-act="motion" data-val="${val}">${label}</div>`;
        }
        html += '<div class="ctx-slider-row">Speed <input type="range" id="ctx-speed" min="0.1" max="5" step="0.1" value="' + (obj.speed_mps || 1.2) + '"> <span id="ctx-speed-val">' + (obj.speed_mps || 1.2).toFixed(1) + ' m/s</span></div>';
    }

    if (isBlocker) {
        html += '<div class="ctx-slider-row">Radius <input type="range" id="ctx-radius" min="10" max="75" step="1" value="' + obj.radius + '"> <span id="ctx-radius-val">' + (obj.radius / PIXELS_PER_METER).toFixed(2) + ' m</span></div>';
        html += '<div class="ctx-section">Physics</div>';
        const curMode = obj.mode || 'occluder';
        const occ = curMode === 'occluder' ? ' ctx-selected' : '';
        const ref = curMode === 'reflector' ? ' ctx-selected' : '';
        html += `<div class="ctx-item ctx-radio${occ}" data-act="mode" data-val="occluder">Occluder (blocks path)</div>`;
        html += `<div class="ctx-item ctx-radio${ref}" data-act="mode" data-val="reflector">Reflector (bends path, −15 dB)</div>`;
    }

    if (isAbsorp) {
        html += '<div class="ctx-slider-row">Humidity <input type="range" id="ctx-humidity" min="10" max="90" step="1" value="' + obj.humidity_pct + '"> <span id="ctx-humidity-val">' + obj.humidity_pct + ' %</span></div>';
        html += '<div class="ctx-slider-row">Temperature <input type="range" id="ctx-temp" min="270" max="310" step="1" value="' + Math.round(obj.temp_k) + '"> <span id="ctx-temp-val">' + Math.round(obj.temp_k) + ' K</span></div>';
        html += '<div class="ctx-slider-row">Size <input type="range" id="ctx-size" min="20" max="120" step="1" value="' + Math.round(obj.rx) + '"> <span id="ctx-size-val">' + (obj.rx / PIXELS_PER_METER).toFixed(1) + ' m</span></div>';
    }

    html += '<div class="ctx-item ctx-danger" data-act="delete">Delete object</div>';
    return html;
}

function attachMenuHandlers(obj) {
    // Info links
    contextMenuEl.querySelectorAll('[data-act="info"]').forEach(el => {
        el.addEventListener('click', (ev) => {
            ev.stopPropagation();
            openInfoModal(el.getAttribute('data-key'));
            closeContextMenu();
        });
    });

    // Motion radio
    contextMenuEl.querySelectorAll('[data-act="motion"]').forEach(el => {
        el.addEventListener('click', (ev) => {
            ev.stopPropagation();
            obj.motion = el.getAttribute('data-val');
            closeContextMenu();
        });
    });

    // Physics mode radio
    contextMenuEl.querySelectorAll('[data-act="mode"]').forEach(el => {
        el.addEventListener('click', (ev) => {
            ev.stopPropagation();
            obj.mode = el.getAttribute('data-val');
            closeContextMenu();
        });
    });

    // Speed slider
    const sp = document.getElementById('ctx-speed');
    if (sp) sp.addEventListener('input', (ev) => {
        ev.stopPropagation();
        obj.speed_mps = parseFloat(ev.target.value);
        document.getElementById('ctx-speed-val').innerText = obj.speed_mps.toFixed(1) + ' m/s';
    });

    // Radius slider (blocker)
    const rd = document.getElementById('ctx-radius');
    if (rd) rd.addEventListener('input', (ev) => {
        ev.stopPropagation();
        obj.radius = parseFloat(ev.target.value);
        document.getElementById('ctx-radius-val').innerText = (obj.radius / PIXELS_PER_METER).toFixed(2) + ' m';
    });

    // Humidity slider (absorption)
    const hm = document.getElementById('ctx-humidity');
    if (hm) hm.addEventListener('input', (ev) => {
        ev.stopPropagation();
        obj.humidity_pct = parseFloat(ev.target.value);
        document.getElementById('ctx-humidity-val').innerText = obj.humidity_pct + ' %';
    });

    // Temperature slider
    const tp = document.getElementById('ctx-temp');
    if (tp) tp.addEventListener('input', (ev) => {
        ev.stopPropagation();
        obj.temp_k = parseFloat(ev.target.value);
        document.getElementById('ctx-temp-val').innerText = Math.round(obj.temp_k) + ' K';
    });

    // Size slider (absorption)
    const sz = document.getElementById('ctx-size');
    if (sz) sz.addEventListener('input', (ev) => {
        ev.stopPropagation();
        obj.rx = parseFloat(ev.target.value);
        obj.ry = obj.rx * 0.75;
        obj.vertices = makeAbsorptionPolygon(obj.cx, obj.cy, obj.rx, obj.ry);
        document.getElementById('ctx-size-val').innerText = (obj.rx / PIXELS_PER_METER).toFixed(1) + ' m';
    });

    // Delete
    contextMenuEl.querySelectorAll('[data-act="delete"]').forEach(el => {
        el.addEventListener('click', (ev) => {
            ev.stopPropagation();
            if (scene.blockers.includes(obj)) {
                scene.blockers = scene.blockers.filter(b => b !== obj);
            } else if (scene.absorption_zones.includes(obj)) {
                scene.absorption_zones = scene.absorption_zones.filter(z => z !== obj);
            }
            closeContextMenu();
        });
    });
}

canvas.addEventListener('contextmenu', (e) => {
    e.preventDefault();
    e.stopPropagation();
    const {x, y} = canvasCoords(e);
    const obj = hitTestForMenu(x, y);
    if (!obj) { closeContextMenu(); return; }
    openContextMenu(e.clientX, e.clientY, obj);
});

// Click anywhere else closes the menu. Right-clicks (button 2)
// are ignored here — the contextmenu handler above handles them,
// and closing on the same event would make the menu flash open
// and immediately vanish.
document.addEventListener('click', (ev) => {
    if (ev.button === 2) return;
    if (!contextMenuEl.contains(ev.target)) closeContextMenu();
});

// Escape also closes the menu.
document.addEventListener('keydown', (ev) => {
    if (ev.key === 'Escape') closeContextMenu();
});

// hitTest with absorption zones included (they aren't draggable so
// the existing hitTest ignores them).
function hitTestForMenu(mx, my) {
    // Blockers first
    for (const b of scene.blockers) {
        if (dist(mx, my, b.x, b.y) < b.radius) return b;
    }
    // Absorption zones (point-in-polygon)
    for (const z of scene.absorption_zones) {
        if (pointInPolygon(mx, my, z.vertices)) return z;
    }
    // RIS
    for (const r of scene.ris) {
        if (mx >= r.x - r.width/2 && mx <= r.x + r.width/2 &&
            my >= r.y && my <= r.y + r.height) return r;
    }
    // Drone
    if (dist(mx, my, scene.drone.x, scene.drone.y) < scene.drone.radius) return scene.drone;
    // Base
    if (Math.abs(mx - scene.base.x) < 20 && Math.abs(my - scene.base.y) < 20) return scene.base;
    return null;
}

// ==================================================================
// SIMULATION LOOP
// ==================================================================
let lastStepTime = performance.now();

async function runSimulationStep() {
    if (!isRunning) {
        requestAnimationFrame(runSimulationStep);
        return;
    }

    // Measure actual elapsed time since the previous step.
    const now = performance.now();
    let dtSeconds = (now - lastStepTime) / 1000;
    lastStepTime = now;

    // Clamp to sane bounds: never less than 10 ms, never more
    // than 200 ms. Protects against tab-throttling and stalls.
    if (!isFinite(dtSeconds) || dtSeconds < 0.01) dtSeconds = 0.06;
    if (dtSeconds > 0.2) dtSeconds = 0.2;

    // Advance any moving scene objects first, so the link state
    // reflects their current positions.
    stepMovingObjects(dtSeconds);

    const linkState = computeLinkState();

    const payload = {
        jitter: parseFloat(jitterSlider.value),
        sway: parseFloat(swaySlider.value),
        los_blocked: linkState.los_blocked,
        ris_blocked: linkState.ris_blocked,
        los_distance_m: linkState.los_distance_m,
        ris_d1_m: linkState.ris_d1_m,
        ris_d2_m: linkState.ris_d2_m,
        ris_count: linkState.ris_count,
        bandwidth_hz: parseFloat(bandwidthSlider.value) * 1e9,
        dt: dtSeconds,
        session_id: getSessionId(),
        window_size: parseInt(windowSizeSelect.value, 10) || 10,

        absorption_loss_los_db:     linkState.absorption_loss_los_db,
        absorption_loss_ris_db:     linkState.absorption_loss_ris_db,
        reflector_los_penalty_db:   linkState.reflector_los_penalty_db,
        reflector_ris_penalty_db:   linkState.reflector_ris_penalty_db
    };

    try {
        const response = await fetch('/api/kalman-step', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        const trueBeamPosition = data.true_pos;
        const x_state = data.est_pos;
        const p_cov = data.p_cov;
        const currentR = data.r_matrix;
        const currentSnr = data.snr;
        const blockageProb = data.prob;
        const kalmanGainK = data.gain;
        const targetAngle = data.target_angle;
        const innovationY = data.innovation_y;

        truePositionData.shift(); truePositionData.push(trueBeamPosition);
        estimatedPositionData.shift(); estimatedPositionData.push(x_state);
        covarianceData.shift(); covarianceData.push(p_cov * 5);
        noiseRData.shift(); noiseRData.push(currentR);

        // Note: snr_los and snr_ris already include absorption and
        // reflector penalties — the backend applies them before
        // returning. The chart therefore shows the *effective* SNR
        // on each path, which is what the handover decision uses.
        snrLosData.shift();    snrLosData.push(data.snr_los);
        snrRisData.shift();    snrRisData.push(data.snr_ris);
        snrActiveData.shift(); snrActiveData.push(currentSnr);
        nEffData.shift();      nEffData.push(data.n_eff);

        chartTracking.update('none');
        chartCovariance.update('none');
        chartSnr.update('none');
        chartNeff.update('none');

        document.getElementById('ui-snr').innerText = currentSnr.toFixed(2) + " dB";

        // Show the probability plus a small tag saying which head
        // produced it: GRU (neural net) or sigma (sigmoid fallback).
        const srcTag = data.prob_source === "GRU" ? "GRU" : "σ";
        document.getElementById('ui-prob').innerText =
            blockageProb.toFixed(3) + "  " + srcTag;
        document.getElementById('ui-prob').style.color =
            data.prob_source === "GRU" ? "#22d3ee" : "#eab308";

        // Reflect the effective window the server actually used.
        // If it's smaller than the trained seq_len (16), the model
        // is running on padded input — show that honestly.
        if (typeof data.effective_window === 'number') {
            const n = data.effective_window;
            const tag = n < 16 ? ` (padded → 16)` : '';
            document.getElementById('window-size-val').innerText = n + tag;
        }

        const shortLink =
            data.active_link === "DIRECT LINE-OF-SIGHT" ? "LOS" :
            data.active_link === "RIS REFLECTED NODE ACTIVE" ? "RIS" :
            data.active_link === "LINK DOWN" ? "DOWN" :
            data.active_link;
        document.getElementById('ui-link').innerText = shortLink;
        document.getElementById('ui-link').style.color =
            shortLink === "RIS" ? "#f43f5e" :
            shortLink === "DOWN" ? "#ef4444" : "#a855f7";
        document.getElementById('ui-gain').innerText = kalmanGainK.toFixed(4);
        document.getElementById('ui-omega').innerText = data.ekf_omega.toFixed(4) + " rad/s";
        document.getElementById('ui-range').innerText = data.ekf_R.toFixed(2) + " m";
        document.getElementById('ui-neff').innerText = data.n_eff.toFixed(3);
        document.getElementById('ui-neff').style.color =
            data.n_eff > 3.5 ? '#ef4444' :
            data.n_eff > 2.5 ? '#eab308' : '#10b981';

        drawScene(targetAngle, innovationY, linkState, blockageProb, data.active_link);

    } catch (error) {
        console.error("[-] Telemetry synchronization API pipeline down:", error);
    }

    setTimeout(() => { requestAnimationFrame(runSimulationStep); }, 60);
}

runSimulationStep();
