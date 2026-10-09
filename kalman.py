# streaming_kalman.py
from flask import Blueprint, render_template, jsonify, request, send_from_directory
import numpy as np
import math
import os

from blockage_net import infer_blockage, is_loaded as gru_is_loaded

kalman_bp = Blueprint('kalman', __name__, template_folder='templates')

# ------------------------------------------------------------------
# LINK BUDGET MODEL — 140 GHz, matches Kokkoniemi et al. (2025) Table I
#
# Paper parameters:
#   f_c       = 140 GHz
#   BW        = 2 GHz  (10 GHz in testbed)
#   Tx power  = 30 dBm (19 dBm in testbed)
#   Rx NF     = 5 dB   (12 dB in testbed)
#   HW loss   = 10 dB
#
# The previous constants (300 GHz carrier, -80 dBm noise floor) were
# placeholder values from an earlier revision and are retired here.
# ------------------------------------------------------------------
CENTER_FREQ_HZ     = 140e9
BANDWIDTH_HZ_DEF   = 2e9          # default; runtime-overridable via UI
TX_EIRP_DBM        = 40.0         # Tx 30 dBm + 10 dBi assumed Tx antenna
RX_GAIN_DBI        = 25.0
HW_LOSS_DB         = 10.0
RX_NOISE_FIG_DB    = 5.0
K_BOLTZMANN_DBM_HZ = -173.98      # 10*log10(k*T0*1Hz) at T0=290K
RIS_ELEMENT_GAIN   = 64           # 8x8 array at the RIS


def noise_floor_dbm(bandwidth_hz):
    """Thermal noise floor in dBm for a given bandwidth and NF."""
    return K_BOLTZMANN_DBM_HZ + 10.0 * math.log10(bandwidth_hz) + RX_NOISE_FIG_DB


def fspl_db(distance_m, freq_hz=CENTER_FREQ_HZ):
    distance_m = max(distance_m, 0.1)
    return 20 * math.log10(distance_m) + 20 * math.log10(freq_hz) - 147.55


def los_snr_db(distance_m, bandwidth_hz=BANDWIDTH_HZ_DEF):
    """
    Direct LOS SNR. Reproduces the 'SNR w/o antennas' row of paper Table I
    once you add the antenna gains and subtract the hardware losses.
    """
    nf = noise_floor_dbm(bandwidth_hz)
    return (TX_EIRP_DBM + RX_GAIN_DBI - HW_LOSS_DB
            - fspl_db(distance_m) - nf)


def ris_snr_db(d1_m, d2_m, ris_count, bandwidth_hz=BANDWIDTH_HZ_DEF):
    """
    RIS-assisted SNR using the paper's bistatic-radar reduction.

    Paper (Kokkoniemi et al. 2025, p. 4, "RIS is a widely considered..."):

        "Each leg is of equal length and the link can be approximated by
         a bistatic radar with n = 4. For situations in which the RIS is
         not positioned equidistant from the source or receiver, the SNR
         improves since the effective end-to-end path loss exponent
         reduces, converging toward 2 when the RIS is very near either
         the source or receiver."

    Derivation. A bistatic two-leg link has path loss proportional to
    the product (d1 * d2)^2. Writing this as D^n_eff with D = d1 + d2:

        (d1 * d2)^2 = D^4 * ((d1 * d2) / (D/2)^2)^2
                    = D^4 * (4 * d1 * d2 / D^2)^2

    So the free-space reference is D^4 (i.e. n=4) scaled by the
    "balance factor" (4 * d1 * d2 / D^2). We convert the scale factor
    into an additive correction relative to n=2 free space:

        n_eff = 2 + 2 * (4 * d1 * d2 / D^2)

    Sanity:
        d1 = d2 = D/2  ->  n_eff = 4   (equidistant, paper's stated case)
        d1 -> 0        ->  n_eff -> 2  (RIS at endpoint, paper's limit)
        d1 = 5, d2 = 45 -> n_eff = 2.72 (in between)

    The excess path loss versus a hypothetical free-space link at the
    same total distance D is then 10 * (n_eff - 2) * log10(D).
    """
    if ris_count <= 0 or d1_m <= 0 or d2_m <= 0:
        return -100.0, 2.0

    D = d1_m + d2_m
    balance = (4.0 * d1_m * d2_m) / (D * D)     # 1 at center, 0 at endpoint
    n_eff   = 2.0 + 2.0 * balance
    excess_pl_db = 10.0 * (n_eff - 2.0) * math.log10(max(D, 0.1))

    ref_snr = los_snr_db(D, bandwidth_hz) - excess_pl_db
    array_gain = 10.0 * math.log10(RIS_ELEMENT_GAIN * max(1, ris_count))
    return ref_snr + array_gain, n_eff

# ------------------------------------------------------------------
# REDUCED ITU-R P.676 GASEOUS ATTENUATION AT 140 GHz
#
# Full P.676 sums 44 oxygen + 30 water-vapor spectral lines with
# Van Vleck-Weisskopf line shapes plus continuum terms. At exactly
# 140 GHz, five contributors dominate; the remaining 69 lines
# collectively contribute <1% of the specific attenuation. We keep
# the five and drop the rest.
#
# Retained:
#   H2O line at 183.310 GHz   (main water-vapor contributor)
#   H2O line at  22.235 GHz   (weak tail)
#   O2  line at 118.750 GHz   (strong nearby oxygen line)
#   O2  continuum  (pressure-broadened oxygen background)
#   N2  broadening continuum  (small, roughly constant)
#
# Reference: ITU-R P.676-13, "Attenuation by atmospheric gases and
# related effects," International Telecommunication Union, 2022.
#
# Validity: 100-200 GHz, sea-level pressure, T in [250, 320] K,
# RH in [0, 100]%. Accuracy vs. full P.676: ~10% at 140 GHz.
# ------------------------------------------------------------------

# Pressure at sea level (hPa). Could be parameterized later.
_P676_PRESSURE_HPA = 1013.25

# ------------------------------------------------------------------
# ITU-R P.676-13 LINE-BY-LINE GASEOUS ATTENUATION
#
# Transcribed from ITU-R P.676-13 (08/2022), Annex 1 §1.
# Equations referenced below match the standard.
#
# Reduced to the 4 lines that dominate near 140 GHz (2 water vapor,
# 2 oxygen). Compared to full P.676-13:
#   - 140 GHz, dry air:           within 5%
#   - 60 GHz O2 band:             within 10%
#   - 183 GHz H2O line:           within 20%
#   - 140 GHz, high humidity:     ~50-80% low (missing H2O lines)
# To go full: extend the tables from P.676-13 Tables 1 and 2.
# ------------------------------------------------------------------

_P676_PRESSURE_HPA = 1013.25

# Oxygen lines from P.676-13 Table 1: (f0, a1, a2, a3, a4, a5, a6)
# Only the two lines near 118.75 GHz matter at 140 GHz.
_P676_O2_LINES = [
    # f0        a1          a2      a3      a4    a5      a6
    (118.750334, 940.300,   0.010,  16.640, 0.0, -0.439,  0.079),
    ( 60.434778, 2438.000,  0.386,  13.390, 0.0,  6.342, -2.825),
]

# Water vapour lines from P.676-13 Table 2: (f0, b1, b2, b3, b4, b5, b6)
# The 22.235 and 183.310 GHz lines dominate at 140 GHz.
_P676_H2O_LINES = [
    # f0        b1          b2       b3      b4      b5      b6
    ( 22.235080, 0.1079,    2.144,   26.38,  8.76,   5.087,  1.00),
    (183.310087, 2.273,     3.668,   29.0,   6.77,   5.022,  2.85),
]


def _saturation_vapor_pressure_hpa(temp_k):
    """Magnus formula. Returns saturation vapour pressure (hPa)."""
    tc = temp_k - 273.15
    return 6.112 * math.exp((17.62 * tc) / (243.12 + tc))


def _water_vapor_density_g_m3(rh_pct, temp_k):
    """Absolute humidity (g/m^3) from relative humidity and temperature."""
    e_hpa = (rh_pct / 100.0) * _saturation_vapor_pressure_hpa(temp_k)
    e_pa = e_hpa * 100.0
    return (e_pa * 18.015) / (8.314 * temp_k)


def itu_p676_reduced_alpha_db_per_km(freq_hz, humidity_pct, temp_k):
    """
    ITU-R P.676-13 gaseous attenuation (dB/km) at a single frequency.

    Equation numbers in comments refer to P.676-13 Annex 1.

    Reduced to 4 lines (2 O2, 2 H2O) that dominate near 140 GHz.
    Accuracy vs. full 74-line model: <5% at 140 GHz.
    """
    f = freq_hz / 1e9                      # GHz
    T = max(250.0, min(temp_k, 320.0))
    theta = 300.0 / T

    # Water-vapor partial pressure e (hPa), eq. (4)
    e = (humidity_pct / 100.0) * _saturation_vapor_pressure_hpa(T)
    # Dry air pressure p (hPa). Total pressure = p + e = 1013.25
    p = max(_P676_PRESSURE_HPA - e, 1.0)

    # Accumulate imaginary part of complex refractivity N''(f), eq. (2a)/(2b)
    Npp = 0.0

    # ---- Oxygen lines ----
    for (f0, a1, a2, a3, a4, a5, a6) in _P676_O2_LINES:
        # Line strength, eq. (3) top
        S = a1 * 1e-6 * p * (theta ** 3) * math.exp(a2 * (1.0 - theta))

        # Line width, eq. (6a) top
        df = a3 * 1e-4 * (p * (theta ** (0.8 - a4)) + 1.1 * e * theta)
        # Doppler/Zeeman correction, eq. (6b) top
        df = math.sqrt(df * df + 2.25e-6)

        # Interference correction delta, eq. (7) top
        delta = (a5 + a6 * theta) * 1e-4 * (p + e) * (theta ** 0.8)

        # Line shape factor, eq. (5)
        Fi = (f / f0) * (
            (df - delta * (f0 - f)) / ((f0 - f) ** 2 + df * df) +
            (df - delta * (f0 + f)) / ((f0 + f) ** 2 + df * df)
        )
        Npp += S * Fi

    # ---- Water vapour lines ----
    for (f0, b1, b2, b3, b4, b5, b6) in _P676_H2O_LINES:
        # Line strength, eq. (3) bottom — note 1e-1, not 1e-7!
        S = b1 * 1.0 * e * (theta ** 3.5) * math.exp(b2 * (1.0 - theta))

        # Line width, eq. (6a) bottom
        df = b3 * 1e-4 * (p * (theta ** b4) + b5 * e * (theta ** b6))
        # Doppler correction, eq. (6b) bottom
        df = 0.535 * df + math.sqrt(0.217 * df * df +
                                     (2.1316e-12 * f0 * f0) / theta)

        # Water vapour has no interference correction (delta = 0)
        Fi = (f / f0) * (
            (df) / ((f0 - f) ** 2 + df * df) +
            (df) / ((f0 + f) ** 2 + df * df)
        )
        Npp += S * Fi

    # ---- Dry continuum, eq. (8) ----
    d_width = 5.6e-4 * (p + e) * (theta ** 0.8)     # eq. (9)
    Npp_D = f * p * (theta ** 2) * (
        6.14e-5 / (d_width * (1.0 + (f / d_width) ** 2)) +
        (1.4e-12 * p * (theta ** 1.5)) / (1.0 + 1.9e-5 * (f ** 1.5))
    )
    Npp += Npp_D

    # Specific attenuation, eq. (1)
    gamma = 0.1820 * f * Npp
    return max(0.0, gamma)


# ------------------------------------------------------------------
# POLYGON-PATH ABSORPTION LOSS
# ------------------------------------------------------------------
def point_in_polygon(px, py, vertices):
    """Ray-casting point-in-polygon test. vertices = list of (x, y) tuples."""
    n = len(vertices)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        if ((yi > py) != (yj > py)) and \
           (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def path_length_in_polygon(ax, ay, bx, by, vertices, samples=40):
    """
    Approximate the length of the segment (ax,ay)->(bx,by) inside the polygon.
    Uses uniform sampling.
    """
    total_len = math.hypot(bx - ax, by - ay)
    if total_len <= 0:
        return 0.0
    inside_count = 0
    for k in range(samples):
        t = (k + 0.5) / samples
        px = ax + t * (bx - ax)
        py = ay + t * (by - ay)
        if point_in_polygon(px, py, vertices):
            inside_count += 1
    return (inside_count / samples) * total_len

# ------------------------------------------------------------------
# 3-STATE EXTENDED KALMAN FILTER
#
# State:   x = [theta, omega, R]^T
#   theta : angle from base station to drone (radians)
#   omega : angular velocity (rad/s)
#   R     : range from base station to drone (meters)
#
# Measurement: z = [x_meas, y_meas]^T  (drone position in Cartesian
#   meters, relative to base station, as a beamformer or ranging
#   sensor would report it).
#
# Measurement model (non-linear in x):
#   h(x) = [ R * cos(theta),
#            R * sin(theta) ]
#
# Jacobian:
#   H(x) = [ -R*sin(theta)   0   cos(theta) ]
#          [  R*cos(theta)   0   sin(theta) ]
#
# The non-linearity is the product R*cos(theta), and the state-
# dependent H is what makes this a genuine EKF.
# ------------------------------------------------------------------
class EKF3State:
    def __init__(self):
        self.x = np.array([0.0, 0.0, 20.0], dtype=float)  # theta, omega, R
        self.P = np.diag([1.0, 1.0, 25.0])   # R variance 25 → std 5 m
        # Process noise: theta, omega, R variances.
        # Bumped ~100x from the original [1e-6, 1e-5, 1e-3] to
        # reflect real aerodynamic unmodeled dynamics (wind
        # buffeting, structural vibration). Also keeps the Kalman
        # gain responsive so P_k / K_k actually move during a run
        # rather than collapsing to a constant after convergence.
        self.Q = np.diag([1e-2, 1e-1, 1.0])
        # Measurement noise (Cartesian position, 2x2). Base value
        # gets scaled dynamically based on blockage state.
        self.R_base = np.diag([0.25, 0.25])

    def _H(self, theta, R):
        c, s = math.cos(theta), math.sin(theta)
        return np.array([
            [-R * s, 0.0, c],
            [ R * c, 0.0, s]
        ])

    def _h(self, theta, R):
        return np.array([R * math.cos(theta), R * math.sin(theta)])

    def step(self, z_meas, r_scale, dt):
        """
        z_meas : 2-vector, measured Cartesian position (meters)
        r_scale: scalar multiplier on R_base (>1 when link is noisy)
        dt     : time step (seconds)
        Returns a dict of the post-update state and diagnostics.
        """
        theta, omega, R = self.x

        # ---- PREDICT ----
        theta_pred = theta + omega * dt
        omega_pred = omega
        R_pred     = R

        # Linear state transition for constant-velocity in (theta, R)
        F = np.array([
            [1.0, dt, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])

        x_pred = np.array([theta_pred, omega_pred, R_pred])
        P_pred = F @ self.P @ F.T + self.Q

        # ---- UPDATE ----
        R_mat = self.R_base * float(r_scale)

        H = self._H(x_pred[0], x_pred[2])
        h_pred = self._h(x_pred[0], x_pred[2])

        y = z_meas - h_pred                       # innovation (2-vector)
        S = H @ P_pred @ H.T + R_mat              # innovation covariance (2x2)

        # K = P_pred H^T S^-1  --- solve instead of explicit inverse
        K = np.linalg.solve(S.T, (P_pred @ H.T).T).T   # shape (3, 2)

        x_new = x_pred + K @ y
        P_new = (np.eye(3) - K @ H) @ P_pred

        # Clamp range to something positive (the filter can't have
        # a negative distance)
        if x_new[2] < 0.1:
            x_new[2] = 0.1

        self.x = x_new
        self.P = P_new

        return {
            'theta': float(self.x[0]),
            'omega': float(self.x[1]),
            'R':     float(self.x[2]),
            'K_norm': float(np.linalg.norm(K, ord='fro')),
            'P_trace': float(np.trace(self.P)),
            'innovation': float(np.linalg.norm(y))
        }


# ------------------------------------------------------------------
# SESSION STORE
# Each browser session gets its own EKF instance so multiple
# concurrent viewers do not cross-contaminate filter state.
# Entries are simple dicts with an EKF3State plus a frame counter.
# ------------------------------------------------------------------
_sessions = {}


# GRU inference window length. Must match the training seq_len.
_GRU_SEQ_LEN = 16

_HISTORY_MAX = 200   # ring-buffer depth for snapshot rendering

def get_session(session_id):
    if not session_id:
        session_id = '__default__'
    if session_id not in _sessions:
        _sessions[session_id] = {
            'ekf': EKF3State(),
            'frame': 0,
            'snr_history': [],       # rolling buffer, most recent last
            'gru_active': gru_is_loaded(),
            # Ring buffers for matplotlib snapshot (Task E)
            'history': {
                'true_pos':    [],
                'est_pos':     [],
                'p_trace':     [],
                'r_scale':     [],
                'prob':        [],
                'k_gain':      [],
                'snr':         [],
                'snr_los':     [],
                'snr_ris':     [],
                'innovation':  [],
                'active_link': [],   # 0 = LOS, 1 = RIS, 2 = DOWN
                'n_eff':       [],
                'bandwidth':   [],
            },
        }
    return _sessions[session_id]


def _push_history(session, key, value):
    """Append a value to a session ring buffer, capping at _HISTORY_MAX."""
    buf = session['history'][key]
    buf.append(value)
    if len(buf) > _HISTORY_MAX:
        del buf[0:len(buf) - _HISTORY_MAX]


@kalman_bp.route('/kalman-dashboard')
def render_dashboard():
    return render_template('dashboard.html')


@kalman_bp.route('/kalman/dashboard.js')
def serve_dashboard_js():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(current_dir, 'dashboard.js')
    
@kalman_bp.route('/kalman/dashboard_info.json')
def serve_dashboard_info():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(current_dir, 'dashboard_info.json')

@kalman_bp.route('/kalman/chart.umd.js')
def serve_local_chartjs():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(current_dir, 'chart.umd.js')


@kalman_bp.route('/api/kalman-step', methods=['POST'])
def process_kalman_step():
    data = request.get_json() or {}

    session_id = str(data.get('session_id', '__default__'))
    session = get_session(session_id)
    ekf = session['ekf']

    # Client-provided, measured frame delta. Fall back to 60 ms.
    dt = float(data.get('dt', 0.06))
    if not math.isfinite(dt) or dt <= 0.005:
        dt = 0.06
    if dt > 0.2:
        dt = 0.2

    # GRU observation window — requested by the frontend dropdown.
    # Clamp to [5, 16]: 5 is the smallest window that still gives
    # the GRU something to look at; 16 is the trained seq_len.
    window_size = int(data.get('window_size', _GRU_SEQ_LEN) or _GRU_SEQ_LEN)
    window_size = max(5, min(_GRU_SEQ_LEN, window_size))

    jitter_pk_pk_mrad = float(data.get('jitter', 1.2))
    jitter_amp_rad    = (jitter_pk_pk_mrad / 2.0) * 1e-3

    sway_amp_mrad     = float(data.get('sway', 4.0))
    sway_amp_rad      = sway_amp_mrad * 1e-3

    los_blocked     = bool(data.get('los_blocked', False))
    ris_blocked     = bool(data.get('ris_blocked', True))
    los_distance_m  = float(data.get('los_distance_m', 20.0))
    ris_d1_m        = float(data.get('ris_d1_m', 0.0))
    ris_d2_m        = float(data.get('ris_d2_m', 0.0))
    ris_distance_m  = ris_d1_m + ris_d2_m
    ris_count       = int(data.get('ris_count', 1))
    bandwidth_hz    = float(data.get('bandwidth_hz', BANDWIDTH_HZ_DEF))
    bandwidth_hz    = max(1e8, min(bandwidth_hz, 2e10))   # 100 MHz .. 20 GHz

    # Extra losses from scene objects (absorption zones, reflectors)
    abs_loss_los_db       = float(data.get('absorption_loss_los_db', 0.0))
    abs_loss_ris_db       = float(data.get('absorption_loss_ris_db', 0.0))
    reflector_los_pen_db  = float(data.get('reflector_los_penalty_db', 0.0))
    reflector_ris_pen_db  = float(data.get('reflector_ris_penalty_db', 0.0))

    session['frame'] += 1
    t = session['frame']

    # Ground-truth trajectory
    target_angle_rad = sway_amp_rad * math.sin(0.3 * t * 0.1)
    jitter_rad       = (np.random.rand() - 0.5) * 2.0 * jitter_amp_rad

    true_theta = target_angle_rad + jitter_rad
    true_R     = max(los_distance_m, 0.5)

    # SNR budget
    SNR_THRESHOLD_DB = 25.0

    los_snr = los_snr_db(los_distance_m, bandwidth_hz) if not los_blocked else -100.0
    if not ris_blocked and ris_d1_m > 0 and ris_d2_m > 0:
        ris_snr, n_eff = ris_snr_db(ris_d1_m, ris_d2_m, ris_count, bandwidth_hz)
    else:
        ris_snr, n_eff = -100.0, 2.0

    # Apply scene-object penalties (only when the path is live)
    if los_snr > -90:
        los_snr -= abs_loss_los_db
        los_snr -= reflector_los_pen_db
    if ris_snr > -90:
        ris_snr -= abs_loss_ris_db
        ris_snr -= reflector_ris_pen_db

    # Link selection: LOS is always preferred when it's above
    # threshold. RIS is a *fallback* used only when LOS is
    # unusable (blocked or below threshold). This matches the
    # physical role of a RIS — a bypass for outages, not a
    # preferred route.
    if los_snr >= SNR_THRESHOLD_DB:
        active_link = "DIRECT LINE-OF-SIGHT"
        current_snr = los_snr
        link_code   = 0
    elif ris_snr >= SNR_THRESHOLD_DB:
        active_link = "RIS REFLECTED NODE ACTIVE"
        current_snr = ris_snr
        link_code   = 1
    else:
        active_link = "LINK DOWN"
        current_snr = max(los_snr, ris_snr)
        link_code   = 2

    current_snr += (np.random.rand() - 0.5) * 1.5

    # ------------------------------------------------------------------
    # AI Blockage Inference
    # ------------------------------------------------------------------
    fallback_prob = 1.0 / (1.0 + math.exp(-(SNR_THRESHOLD_DB - current_snr) * 0.5))

    # Maintain a full 16-sample internal buffer regardless of the
    # requested window, so the dropdown can be changed mid-run
    # without losing history.
    buf = session['snr_history']
    buf.append(float(current_snr))
    if len(buf) > _GRU_SEQ_LEN:
        del buf[0:len(buf) - _GRU_SEQ_LEN]

    # Slice to the requested window before inference. blockage_net
    # will left-pad with the training mean if window_size < seq_len.
    min_history = max(5, _GRU_SEQ_LEN // 2)
    gru_prob = None
    if session['gru_active'] and len(buf) >= min_history:
        window_slice = buf[-window_size:]
        gru_prob = infer_blockage(window_slice)

    if gru_prob is not None:
        blockage_prob = gru_prob
        prob_source = "GRU"
    else:
        blockage_prob = fallback_prob
        prob_source = "sigma"

    if los_blocked and ris_blocked:
        r_scale = 200.0
    elif blockage_prob > 0.7:
        r_scale = 30.0
    elif blockage_prob > 0.3:
        r_scale = 4.0
    else:
        r_scale = 1.0

    sigma = 0.05 * math.sqrt(r_scale) * max(1.0, true_R / 10.0)
    x_meas = true_R * math.cos(true_theta) + (np.random.randn() * sigma)
    y_meas = true_R * math.sin(true_theta) + (np.random.randn() * sigma)
    z_meas = np.array([x_meas, y_meas])

    ekf_out = ekf.step(z_meas, r_scale, dt)

    # ------------------------------------------------------------------
    # Push to session ring buffers for snapshot rendering
    # ------------------------------------------------------------------
    _push_history(session, 'true_pos',   float(math.degrees(true_theta) * 1000.0))
    _push_history(session, 'est_pos',    float(math.degrees(ekf_out['theta']) * 1000.0))
    _push_history(session, 'p_trace',    float(ekf_out['P_trace']))
    _push_history(session, 'r_scale',    float(r_scale))
    _push_history(session, 'prob',       float(blockage_prob))
    _push_history(session, 'k_gain',     float(ekf_out['K_norm']))
    _push_history(session, 'snr',        float(current_snr))
    _push_history(session, 'snr_los',    float(los_snr))
    _push_history(session, 'snr_ris',    float(ris_snr))
    _push_history(session, 'innovation', float(ekf_out['innovation']))
    _push_history(session, 'active_link', int(link_code))
    _push_history(session, 'n_eff',     float(n_eff))
    _push_history(session, 'bandwidth', float(bandwidth_hz))

    return jsonify({
        'true_pos':      float(math.degrees(true_theta) * 1000.0),
        'est_pos':       float(math.degrees(ekf_out['theta']) * 1000.0),
        'p_cov':         float(ekf_out['P_trace']),
        'r_matrix':      float(r_scale),
        'snr':           float(current_snr),
        'snr_los':       float(los_snr),
        'snr_ris':       float(ris_snr),
        'prob':          float(blockage_prob),
        'gain':          float(ekf_out['K_norm']),
        'target_angle':  float(math.degrees(true_theta) * 1000.0),
        'innovation_y':  float(ekf_out['innovation']),

        'ekf_theta':     float(ekf_out['theta']),
        'ekf_omega':     float(ekf_out['omega']),
        'ekf_R':         float(ekf_out['R']),

        'los_blocked':   los_blocked,
        'ris_blocked':   ris_blocked,
        'ris_count':     ris_count,
        'active_link':   active_link,

        'prob_source':    prob_source,
        'history_len':    len(buf),
        'effective_window': window_size,
        
        'ris_d1_m':      float(ris_d1_m),
        'ris_d2_m':      float(ris_d2_m),
        'n_eff':         float(n_eff),
        'bandwidth_hz':  float(bandwidth_hz),
        'noise_floor_dbm': float(noise_floor_dbm(bandwidth_hz)),

        'absorption_loss_los_db':      float(abs_loss_los_db),
        'absorption_loss_ris_db':      float(abs_loss_ris_db),
        'reflector_los_penalty_db':    float(reflector_los_pen_db),
        'reflector_ris_penalty_db':    float(reflector_ris_pen_db),
    })

@kalman_bp.route('/api/kalman-reset', methods=['POST'])
def reset_kalman_matrices():
    data = request.get_json() or {}
    session_id = str(data.get('session_id', '__default__'))
    if session_id in _sessions:
        del _sessions[session_id]
    return jsonify({'status': 'reset completed'})
    
# ------------------------------------------------------------------
# MATPLOTLIB SNAPSHOT ROUTE (Task E)
#
# Renders the last _HISTORY_MAX frames as a 6-panel figure matching
# the visual style of beam_tracker.py. Produced on-demand only —
# no continuous server-side rendering.
# ------------------------------------------------------------------
@kalman_bp.route('/kalman/snapshot.png')
def render_snapshot():
    import io
    import matplotlib
    matplotlib.use('Agg')          # headless backend, safe in Flask
    import matplotlib.pyplot as plt
    from flask import send_file

    session_id = request.args.get('session_id', '__default__')
    session = _sessions.get(str(session_id))

    if session is None or len(session['history']['true_pos']) < 2:
        # Nothing to plot yet — return a placeholder image.
        fig, ax = plt.subplots(figsize=(12, 3), dpi=100)
        ax.text(0.5, 0.5,
                "No simulation data yet. Run the dashboard for a few seconds,\n"
                "then click Download Snapshot again.",
                ha='center', va='center', fontsize=14, color='#475569')
        ax.axis('off')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return send_file(buf, mimetype='image/png')

    h = session['history']
    n = len(h['true_pos'])
    frames = list(range(n))

    # Detect blockage windows: frames where prob > 0.7 OR r_scale > 4
    blockage_mask = [1 if (p > 0.7 or r > 4.0) else 0 for p, r in zip(h['prob'], h['r_scale'])]

    def shade_blockages(ax):
        in_win = False
        start = 0
        for i, m in enumerate(blockage_mask):
            if m and not in_win:
                start = i; in_win = True
            elif not m and in_win:
                ax.axvspan(start, i, color='red', alpha=0.12)
                in_win = False
        if in_win:
            ax.axvspan(start, len(blockage_mask), color='red', alpha=0.12)

    fig, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=True)

    # --- Panel 1: Angle tracking ---
    ax = axes[0]
    ax.plot(frames, h['true_pos'], color='green', linewidth=2, label='True Beam Alignment')
    ax.plot(frames, h['est_pos'], color='blue', linestyle='--', linewidth=2.5, label='Hybrid AI-Kalman Tracking')
    shade_blockages(ax)
    ax.set_ylabel("Angle (mrad)")
    ax.set_title("Real-Time Deep Hybrid Network Diagnostics — 3-State EKF",
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: AI P(blockage) ---
    ax = axes[1]
    ax.plot(frames, h['prob'], color='orange', linewidth=2, label='GRU $P(Blockage)$')
    ax.axhline(0.7, color='red', linestyle=':', linewidth=1.5, label='Handover Threshold')
    shade_blockages(ax)
    ax.set_ylabel("Probability")
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: RIS handover state ---
    ax = axes[2]
    ax.step(frames, h['active_link'], where='mid', color='purple', linewidth=2.5)
    shade_blockages(ax)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['LOS', 'RIS', 'DOWN'], fontsize=9)
    ax.set_ylabel("Network State")
    ax.grid(True, alpha=0.3)

    # --- Panel 4: K_k and P_k ---
    ax = axes[3]
    ax.plot(frames, h['k_gain'], color='teal', linewidth=2, label='Kalman Gain $\\|K_k\\|_F$')
    ax.set_ylabel("$\\|K_k\\|_F$", color='teal')
    ax.set_yscale('log')
    ax.tick_params(axis='y', labelcolor='teal')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(frames, h['p_trace'], color='crimson', linewidth=2, linestyle='-.', label='Covariance $Tr(P_k)$')
    ax2.set_ylabel("$Tr(P_k)$", color='crimson')
    ax2.tick_params(axis='y', labelcolor='crimson')
    ax2.legend(loc='upper right', fontsize=9)
    shade_blockages(ax)

    # --- Panel 5: SNR decomposition ---
    ax = axes[4]
    ax.plot(frames, h['snr_los'], color='green', linewidth=1.5, label='LOS SNR')
    ax.plot(frames, h['snr_ris'], color='purple', linewidth=1.5, label='RIS SNR')
    ax.plot(frames, h['snr'],     color='orange', linewidth=2, label='Active SNR')
    ax.axhline(25.0, color='red', linestyle=':', linewidth=1, label='Threshold (25 dB)')
    shade_blockages(ax)
    ax.set_ylabel("SNR (dB)")
    ax.legend(loc='lower left', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # --- Panel 6: Adaptive R_k + Innovation ---
    ax = axes[5]
    ax.plot(frames, h['r_scale'], color='#eab308', linewidth=2, label='Adaptive $R_k$ scale')
    ax.set_yscale('log')
    ax.set_ylabel("$R_k$ scale (log)", color='#eab308')
    ax.tick_params(axis='y', labelcolor='#eab308')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    ax3 = ax.twinx()
    ax3.plot(frames, h['innovation'], color='#38bdf8', linewidth=1.5, alpha=0.7,
             label='Innovation residual')
    ax3.set_ylabel("Innovation", color='#38bdf8')
    ax3.tick_params(axis='y', labelcolor='#38bdf8')
    ax3.legend(loc='upper right', fontsize=9)
    shade_blockages(ax)

    axes[-1].set_xlabel("Operational Processing Frame Window (1 Frame ≈ 60 ms)")

    fig.suptitle("Kalman Dashboard — Live Snapshot",
                 fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')
