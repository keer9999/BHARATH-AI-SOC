/* ═══════════════════════════════════════════════════════════════════════════════
   EcoSort — SAHI (Slicing Aided Hyper Inference) Frontend Controller
   Bharat AI-SoC 2026–27 · PS2
   ═══════════════════════════════════════════════════════════════════════════════ */

// ─── SAHI State ───────────────────────────────────────────────────────────────
let _sahiEnabled   = false;
let _sahiAvailable = false;
let _sahiProfile   = 'default';

/** Fetch SAHI status from backend and sync UI pill + toggle. */
async function fetchSahiStatus() {
    try {
        const res  = await fetch('/api/sahi/status');
        const data = await res.json();
        if (!data.success) return;
        _sahiEnabled   = !!data.enabled;
        _sahiAvailable = !!data.available;
        _sahiProfile   = data.config?.profile || 'default';
        _renderSahiBadge();
        _syncSahiToggle();
    } catch (_) { /* backend not running yet */ }
}

/** Toggle SAHI on / off, optionally switching profile. */
async function toggleSahi(profile) {
    if (!_sahiAvailable) {
        showToast('SAHI not available on this install', 'warn');
        return;
    }
    const selProfile = profile || document.getElementById('sahiProfileSelect')?.value || 'default';
    const body = { enabled: !_sahiEnabled, profile: selProfile };
    try {
        const res  = await fetch('/api/sahi/toggle', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        const data = await res.json();
        if (data.success) {
            _sahiEnabled = data.enabled;
            if (data.profile) {
                _sahiProfile = data.profile;
                const profileSel = document.getElementById('sahiProfileSelect');
                if (profileSel) profileSel.value = data.profile;
            }
            _renderSahiBadge();
            _syncSahiToggle();
            showToast(
                _sahiEnabled
                    ? `🔬 SAHI ON — tiled inference active (${data.profile || 'default'})`
                    : '⚡ SAHI OFF — standard single-pass inference',
                _sahiEnabled ? 'success' : 'info'
            );
        } else {
            showToast('SAHI toggle failed: ' + (data.error || '?'), 'error');
        }
    } catch (err) {
        showToast('Could not reach backend: ' + err.message, 'error');
    }
}

/** Render the SAHI status badge in the scanner toolbar. */
function _renderSahiBadge() {
    const badge = document.getElementById('sahiBadge');
    if (!badge) return;
    if (!_sahiAvailable) {
        badge.textContent  = 'SAHI N/A';
        badge.className    = 'sahi-badge sahi-na';
        badge.title        = 'sahi_detector.py not importable on this install';
        return;
    }
    if (_sahiEnabled) {
        badge.textContent = '🔬 SAHI ON';
        badge.className   = 'sahi-badge sahi-on';
        badge.title       = 'Slicing Aided Hyper Inference active — better small-object recall';
    } else {
        badge.textContent = '⚡ SAHI OFF';
        badge.className   = 'sahi-badge sahi-off';
        badge.title       = 'Standard single-pass YOLO inference';
    }
}

/** Keep the toggle checkbox in sync with _sahiEnabled. */
function _syncSahiToggle() {
    const cb = document.getElementById('sahiToggle');
    if (cb) cb.checked = _sahiEnabled;
    const profileSel = document.getElementById('sahiProfileSelect');
    if (profileSel) profileSel.disabled = !_sahiAvailable;
}

/** Minimal toast helper (falls back gracefully if no toast container). */
function showToast(msg, type = 'info') {
    // Try existing toast / snackbar system if present
    if (typeof window.showNotification === 'function') {
        window.showNotification(msg, type);
        return;
    }
    let container = document.getElementById('toastContainer');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toastContainer';
        container.style.cssText = [
            'position:fixed', 'bottom:1.5rem', 'right:1.5rem', 'z-index:9999',
            'display:flex', 'flex-direction:column', 'gap:0.5rem',
            'pointer-events:none',
        ].join(';');
        document.body.appendChild(container);
    }
    const colors = { success:'#10b981', warn:'#f59e0b', error:'#ef4444', info:'#6366f1' };
    const t = document.createElement('div');
    t.style.cssText = [
        `background:${colors[type]||colors.info}`, 'color:#fff',
        'padding:0.6rem 1.1rem', 'border-radius:8px', 'font-size:0.85rem',
        'box-shadow:0 4px 16px rgba(0,0,0,.35)', 'transition:opacity .4s',
        'pointer-events:all',
    ].join(';');
    t.textContent = msg;
    container.appendChild(t);
    setTimeout(() => { t.style.opacity = '0'; setTimeout(() => t.remove(), 450); }, 3200);
}

/** Update the inference badge text to include SAHI label when active. */
function sahiUpdateBadge(inferenceMs) {
    const badge = document.getElementById('inferenceBadge');
    if (!badge) return;
    badge.textContent = _sahiEnabled
        ? `${inferenceMs} ms (SAHI+YOLO)`
        : `${inferenceMs} ms (YOLO)`;
}

// ─── Initialisation ───────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    fetchSahiStatus();

    // Wire up the toggle checkbox if present
    const cb = document.getElementById('sahiToggle');
    if (cb) {
        cb.addEventListener('change', () => {
            const profileSel = document.getElementById('sahiProfileSelect');
            const profile = profileSel ? profileSel.value : 'default';
            toggleSahi(profile);
        });
    }

    // Wire up profile selection change
    const profileSel = document.getElementById('sahiProfileSelect');
    if (profileSel) {
        profileSel.addEventListener('change', async () => {
            if (_sahiEnabled) {
                // If already on, re-apply with new profile
                try {
                    const res = await fetch('/api/sahi/toggle', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ enabled: true, profile: profileSel.value }),
                    });
                    const data = await res.json();
                    if (data.success) {
                        _sahiProfile = data.profile;
                        showToast(`🔬 SAHI profile switched to ${data.profile}`, 'info');
                    }
                } catch (e) {
                    showToast('Failed to switch SAHI profile: ' + e.message, 'error');
                }
            }
        });
    }
});

// Expose globally
window.toggleSahi     = toggleSahi;
window.fetchSahiStatus = fetchSahiStatus;
window.sahiUpdateBadge = sahiUpdateBadge;
