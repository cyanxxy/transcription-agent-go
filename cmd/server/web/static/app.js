(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);

  // ---- Static data -----------------------------------------------------------
  const MODELS = {
    'gemini-3.5-transcribe': { label: 'Gemini 3.5 Transcribe', provider: 'gemini', direct: true, languages: true },
    'MAI-Transcribe-2': { label: 'Microsoft MAI-Transcribe-2', provider: 'microsoft', direct: true, languages: true },
    'muse-voice-transcribe-1.0': { label: 'Meta Muse Voice', provider: 'meta', direct: true, languages: false },
    'gemini-4-hydrogen': { label: 'Gemini 4 Hydrogen', provider: 'gemini', direct: false, languages: true },
    'gemini-3.5-flash-lite': { label: 'Gemini 3.5 Flash-Lite', provider: 'gemini', direct: false, languages: true },
    'gemini-4-argon': { label: 'Gemini 4 Argon', provider: 'gemini', direct: false, languages: true },
  };
  // Saved settings can name a model that was replaced.
  const REPLACED_MODELS = { 'gemini-3.8-flash': 'gemini-4-hydrogen' };
  // Thinking levels beyond low, medium, and high.
  const EXTRA_THINKING = { 'gemini-3.5-flash-lite': ['minimal'], 'gemini-4-hydrogen': ['xhigh'] };
  const PROVIDERS = { gemini: 'Gemini', microsoft: 'Microsoft', meta: 'Meta' };

  const FORMATS = [
    { id: 'docx', label: 'Word', ext: '.docx', timestamps: true, binary: true },
    { id: 'txt', label: 'Text', ext: '.txt', timestamps: true },
    { id: 'md', label: 'Markdown', ext: '.md', timestamps: true },
    { id: 'srt', label: 'SubRip', ext: '.srt' },
    { id: 'vtt', label: 'WebVTT', ext: '.vtt' },
    { id: 'csv', label: 'CSV', ext: '.csv' },
    { id: 'json', label: 'JSON', ext: '.json' },
  ];

  const SPEAKER_COLORS = Array.from({ length: 8 }, (_, i) => `var(--sp-${i + 1})`);
  const GAP_SPEAKER = 'Untranscribed';
  const HISTORY_KEY = 'transcription-history';
  const ACTIVE_JOB_KEY = 'transcription-job-id';
  const TOKEN_KEY = 'transcription-access-token';

  // ---- Elements ----------------------------------------------------------------
  const form = $('upload-form');
  const speechReview = $('speech-review');
  const submitBtn = $('submit-btn');
  const spinner = submitBtn.querySelector('.spinner');
  const submitLabel = submitBtn.querySelector('.label');
  const formError = $('form-error');
  const dropzone = $('dropzone');
  const fileInput = $('audio-input');
  const fileTitle = $('composer-title');
  const fileHint = $('file-hint');
  const keyCallout = $('key-callout');
  const thinkingSelect = form.elements.namedItem('transcription_thinking_level');

  const compose = $('compose');
  const workspace = $('workspace');
  const navCompose = $('new-transcription');
  const sidebar = $('sidebar');
  const shell = $('shell');
  const statusCard = $('status-card');
  const resultCard = $('result-card');
  const progress = statusCard.querySelector('.progress');
  const progressBar = $('progress-bar');
  const runTitle = $('run-title');
  const runBack = $('run-back');
  const progressLabel = $('progress-label');
  const progressElapsed = $('progress-elapsed');
  const logEl = $('log');
  const connectionState = $('connection-state');
  const jobReference = $('job-reference');
  const cancelJob = $('cancel-job');
  const stageItems = [...$('stages').children];

  const reviewPanel = $('review-panel');
  const reviewActions = $('review-actions');
  const reviewReasons = $('review-reasons');
  const reviewDeadline = $('review-deadline');
  const reviewNote = $('review-note');
  const acceptReview = $('accept-review');
  const rejectReview = $('reject-review');

  const timelineLanes = $('timeline-lanes');
  const timelineHint = $('timeline-hint');
  const playhead = $('playhead');
  const player = $('player');
  const audioPlayer = $('audio-player');
  const followAudio = $('follow-audio');
  const transcriptView = $('transcript-view');
  const searchInput = $('transcript-search');
  const searchCount = $('search-count');
  const speakerList = $('speaker-list');
  const timeline = $('timeline');
  const seek = $('seek');
  const playToggle = $('play-toggle');
  const playbackRate = $('playback-rate');
  const playerCurrent = $('player-current');
  const resultTitle = $('result-title');
  const wave = $('wave');
  const waveTip = $('wave-tip');
  const resultScroll = $('result-scroll');
  const palette = $('palette');
  const paletteInput = $('palette-input');
  const paletteList = $('palette-list');
  const exportTimestamps = $('export-timestamps');

  const keysDialog = $('keys-dialog');
  const keysStatus = $('keys-status');
  const authToken = $('auth-token');
  const toastEl = $('toast');

  const tabs = [...document.querySelectorAll('.tab')];
  const panes = {
    transcript: $('tab-transcript'),
    subtitles: $('tab-subtitles'),
    judge: $('tab-judge'),
    quality: $('tab-quality'),
    evidence: $('tab-evidence'),
    json: $('tab-json'),
  };

  // ---- State -------------------------------------------------------------------
  let serverConfig = null;
  let currentJobId = null;
  let currentJobName = '';
  let currentResult = null;
  let currentStream = null;
  let awaitingReview = false;
  let reviewExpiresAt = null;
  let reviewTimer = null;
  const checkingStreams = new WeakSet();
  // Changes whenever another job takes over the screen, so work that was
  // awaiting a response can tell that it no longer owns the view.
  let runToken = 0;
  let uploading = '';
  // True while an upload or transcription is in progress. A transcript that
  // only awaits review is not busy, so a new job can start meanwhile.
  let busy = false;
  let elapsedTimer = null;
  let audioURL = null; // file selected for the next upload
  let jobAudioURL = null; // file behind the transcript on screen
  let segmentRows = [];
  let activeRow = null;
  let timelineDuration = 0;
  let filePeaks = null; // waveform of the file selected for the next upload
  let jobPeaks = null; // waveform behind the transcript on screen
  let focusedRow = -1;
  let liveFraction = 0;
  let stageStarted = 0;
  let stageIndex = -1;
  const speakerNames = new Map();
  const speakerColors = new Map();

  // ---- Helpers -----------------------------------------------------------------
  const storage = {
    get(store, key, fallback) {
      try { const value = store.getItem(key); return value === null ? fallback : value; } catch (_) { return fallback; }
    },
    set(store, key, value) {
      try { store.setItem(key, value); } catch (_) { /* storage unavailable */ }
    },
    remove(store, key) {
      try { store.removeItem(key); } catch (_) { /* storage unavailable */ }
    },
  };

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function authHeaders(extra = {}) {
    const token = authToken.value.trim();
    return token ? { ...extra, Authorization: `Bearer ${token}` } : extra;
  }

  function formatBytes(bytes) {
    if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1).replace(/\.0$/, '')} GB`;
    if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1).replace(/\.0$/, '')} MB`;
    return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  }

  function formatClock(seconds) {
    const total = Math.max(0, Math.floor(seconds || 0));
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
    const pad = (n) => String(n).padStart(2, '0');
    return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
  }

  function formatDuration(seconds) {
    const total = Math.round(seconds || 0);
    if (total < 60) return `${total}s`;
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    if (h) return m ? `${h}h ${m}m` : `${h}h`;
    return total % 60 ? `${m}m ${total % 60}s` : `${m}m`;
  }

  function relativeTime(date) {
    const minutes = Math.round((Date.now() - date) / 60000);
    if (minutes < 1) return 'now';
    if (minutes < 60) return `${minutes}m`;
    const hours = Math.round(minutes / 60);
    return hours < 24 ? `${hours}h` : `${Math.round(hours / 24)}d`;
  }

  const SVG = 'http://www.w3.org/2000/svg';
  function svgEl(tag, attributes) {
    const node = document.createElementNS(SVG, tag);
    Object.entries(attributes).forEach(([name, value]) => node.setAttribute(name, String(value)));
    return node;
  }

  function icon(name) {
    const svg = svgEl('svg', { class: 'i', 'aria-hidden': 'true' });
    svg.append(svgEl('use', { href: `#i-${name}` }));
    return svg;
  }

  // One small mark per job state; a running job shows its real progress.
  const STATUS_LABELS = { running: 'Running', review: 'Needs review', done: 'Done', failed: 'Failed', canceled: 'Canceled', rejected: 'Rejected', expired: 'Expired' };
  function statusIcon(status, fraction = 0) {
    const svg = svgEl('svg', { class: 'status-icon', viewBox: '0 0 14 14', 'data-status': status, role: 'img', 'aria-label': STATUS_LABELS[status] || 'Unknown' });
    if (status === 'done' || status === 'failed') {
      svg.append(svgEl('circle', { cx: 7, cy: 7, r: 6, class: 'fill' }));
      svg.append(svgEl('path', { d: status === 'done' ? 'M4.300 7.200 6.300 9.200 9.800 5.100' : 'M5 5l4 4M9 5l-4 4', class: 'mark' }));
      return svg;
    }
    svg.append(svgEl('circle', { cx: 7, cy: 7, r: 5.25, class: 'ring' }));
    if (status === 'running' || status === 'review') {
      const part = status === 'review' ? 0.5 : Math.max(0.04, fraction);
      svg.append(svgEl('circle', { cx: 7, cy: 7, r: 1.75, class: 'pie', pathLength: 1, 'stroke-dasharray': `${part} 1`, transform: 'rotate(-90 7 7)' }));
    }
    return svg;
  }

  function segmentStart(segment) {
    if (typeof segment.start === 'number') return segment.start;
    const match = /^\[(\d+):(\d{2}):(\d{2})\]$/.exec(segment.timestamp || '');
    return match ? Number(match[1]) * 3600 + Number(match[2]) * 60 + Number(match[3]) : 0;
  }

  function messageFrom(text, status) {
    try {
      const body = JSON.parse(text);
      if (body && body.error) return body.error;
    } catch (_) { /* not JSON */ }
    return String(text || '').trim() || `The server returned ${status}.`;
  }

  async function errorMessage(response) {
    return messageFrom(await response.text(), response.status);
  }

  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const scrollBehavior = () => (reducedMotion.matches ? 'auto' : 'smooth');

  // Show and hide with the CSS enter (@starting-style) and exit (.leaving)
  // transitions. Showing mid-exit reverses the transition from where it is.
  const leaveTimers = new WeakMap();
  function showAnimated(node) {
    window.clearTimeout(leaveTimers.get(node));
    node.classList.remove('leaving');
    node.hidden = false;
  }
  function hideAnimated(node, ms) {
    if (node.hidden) return;
    node.classList.add('leaving');
    window.clearTimeout(leaveTimers.get(node));
    leaveTimers.set(node, window.setTimeout(() => {
      node.hidden = true;
      node.classList.remove('leaving');
    }, ms));
  }

  // ---- Views ---------------------------------------------------------------------
  function showView(view, focusTarget) {
    const job = view === 'job';
    compose.hidden = job;
    workspace.hidden = !job;
    // Only one player is on screen at a time.
    (job ? previewAudio : audioPlayer).pause();
    navCompose.toggleAttribute('aria-current', false);
    if (!job) navCompose.setAttribute('aria-current', 'page');
    closeSidebar();
    renderHistory();
    syncDocumentTitle();
    if (focusTarget) focusTarget.focus({ preventScroll: true });
  }

  function syncDocumentTitle() {
    const name = !workspace.hidden && currentJobName ? currentJobName : '';
    document.title = busy && currentJobId
      ? `${Math.round(liveFraction * 100)}% ${currentJobName || 'Transcribing'}`
      : name || 'Transcription Studio';
  }

  function closeSidebar() {
    sidebar.classList.remove('open');
    $('scrim').hidden = true;
  }

  document.querySelectorAll('[data-sidebar-toggle]').forEach((button) => button.addEventListener('click', () => {
    sidebar.classList.add('open');
    $('scrim').hidden = false;
    navCompose.focus();
  }));
  $('scrim').addEventListener('click', closeSidebar);
  navCompose.addEventListener('click', () => showView('compose'));
  runBack.addEventListener('click', () => showView('compose', fileTitle));

  const DETAILS_KEY = 'transcription-details';
  function setDetails(open) {
    shell.classList.toggle('no-details', !open);
    storage.set(localStorage, DETAILS_KEY, open ? 'open' : 'closed');
    document.querySelectorAll('[data-details-toggle]').forEach((button) => button.setAttribute('aria-pressed', String(open)));
  }
  const toggleDetails = () => setDetails(shell.classList.contains('no-details'));
  document.querySelectorAll('[data-details-toggle]').forEach((button) => button.addEventListener('click', toggleDetails));
  setDetails(storage.get(localStorage, DETAILS_KEY, 'open') !== 'closed');

  // ---- Theme -----------------------------------------------------------------------
  const THEME_KEY = 'transcription-theme';
  const THEMES = { system: 'System theme', light: 'Light theme', dark: 'Dark theme' };
  function setTheme(theme) {
    if (theme === 'system') delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = theme;
    if (theme === 'system') storage.remove(localStorage, THEME_KEY);
    else storage.set(localStorage, THEME_KEY, theme);
    $('theme-label').textContent = THEMES[theme];
    themeChanged();
  }
  const currentTheme = () => document.documentElement.dataset.theme || 'system';
  function nextTheme() {
    const order = Object.keys(THEMES);
    setTheme(order[(order.indexOf(currentTheme()) + 1) % order.length]);
  }
  $('theme-button').addEventListener('click', nextTheme);
  $('theme-label').textContent = THEMES[currentTheme()];

  // ---- Menus -----------------------------------------------------------------------
  // Native popovers: the browser handles light dismiss, Escape and focus return.
  function placeMenu(trigger, menu) {
    const box = trigger.getBoundingClientRect();
    const gap = 6;
    const left = Math.min(Math.max(8, box.right - menu.offsetWidth), window.innerWidth - menu.offsetWidth - 8);
    const below = box.bottom + gap + menu.offsetHeight <= window.innerHeight - 8;
    const top = below ? box.bottom + gap : Math.max(8, box.top - gap - menu.offsetHeight);
    menu.style.left = `${left}px`;
    menu.style.top = `${top}px`;
    menu.style.transformOrigin = below ? 'top right' : 'bottom right';
  }

  document.querySelectorAll('[data-menu]').forEach((trigger) => {
    const menu = $(trigger.dataset.menu);
    trigger.setAttribute('aria-haspopup', 'true');
    trigger.setAttribute('aria-expanded', 'false');
    trigger.setAttribute('aria-controls', menu.id);
    trigger.popoverTargetElement = menu;
    menu.addEventListener('toggle', (event) => {
      const open = event.newState === 'open';
      trigger.setAttribute('aria-expanded', String(open));
      if (!open) return;
      placeMenu(trigger, menu);
    });
    // Opening focuses the autofocus element at once, so fast typing lands.
    menu.addEventListener('beforetoggle', (event) => {
      if (event.newState !== 'open') return;
      menu.querySelectorAll('[autofocus]').forEach((node) => node.removeAttribute('autofocus'));
      const target = menu.querySelector('input[type="radio"]:checked, .menu-input, button.menu-item');
      if (target) target.setAttribute('autofocus', '');
    });
    menu.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && event.target.matches('input[type="radio"], input.menu-input')) {
        event.preventDefault();
        menu.hidePopover();
      }
      if (!['ArrowDown', 'ArrowUp'].includes(event.key) || event.target.matches('textarea, input[type="radio"]')) return;
      const items = [...menu.querySelectorAll('.menu-input, button.menu-item')].filter((item) => item.offsetParent !== null);
      const index = items.indexOf(event.target);
      if (index < 0) return;
      event.preventDefault();
      items[(index + (event.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length].focus();
    });
  });
  window.addEventListener('resize', () => document.querySelectorAll('.menu:popover-open').forEach((menu) => menu.hidePopover()));

  // ---- Waveforms -------------------------------------------------------------------
  // Decoding holds the whole file in memory, so long recordings use the plain
  // speaker timeline instead.
  const WAVE_MAX_SECONDS = 15 * 60;
  const WAVE_MAX_BYTES = 80 * 1024 * 1024;

  async function readPeaks(file) {
    const context = new OfflineAudioContext(1, 1, 8000);
    const audio = await context.decodeAudioData(await file.arrayBuffer());
    const samples = audio.getChannelData(0);
    const peaks = new Float32Array(1600);
    const size = Math.max(1, samples.length / peaks.length);
    let top = 0;
    for (let i = 0; i < peaks.length; i += 1) {
      let sum = 0;
      let count = 0;
      for (let j = Math.floor(i * size); j < Math.min(samples.length, (i + 1) * size); j += 1) {
        sum += samples[j] * samples[j];
        count += 1;
      }
      peaks[i] = count ? Math.sqrt(sum / count) : 0;
      top = Math.max(top, peaks[i]);
    }
    return top > 0 ? peaks.map((peak) => peak / top) : null;
  }

  // Canvas needs resolved colors, not light-dark() or var() text.
  const colorProbe = el('span', 'visually-hidden');
  document.body.append(colorProbe);
  function resolveColor(value) {
    colorProbe.style.color = value;
    return getComputedStyle(colorProbe).color;
  }

  const BAR = 2;
  const STEP = 4;
  function paintWave(canvas, peaks, colorAt, progress) {
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (!peaks || !width || !height) return;
    const ratio = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
    }
    const context = canvas.getContext('2d');
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);
    const count = Math.floor(width / STEP);
    const bucket = peaks.length / count;
    for (let i = 0; i < count; i += 1) {
      let peak = 0;
      for (let j = Math.floor(i * bucket); j < Math.max((i + 1) * bucket, Math.floor(i * bucket) + 1); j += 1) peak = Math.max(peak, peaks[j] || 0);
      const size = Math.max(2, peak ** 0.8 * height);
      context.globalAlpha = (i + 0.5) / count <= progress ? 1 : 0.38;
      context.fillStyle = colorAt(i, count);
      context.beginPath();
      context.roundRect(i * STEP, (height - size) / 2, BAR, size, 1);
      context.fill();
    }
  }

  const previewAudio = new Audio();
  const previewWave = $('preview-wave');
  const previewPlay = $('preview-play');
  let previewInk = '';
  function paintPreview() {
    if (!filePeaks || compose.hidden) return;
    previewInk = previewInk || resolveColor('var(--accent)');
    paintWave(previewWave, filePeaks, () => previewInk, previewAudio.duration ? previewAudio.currentTime / previewAudio.duration : 0);
  }
  let previewFrame = 0;
  function followPreview() {
    paintPreview();
    $('preview-time').textContent = formatClock(previewAudio.currentTime);
    previewFrame = previewAudio.paused ? 0 : window.requestAnimationFrame(followPreview);
  }
  function syncPreview() {
    previewPlay.classList.toggle('playing', !previewAudio.paused);
    previewPlay.setAttribute('aria-label', previewAudio.paused ? 'Play recording' : 'Pause recording');
    window.cancelAnimationFrame(previewFrame);
    followPreview();
  }
  ['play', 'pause', 'ended', 'seeked'].forEach((name) => previewAudio.addEventListener(name, syncPreview));
  function togglePreview() {
    if (previewAudio.paused) void previewAudio.play().catch(() => toast('This browser cannot play the recording.'));
    else previewAudio.pause();
  }
  previewPlay.addEventListener('click', togglePreview);
  previewWave.addEventListener('click', (event) => {
    if (!previewAudio.duration) return;
    const box = previewWave.getBoundingClientRect();
    previewAudio.currentTime = ((event.clientX - box.left) / box.width) * previewAudio.duration;
  });
  new ResizeObserver(paintPreview).observe(previewWave);

  // The empty state's signal: a fixed, speech-like shape.
  const idleWave = $('idle-wave');
  const idlePeaks = Float32Array.from({ length: 105 }, (_, i) => {
    const phrase = Math.max(0, Math.sin(i * 0.19) * Math.sin(i * 0.067 + 0.4));
    return (0.14 + phrase * 0.86) * (0.55 + 0.45 * Math.abs(Math.sin(i * 1.7)));
  });
  function paintIdle() {
    const ink = resolveColor('var(--accent)');
    paintWave(idleWave, idlePeaks, () => ink, 1);
  }
  new ResizeObserver(paintIdle).observe(idleWave);

  // Each bar takes the color of whoever is speaking at that moment.
  let waveColors = [];
  function colorWave(count) {
    const segments = currentResult ? currentResult.result.segments || [] : [];
    const quiet = resolveColor('var(--muted)');
    const inks = new Map();
    waveColors = new Array(count).fill(quiet);
    segments.forEach((segment, index) => {
      if (segment.speaker === GAP_SPEAKER) return;
      if (!inks.has(segment.speaker)) inks.set(segment.speaker, resolveColor(colorFor(segment.speaker)));
      const from = Math.max(0, Math.floor((segmentStart(segment) / timelineDuration) * count));
      const to = Math.min(count - 1, Math.floor((segmentEnd(segments, index) / timelineDuration) * count));
      for (let i = from; i <= to; i += 1) waveColors[i] = inks.get(segment.speaker);
    });
  }
  function paintPlayer() {
    if (!jobPeaks || wave.hidden) return;
    const count = Math.floor(wave.clientWidth / STEP);
    if (waveColors.length !== count) colorWave(count);
    paintWave(wave, jobPeaks, (i) => waveColors[i], timelineDuration ? audioPlayer.currentTime / timelineDuration : 0);
  }
  new ResizeObserver(paintPlayer).observe(wave);

  function themeChanged() {
    previewInk = '';
    waveColors = [];
    paintIdle();
    paintPreview();
    paintPlayer();
  }
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', themeChanged);

  let toastTimer = null;
  function toast(message) {
    toastEl.textContent = message;
    showAnimated(toastEl);
    window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => hideAnimated(toastEl, 150), 2600);
  }

  function selectedModel() {
    return form.elements.namedItem('model_name').value;
  }

  function isTyping(target) {
    if (!(target instanceof HTMLElement)) return false;
    if (target.tagName === 'INPUT') return !['range', 'checkbox', 'radio', 'file'].includes(target.type);
    return target.isContentEditable || ['TEXTAREA', 'SELECT'].includes(target.tagName);
  }

  // ---- Server configuration and keys -----------------------------------------
  function providerConfigured(provider) {
    if (!serverConfig || !serverConfig.providers) return true; // unknown: let the server decide
    const status = serverConfig.providers[provider];
    return Boolean(status && status.configured);
  }

  function requiredProviders(model, review) {
    const info = MODELS[model];
    if (!info) return [];
    const needed = [info.provider];
    if (info.direct && review && info.provider !== 'gemini') needed.push('gemini');
    return needed;
  }

  async function loadConfig() {
    try {
      const response = await fetch('/api/config', { cache: 'no-store' });
      if (!response.ok) throw new Error(await errorMessage(response));
      serverConfig = await response.json();
    } catch (error) {
      serverConfig = null;
      keysStatus.textContent = `Could not load key status: ${error.message}`;
      keysStatus.dataset.state = 'error';
    }
    syncForm();
    if (!fileInput.files || !fileInput.files.length) renderFileHint();
    renderHistory();
  }

  // Per-provider dialog state. Key status itself comes from serverConfig.
  const keyRows = new Map();
  document.querySelectorAll('.key-row[data-provider]').forEach((row) => {
    keyRows.set(row.dataset.provider, { row, editing: false, confirming: false, busy: false });
  });
  const tokenFeedback = $('token-feedback');

  function keyStatus(provider) {
    return ((serverConfig && serverConfig.providers) || {})[provider] || {};
  }

  function keyInputs(state) {
    return [...state.row.querySelectorAll('input[data-field]')];
  }

  function keyInputsReady(state) {
    return keyInputs(state).every((input) => input.value.trim() !== '');
  }

  function setRowFeedback(state, text, kind = '') {
    const node = state.row.querySelector('[data-feedback]');
    node.textContent = text;
    node.dataset.state = kind;
  }

  function setPending(button, label) {
    button.querySelector('.spinner').hidden = !label;
    if (label) button.querySelector('[data-label]').textContent = label;
  }

  function concealKey(state) {
    state.row.querySelector('input[data-field="api_key"]').type = 'password';
    const toggle = state.row.querySelector('[data-action="reveal"]');
    toggle.setAttribute('aria-pressed', 'false');
    toggle.setAttribute('aria-label', 'Show key');
  }

  function resetKeyInputs(state) {
    keyInputs(state).forEach((input) => { input.value = ''; });
    concealKey(state);
  }

  function renderKeyRow(provider, state) {
    const { row } = state;
    const status = keyStatus(provider);
    const fromEnv = status.source === 'environment';
    const manageable = Boolean(serverConfig && serverConfig.can_manage_keys) && !fromEnv;
    if (!manageable || !status.configured) {
      state.editing = false;
      state.confirming = false;
    }
    const badge = row.querySelector('[data-status]');
    badge.dataset.state = status.source || 'missing';
    badge.textContent = fromEnv ? 'Set by server' : status.configured ? 'Saved' : 'Not set';

    row.querySelector('[data-summary]').hidden = !status.configured || state.editing || state.confirming || (!fromEnv && !manageable);
    row.querySelector('[data-note]').textContent = fromEnv ? 'Set by an environment variable on the server.' : '';
    row.querySelector('[data-action="edit"]').hidden = !manageable;
    row.querySelector('[data-action="remove"]').hidden = !manageable;
    row.querySelector('[data-confirm]').hidden = !state.confirming;
    row.querySelector('[data-editor]').hidden = !manageable || (status.configured && !state.editing);
    row.querySelector('[data-action="cancel-edit"]').hidden = !status.configured;

    const save = row.querySelector('[data-action="save"]');
    save.disabled = state.busy || !keyInputsReady(state);
    if (!state.busy) {
      setPending(save, null);
      save.querySelector('[data-label]').textContent = status.configured ? 'Replace' : 'Save';
      const confirm = row.querySelector('[data-action="confirm-remove"]');
      setPending(confirm, null);
      confirm.querySelector('[data-label]').textContent = 'Remove';
    }
    row.querySelector('[data-action="confirm-remove"]').disabled = state.busy;
    row.setAttribute('aria-busy', String(state.busy));
  }

  function renderKeys() {
    const canManage = Boolean(serverConfig && serverConfig.can_manage_keys);
    $('keys-intro').textContent = canManage
      ? 'Keys are stored on this server, never in your browser. Add a key for each provider you want to use.'
      : 'Keys are stored on the server. To add or change them, open this page on the server computer (localhost).';
    keyRows.forEach((state, provider) => renderKeyRow(provider, state));
    const anyMissing = requiredProviders(selectedModel(), speechReview.checked).some((p) => !providerConfigured(p));
    $('keys-dot').dataset.state = serverConfig ? (anyMissing ? 'missing' : 'ok') : 'unknown';
  }

  async function updateKey(provider, remove) {
    const state = keyRows.get(provider);
    if (state.busy) return;
    if (!remove && !keyInputsReady(state)) {
      setRowFeedback(state, provider === 'microsoft' ? 'Enter both the key and the endpoint.' : `Paste a ${PROVIDERS[provider]} key first.`, 'error');
      const empty = keyInputs(state).find((input) => !input.value.trim());
      if (empty) empty.focus();
      return;
    }
    const body = { provider };
    keyInputs(state).forEach((input) => { body[input.dataset.field] = input.value.trim(); });
    const button = state.row.querySelector(remove ? '[data-action="confirm-remove"]' : '[data-action="save"]');
    state.busy = true;
    setRowFeedback(state, '');
    renderKeyRow(provider, state);
    setPending(button, remove ? 'Removing…' : 'Saving…');
    let focusAfter = null;
    try {
      const response = await fetch(remove ? `/api/config/keys?provider=${provider}` : '/api/config/keys', {
        method: remove ? 'DELETE' : 'POST',
        headers: authHeaders({ 'Content-Type': 'application/json' }),
        body: remove ? undefined : JSON.stringify(body),
      });
      if (!response.ok) throw new Error(await errorMessage(response));
      state.editing = false;
      state.confirming = false;
      resetKeyInputs(state);
      await loadConfig();
      setRowFeedback(state, remove ? 'Key removed.' : 'Key saved.', 'ok');
      focusAfter = remove ? '[data-editor] input[data-field="api_key"]' : '[data-action="edit"]';
    } catch (error) {
      setRowFeedback(state, error.message, 'error');
    } finally {
      state.busy = false;
      renderKeys();
    }
    const target = focusAfter && state.row.querySelector(focusAfter);
    if (target && target.offsetParent !== null) target.focus();
  }

  function cancelKeyEdit(provider, state) {
    state.editing = false;
    resetKeyInputs(state);
    renderKeyRow(provider, state);
    state.row.querySelector('[data-action="edit"]').focus();
  }

  keyRows.forEach((state, provider) => {
    const { row } = state;
    const on = (action, handler) => row.querySelector(`[data-action="${action}"]`).addEventListener('click', handler);
    on('save', () => updateKey(provider, false));
    on('confirm-remove', () => updateKey(provider, true));
    on('edit', () => {
      state.editing = true;
      setRowFeedback(state, '');
      const endpoint = row.querySelector('input[data-field="endpoint"]');
      if (endpoint) endpoint.value = keyStatus(provider).endpoint || '';
      renderKeyRow(provider, state);
      row.querySelector('input[data-field="api_key"]').focus();
    });
    on('cancel-edit', () => cancelKeyEdit(provider, state));
    on('remove', () => {
      state.confirming = true;
      setRowFeedback(state, '');
      renderKeyRow(provider, state);
      row.querySelector('[data-action="cancel-remove"]').focus();
    });
    on('cancel-remove', () => {
      state.confirming = false;
      renderKeyRow(provider, state);
      row.querySelector('[data-action="remove"]').focus();
    });
    on('reveal', () => {
      const input = row.querySelector('input[data-field="api_key"]');
      const show = input.type === 'password';
      input.type = show ? 'text' : 'password';
      const toggle = row.querySelector('[data-action="reveal"]');
      toggle.setAttribute('aria-pressed', String(show));
      toggle.setAttribute('aria-label', show ? 'Hide key' : 'Show key');
    });
    keyInputs(state).forEach((input) => {
      input.addEventListener('input', () => {
        if (row.querySelector('[data-feedback]').dataset.state === 'error') setRowFeedback(state, '');
        renderKeyRow(provider, state);
      });
      input.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
          event.preventDefault();
          updateKey(provider, false);
        } else if (event.key === 'Escape' && state.editing) {
          // Back out of replacing a key instead of closing the whole dialog.
          event.preventDefault();
          cancelKeyEdit(provider, state);
        }
      });
    });
  });

  function openKeys(provider) {
    keysStatus.textContent = '';
    keysStatus.dataset.state = '';
    tokenFeedback.textContent = '';
    keyRows.forEach((state) => {
      if (state.busy) return;
      state.editing = false;
      state.confirming = false;
      setRowFeedback(state, '');
      concealKey(state);
    });
    renderKeys();
    if (typeof keysDialog.showModal === 'function') keysDialog.showModal();
    else keysDialog.setAttribute('open', '');
    // Land on the input that still needs a key rather than the close button.
    const editable = '[data-editor]:not([hidden]) input[data-field="api_key"]:not(:disabled)';
    const target = (provider && keysDialog.querySelector(`.key-row[data-provider="${provider}"] ${editable}`))
      || keysDialog.querySelector(editable);
    if (target) target.focus();
    else keysDialog.focus();
  }

  $('keys-button').addEventListener('click', () => openKeys());
  $('key-callout-action').addEventListener('click', () => {
    const missing = requiredProviders(selectedModel(), speechReview.checked).filter((p) => !providerConfigured(p));
    openKeys(missing[0]);
  });
  keysDialog.addEventListener('click', (event) => {
    if (event.target === keysDialog) keysDialog.close();
  });
  authToken.value = storage.get(sessionStorage, TOKEN_KEY, '');
  authToken.addEventListener('input', () => storage.set(sessionStorage, TOKEN_KEY, authToken.value.trim()));
  authToken.addEventListener('keydown', (event) => {
    // Enter would otherwise submit the method="dialog" form and close the dialog.
    if (event.key !== 'Enter') return;
    event.preventDefault();
    storage.set(sessionStorage, TOKEN_KEY, authToken.value.trim());
    tokenFeedback.textContent = authToken.value.trim() ? 'Token saved for this browser tab.' : 'Token cleared.';
    tokenFeedback.dataset.state = 'ok';
  });

  // ---- Form ---------------------------------------------------------------------
  function reviewEnabled() {
    const info = MODELS[selectedModel()];
    return !info || !info.direct || speechReview.checked;
  }

  function syncForm() {
    const model = selectedModel();
    const info = MODELS[model] || MODELS['gemini-3.5-transcribe'];
    const pipeline = reviewEnabled();

    document.querySelectorAll('[data-review-only]').forEach((group) => {
      group.hidden = !pipeline;
      group.querySelectorAll('input, select, textarea').forEach((input) => { input.disabled = !pipeline; });
    });
    $('speech-review-field').hidden = !info.direct;
    speechReview.disabled = !info.direct;
    $('judge-pipeline-field').hidden = info.direct;
    form.elements.namedItem('use_judge_pipeline').disabled = !pipeline || info.direct;
    $('language-codes-field').hidden = !info.languages;
    form.elements.namedItem('language_codes').disabled = !info.languages;

    syncThinking(thinkingSelect, model);
    syncThinking(form.elements.namedItem('judge_thinking_level'), form.elements.namedItem('judge_model_name').value);

    const noKeys = Boolean(serverConfig) && !Object.keys(PROVIDERS).some(providerConfigured);
    const missing = requiredProviders(model, speechReview.checked).filter((p) => !providerConfigured(p));
    keyCallout.hidden = missing.length === 0;
    if (missing.length) {
      $('key-callout-text').textContent = noKeys
        ? 'Add an API key for Gemini, Microsoft, or Meta to start transcribing.'
        : missing.includes(info.provider)
          ? `Add a ${PROVIDERS[info.provider]} API key to use ${info.label}.`
          : 'Double-checking with a judge needs a Gemini API key.';
    }
    document.querySelectorAll('[data-key-badge]').forEach((badge) => {
      badge.hidden = !serverConfig || noKeys || providerConfigured(badge.dataset.keyBadge);
    });
    renderSettingLabels();
    updateSubmitState();
    renderKeys();
  }

  // The settings rows show their current value, like the rest of the panel.
  const languageInput = form.elements.namedItem('language_codes');
  const speakerInput = form.elements.namedItem('speakers');
  const languageButtons = [...document.querySelectorAll('#language-options [data-code]')];
  const languageCodes = () => languageInput.value.split(/[\s,]+/).filter(Boolean);

  function renderSettingLabels() {
    const info = MODELS[selectedModel()] || MODELS['gemini-3.5-transcribe'];
    $('model-label').textContent = info.label;
    const codes = languageCodes();
    $('language-label').textContent = codes.length ? codes.join(', ') : 'Detect';
    languageButtons.forEach((button) => {
      const on = codes.some((code) => code.toLowerCase() === button.dataset.code.toLowerCase());
      button.setAttribute('aria-pressed', String(on));
    });
    const names = speakerInput.value.split('\n').map((name) => name.trim()).filter(Boolean);
    $('speakers-label').textContent = names.length ? names.join(', ') : 'Detect';
  }

  languageButtons.forEach((button) => button.addEventListener('click', () => {
    const { code } = button.dataset;
    const codes = languageCodes();
    const rest = codes.filter((item) => item.toLowerCase() !== code.toLowerCase());
    languageInput.value = (rest.length === codes.length ? [...codes, code] : rest).join(', ');
    languageInput.dispatchEvent(new Event('change', { bubbles: true }));
    renderSettingLabels();
  }));
  [languageInput, speakerInput].forEach((input) => input.addEventListener('input', renderSettingLabels));
  document.querySelectorAll('input[name="model_name"]').forEach((radio) => {
    // A pointer choice closes the menu; arrow keys keep it open to compare.
    radio.closest('.option').addEventListener('click', (event) => {
      if (event.detail > 0 && event.target !== radio) window.setTimeout(() => $('model-menu').hidePopover(), 0);
    });
  });

  // Offers only the levels a model accepts; a level it rejects falls back to
  // the closest one.
  function syncThinking(select, model) {
    const extra = EXTRA_THINKING[model] || [];
    select.querySelectorAll('[data-extra-thinking]').forEach((option) => {
      option.hidden = !extra.includes(option.value);
      option.disabled = option.hidden;
      if (option.hidden && select.value === option.value) select.value = option.value === 'minimal' ? 'low' : 'high';
    });
  }

  function updateSubmitState() {
    const running = busy;
    const missing = requiredProviders(selectedModel(), speechReview.checked).some((p) => !providerConfigured(p));
    submitBtn.disabled = running || missing;
    submitBtn.title = missing ? 'Add the missing API key first' : '';
  }

  form.addEventListener('change', (event) => {
    if (['model_name', 'judge_model_name'].includes(event.target.name) || event.target === speechReview) syncForm();
    if (!fileInput.files || !fileInput.files.length) showFormError('');
    saveSettings();
  });

  // Settings worth keeping between recordings. Per-recording context (speaker
  // names, topic, instructions, keywords) is deliberately not remembered.
  const SETTINGS_KEY = 'transcription-settings';
  const REMEMBERED_FIELDS = [
    'model_name', 'language_codes', 'remove_fillers', 'speech_review', 'judge_model_name',
    'candidate_strategy', 'expected_format', 'transcription_thinking_level', 'judge_thinking_level',
    'service_tier', 'use_judge_pipeline', 'agentic_mode', 'auto_format', 'chunk_strategy', 'chunk_concurrency',
  ];

  function saveSettings() {
    const values = {};
    REMEMBERED_FIELDS.forEach((name) => {
      const field = form.elements.namedItem(name);
      if (!field) return;
      values[name] = field.type === 'checkbox' ? field.checked : field.value;
    });
    storage.set(localStorage, SETTINGS_KEY, JSON.stringify(values));
  }

  function restoreSettings() {
    let values;
    try {
      values = JSON.parse(storage.get(localStorage, SETTINGS_KEY, '{}')) || {};
    } catch (_) {
      return;
    }
    ['model_name', 'judge_model_name'].forEach((name) => {
      if (values[name] in REPLACED_MODELS) values[name] = REPLACED_MODELS[values[name]];
    });
    REMEMBERED_FIELDS.forEach((name) => {
      const field = form.elements.namedItem(name);
      if (!field || !(name in values)) return;
      if (field.type === 'checkbox') {
        field.checked = Boolean(values[name]);
      } else if (field instanceof RadioNodeList) {
        // Only restore a model that still exists.
        if ([...field].some((radio) => radio.value === values[name])) field.value = values[name];
      } else if (field.tagName === 'SELECT') {
        if ([...field.options].some((option) => option.value === values[name])) field.value = values[name];
      } else {
        field.value = String(values[name]);
      }
    });
  }

  // ---- File selection --------------------------------------------------------------
  function showFormError(message) {
    formError.textContent = message;
    formError.hidden = !message;
  }

  function renderFileHint() {
    const limit = serverConfig && serverConfig.max_upload_bytes;
    fileHint.textContent = `MP3, WAV, M4A, FLAC or OGG${limit ? `, up to ${formatBytes(limit)}` : ''}`;
  }

  function setFile(file) {
    if (audioURL && audioURL !== jobAudioURL) URL.revokeObjectURL(audioURL);
    audioURL = null;
    dropzone.classList.toggle('has-file', Boolean(file));
    $('file-clear').hidden = !file;
    $('file-choose').hidden = Boolean(file);
    $('idle-wave').hidden = Boolean(file);
    $('preview').hidden = !file;
    filePeaks = null;
    previewWave.hidden = true;
    previewAudio.pause();
    previewAudio.removeAttribute('src');
    if (!file) {
      fileInput.value = '';
      fileTitle.textContent = 'Drop a recording';
      renderFileHint();
      showFormError('');
      return;
    }
    const limit = serverConfig && serverConfig.max_upload_bytes;
    fileTitle.textContent = file.name;
    fileHint.textContent = formatBytes(file.size);
    showFormError(limit && file.size > limit ? `This file is ${formatBytes(file.size)}; the server accepts up to ${formatBytes(limit)}.` : '');
    audioURL = URL.createObjectURL(file);
    const url = audioURL;
    previewAudio.src = url;
    $('preview-time').textContent = '0:00';
    const probe = new Audio();
    probe.preload = 'metadata';
    probe.addEventListener('loadedmetadata', () => {
      if (url !== audioURL || !Number.isFinite(probe.duration)) return;
      fileHint.textContent = `${formatDuration(probe.duration)}, ${formatBytes(file.size)}`;
      if (probe.duration > WAVE_MAX_SECONDS || file.size > WAVE_MAX_BYTES) return;
      readPeaks(file).then((peaks) => {
        // The upload may already be running, or another file chosen meanwhile.
        if (url === jobAudioURL) {
          jobPeaks = peaks;
          if (currentResult) renderTimeline(currentResult.result);
        }
        if (url !== audioURL || !peaks) return;
        filePeaks = peaks;
        previewWave.hidden = false;
        paintPreview();
      }).catch(() => { /* not decodable in this browser: no waveform */ });
    }, { once: true });
    probe.src = audioURL;
  }

  fileInput.addEventListener('change', () => setFile(fileInput.files && fileInput.files[0]));
  $('file-clear').addEventListener('click', () => {
    setFile(null);
    fileInput.focus();
  });
  // Drop a file anywhere in the app.
  const dropOverlay = $('drop-overlay');
  const hasFiles = (event) => Boolean(event.dataTransfer && [...event.dataTransfer.types].includes('Files'));
  ['dragenter', 'dragover'].forEach((name) => document.addEventListener(name, (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    dropOverlay.hidden = false;
  }));
  document.addEventListener('dragleave', (event) => {
    if (!event.relatedTarget) dropOverlay.hidden = true;
  });
  document.addEventListener('drop', (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    dropOverlay.hidden = true;
    const { files } = event.dataTransfer;
    if (!files.length) return;
    fileInput.files = files;
    setFile(files[0]);
    showView('compose', fileTitle);
  });

  // ---- History --------------------------------------------------------------------
  function readHistory() {
    try {
      const items = JSON.parse(storage.get(localStorage, HISTORY_KEY, '[]'));
      return Array.isArray(items) ? items.filter((item) => item && typeof item.id === 'string') : [];
    } catch (_) {
      return [];
    }
  }

  function writeHistory(items) {
    storage.set(localStorage, HISTORY_KEY, JSON.stringify(items.slice(0, 20)));
  }

  function rememberJob(entry) {
    writeHistory([entry, ...readHistory().filter((item) => item.id !== entry.id)]);
    renderHistory();
  }

  function forgetJob(id) {
    writeHistory(readHistory().filter((item) => item.id !== id));
    renderHistory();
  }

  function updateJob(id, patch) {
    const items = readHistory();
    const item = items.find((entry) => entry.id === id);
    if (!item) return;
    Object.assign(item, patch);
    if (isFinal(item.status) && !item.finishedAt) item.finishedAt = Date.now();
    writeHistory(items);
    renderHistory();
  }

  function isFinal(status) {
    return ['done', 'failed', 'canceled', 'rejected', 'expired'].includes(status);
  }

  function isLive(item) {
    return busy && item.id === currentJobId;
  }

  function jobStatus(item) {
    if (isLive(item)) return 'running';
    const status = item.status || 'done';
    const ttl = (serverConfig && serverConfig.job_ttl_seconds) || 0;
    // Retention starts when the job finishes. The server sweeps every quarter
    // of it, so a transcript can stay available that much longer.
    if (ttl > 0 && isFinal(status) && item.finishedAt && Date.now() - item.finishedAt > ttl * 1250) return 'expired';
    return status;
  }

  function liveIcon(item) {
    return statusIcon(jobStatus(item), isLive(item) ? liveFraction : 0);
  }

  function dayLabel(at) {
    const day = (value) => new Date(value).setHours(0, 0, 0, 0);
    const days = Math.round((day(Date.now()) - day(at)) / 86400000);
    return days <= 0 ? 'Today' : days === 1 ? 'Yesterday' : 'Earlier';
  }

  // Rows are reused: replacing one between mousedown and mouseup would lose
  // the click, and progress can arrive many times a second.
  const historyRows = new Map();
  let historyShape = '';

  function renderHistory() {
    const items = readHistory();
    const ttl = (serverConfig && serverConfig.job_ttl_seconds) || 0;
    const groups = new Map();
    items.forEach((item) => {
      const label = dayLabel(item.at);
      if (!groups.has(label)) groups.set(label, []);
      groups.get(label).push(item);
    });
    items.forEach(updateHistoryRow);
    const shape = JSON.stringify([...groups].map(([label, entries]) => [label, entries.map((item) => item.id)]));
    if (shape !== historyShape) {
      historyShape = shape;
      const ids = new Set(items.map((item) => item.id));
      [...historyRows.keys()].forEach((id) => { if (!ids.has(id)) historyRows.delete(id); });
      const list = $('history-list');
      const focused = list.contains(document.activeElement) ? document.activeElement : null;
      list.replaceChildren(...[...groups].map(([label, entries]) => {
        const group = el('section', 'side-group');
        group.append(el('h2', 'side-label', label));
        const ul = el('ul');
        ul.append(...entries.map((item) => historyRows.get(item.id).li));
        group.append(ul);
        return group;
      }));
      if (focused && focused.isConnected) focused.focus({ preventScroll: true });
    }
    $('history-note').textContent = items.length === 0
      ? 'Transcripts you start in this browser appear here.'
      : ttl ? `Kept on the server for ${formatDuration(ttl)} after finishing.` : '';
  }

  function createHistoryRow(item) {
    const li = el('li');
    const button = el('button', 'side-link history-item');
    button.type = 'button';
    button.dataset.id = item.id;
    const name = el('span', 'history-name');
    const meta = el('span', 'side-meta tabular');
    button.append(statusIcon('done'), name, meta);
    li.append(button);
    const row = { li, button, name, meta, item, icon: '' };
    button.addEventListener('click', () => openFromHistory(row.item));
    historyRows.set(item.id, row);
    return row;
  }

  function updateHistoryRow(item) {
    const row = historyRows.get(item.id) || createHistoryRow(item);
    const status = jobStatus(item);
    row.item = item;
    row.button.dataset.status = status;
    const model = MODELS[item.model] ? MODELS[item.model].label : '';
    const title = [item.name, STATUS_LABELS[status], model, item.duration ? formatDuration(item.duration) : ''].filter(Boolean).join('\n');
    if (row.button.title !== title) row.button.title = title;
    if (!workspace.hidden && item.id === currentJobId) row.button.setAttribute('aria-current', 'page');
    else row.button.removeAttribute('aria-current');
    const look = `${status} ${isLive(item) ? liveFraction : 0}`;
    if (row.icon !== look) {
      row.icon = look;
      row.button.firstElementChild.replaceWith(liveIcon(item));
    }
    const name = (item.name || 'Recording').replace(/\.[^.]+$/, '');
    if (row.name.textContent !== name) row.name.textContent = name;
    const when = relativeTime(item.at);
    if (row.meta.textContent !== when) row.meta.textContent = when;
  }

  function openFromHistory(item) {
    // Reopening the job on screen would lose its audio; just show it.
    if (item.id === currentJobId && (currentResult || currentStream)) showView('job');
    else openJob(item.id, item.name, item.model);
  }

  // Jobs whose outcome this browser has not seen: left running in the
  // background, waiting for review, or finished while the page was closed.
  let refreshing = false;
  async function refreshStatuses() {
    if (refreshing) return;
    refreshing = true;
    const known = { succeeded: 'done', failed: 'failed', canceled: 'canceled', awaiting_review: 'review', expired: 'expired' };
    const open = readHistory().filter((item) => item.id !== currentJobId && !(isFinal(item.status) && item.finishedAt));
    await Promise.all(open.map(async (item) => {
      try {
        const response = await fetch(`/api/jobs/${encodeURIComponent(item.id)}`);
        const found = response.ok ? known[(await response.json()).status] || 'running' : '';
        const status = response.status === 404 ? 'expired' : found;
        // Once on screen, the job's own stream is the source of truth.
        if (!status || item.id === currentJobId) return;
        // The server records a rejected review as a failed job.
        updateJob(item.id, { status: item.status === 'rejected' && status !== 'expired' ? 'rejected' : status });
      } catch (_) { /* offline: keep the last known state */ }
    }));
    refreshing = false;
  }
  window.setInterval(() => {
    if (document.hidden) return;
    renderHistory();
    void refreshStatuses();
  }, 20000);
  document.addEventListener('visibilitychange', () => { if (!document.hidden) void refreshStatuses(); });

  // ---- Progress -----------------------------------------------------------------
  const STAGE_ORDER = ['upload', 'prepare', 'transcribe', 'review', 'finish'];

  function stageFor(message) {
    if (/judg/i.test(message)) return 'review';
    if (/validat|processing audio|splitting/i.test(message)) return 'prepare';
    if (/transcrib|candidate/i.test(message)) return 'transcribe';
    if (/merging|formatting|analyzing|finaliz|complete/i.test(message)) return 'finish';
    return null;
  }

  // "Transcribed 4/9 chunks..." -> "Transcribed 4 of 9 chunks…"
  function friendlyStage(message) {
    return String(message || '').replace(/(\d+)\/(\d+)/, '$1 of $2').replace(/\.\.\.$/, '…').replace(/!$/, '');
  }

  function setStage(stage, detail) {
    const index = STAGE_ORDER.indexOf(stage);
    if (index < 0) return;
    if (stage === 'review') stageItems[3].hidden = false;
    const before = stageIndex;
    stageIndex = Math.max(stageIndex, index);
    if (stageIndex !== before) {
      if (before >= 0) stampStage(before);
      stageStarted = Date.now();
    }
    stageItems.forEach((item) => {
      const itemIndex = STAGE_ORDER.indexOf(item.dataset.stage);
      item.dataset.state = itemIndex < stageIndex ? 'done' : itemIndex === stageIndex ? 'active' : '';
      const detailEl = item.querySelector('.stage-detail');
      if (itemIndex === stageIndex && detail !== undefined) detailEl.textContent = detail;
      if (itemIndex < stageIndex) detailEl.textContent = '';
    });
  }

  function stampStage(index) {
    const item = stageItems.find((node) => node.dataset.stage === STAGE_ORDER[index]);
    if (item && stageStarted) item.querySelector('.stage-time').textContent = formatDuration((Date.now() - stageStarted) / 1000);
  }

  function finishStages(failed) {
    if (failed) runBack.hidden = false;
    stageItems.forEach((item) => {
      const itemIndex = STAGE_ORDER.indexOf(item.dataset.stage);
      if (failed) {
        if (itemIndex === stageIndex) item.dataset.state = 'failed';
        return;
      }
      item.dataset.state = 'done';
      item.querySelector('.stage-detail').textContent = '';
    });
  }

  function resetStages(showReview) {
    stageIndex = -1;
    stageItems.forEach((item) => {
      item.dataset.state = '';
      item.querySelector('.stage-detail').textContent = '';
      item.querySelector('.stage-time').textContent = '';
    });
    stageStarted = 0;
    renderChunks(0, 0);
    stageItems[3].hidden = !showReview;
  }

  // One cell per chunk of audio.
  function renderChunks(done, total) {
    const chunks = $('chunks');
    chunks.hidden = total < 2;
    if (total < 2) return;
    if (chunks.childElementCount !== Math.min(total, 120)) chunks.replaceChildren(...Array.from({ length: Math.min(total, 120) }, () => el('i')));
    const filled = Math.round((done / total) * chunks.childElementCount);
    [...chunks.children].forEach((cell, index) => cell.classList.toggle('done', index < filled));
  }

  function setProgress(fraction) {
    const percent = Math.round(Math.min(1, Math.max(0, fraction)) * 100);
    liveFraction = percent / 100;
    $('progress-percent').textContent = `${percent}%`;
    progressBar.style.transform = `scaleX(${percent / 100})`;
    progress.setAttribute('aria-valuenow', String(percent));
    const row = historyRows.get(currentJobId);
    if (row) updateHistoryRow(row.item);
    syncDocumentTitle();
  }

  function setConnectionState(text, state = 'pending') {
    connectionState.textContent = text;
    connectionState.dataset.state = state;
  }

  function setJobReference(jobId) {
    jobReference.textContent = jobId ? jobId.slice(0, 12) : '';
    jobReference.hidden = !jobId;
  }

  jobReference.addEventListener('click', async () => {
    if (!currentJobId) return;
    try {
      await navigator.clipboard.writeText(currentJobId);
      toast('Job ID copied');
    } catch (_) { /* clipboard unavailable */ }
  });

  function startElapsed() {
    stopElapsed();
    const started = Date.now();
    const tick = () => { progressElapsed.textContent = formatClock((Date.now() - started) / 1000); };
    tick();
    elapsedTimer = window.setInterval(tick, 1000);
  }

  function stopElapsed() {
    if (elapsedTimer) window.clearInterval(elapsedTimer);
    elapsedTimer = null;
  }

  function appendLog(line) {
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const item = el('li');
    item.append(el('span', 'feed-text', friendlyStage(line)), el('span', 'feed-time tabular', time));
    logEl.prepend(item);
  }

  function showRunView(name, model, review) {
    audioPlayer.pause();
    statusCard.hidden = false;
    resultCard.hidden = true;
    runBack.hidden = true;
    cancelJob.hidden = true;
    runTitle.textContent = 'Transcribing';
    $('run-crumb').textContent = name || 'Transcript';
    $('run-model').textContent = MODELS[model] ? MODELS[model].label : '';
    logEl.replaceChildren();
    progressElapsed.textContent = '';
    setProgress(0);
    resetStages(review);
    setJobReference('');
    showView('job', runTitle);
  }

  function setBusy(value, label) {
    busy = value;
    spinner.hidden = !value;
    submitLabel.textContent = label || 'Transcribe';
    updateSubmitState();
    renderHistory();
    syncDocumentTitle();
  }

  // ---- Submit -------------------------------------------------------------------
  // XMLHttpRequest rather than fetch: fetch cannot report upload progress.
  function uploadJob(formData, idempotencyKey, onProgress) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', '/api/jobs');
      Object.entries(authHeaders({ 'Idempotency-Key': idempotencyKey })).forEach(([name, value]) => xhr.setRequestHeader(name, value));
      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable) onProgress(event.loaded / event.total);
      });
      xhr.addEventListener('load', () => resolve({ status: xhr.status, text: xhr.responseText }));
      xhr.addEventListener('error', () => reject(new Error('The connection dropped during the upload.')));
      xhr.send(formData);
    });
  }

  // The same Idempotency-Key makes a retry return the job the first attempt
  // may already have created.
  async function createJob(formData, idempotencyKey, onProgress, onRetry) {
    for (let attempt = 0; attempt < 2; attempt += 1) {
      try {
        return await uploadJob(formData, idempotencyKey, onProgress);
      } catch (error) {
        if (attempt > 0) throw error;
        onRetry();
        await new Promise((resolve) => window.setTimeout(resolve, 450));
      }
    }
    throw new Error('The upload failed.');
  }

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const file = fileInput.files && fileInput.files[0];
    if (!file) {
      showFormError('Add a recording first.');
      return;
    }
    const limit = serverConfig && serverConfig.max_upload_bytes;
    if (limit && file.size > limit) {
      showFormError(`This file is ${formatBytes(file.size)}; the server accepts up to ${formatBytes(limit)}.`);
      return;
    }
    showFormError('');
    runToken += 1;
    const run = runToken;
    const onScreen = () => run === runToken;
    uploading = file.name;
    const model = selectedModel();
    if (currentStream) {
      currentStream.close();
      currentStream = null;
    }
    if (jobAudioURL && jobAudioURL !== audioURL) URL.revokeObjectURL(jobAudioURL);
    jobAudioURL = audioURL;
    jobPeaks = filePeaks;
    audioPlayer.currentTime = 0;
    clearReviewState();
    currentResult = null;
    currentJobId = null;
    currentJobName = file.name;
    showRunView(file.name, model, reviewEnabled());
    setStage('upload', formatBytes(file.size));
    progressLabel.textContent = 'Uploading…';
    setConnectionState('Uploading', 'pending');
    startElapsed();
    setBusy(true, 'Uploading…');
    submitBtn.disabled = true;

    const data = new FormData(form);
    ['use_judge_pipeline', 'agentic_mode', 'auto_format', 'remove_fillers', 'speech_review'].forEach((name) => {
      const input = form.elements.namedItem(name);
      if (input && !input.disabled) data.set(name, input.checked ? 'true' : 'false');
    });

    try {
      const response = await createJob(data, crypto.randomUUID(), (fraction) => {
        if (!onScreen()) return;
        setStage('upload', `${Math.round(fraction * 100)}%`);
        setProgress(fraction * 0.08);
      }, () => {
        if (onScreen()) progressLabel.textContent = 'Connection dropped; checking whether the upload arrived…';
      });
      if (response.status < 200 || response.status >= 300) throw new Error(messageFrom(response.text, response.status));
      const started = JSON.parse(response.text);
      const entry = { id: started.job_id, name: file.name, model, at: Date.now(), status: 'running' };
      if (!onScreen()) {
        // Another transcript is open: the job carries on in the sidebar.
        rememberJob(entry);
        toast(`${file.name} uploaded. Transcribing in the background.`);
        return;
      }
      currentJobId = started.job_id;
      setJobReference(currentJobId);
      rememberJob(entry);
      if (started.idempotent_replay) appendLog('Recovered a submission the server had already accepted.');
      appendLog('Upload complete.');
      setStage('prepare');
      progressLabel.textContent = 'Waiting for a free worker…';
      setConnectionState('Queued', 'pending');
    } catch (error) {
      if (!onScreen()) {
        toast(`Could not upload ${file.name}: ${error.message}`);
        return;
      }
      runTitle.textContent = 'Upload failed';
      progressLabel.textContent = error.message;
      appendLog(`Error: ${error.message}`);
      setConnectionState('Failed', 'error');
      finishStages(true);
      showFormError(error.message);
      stopElapsed();
      setBusy(false);
      return;
    } finally {
      if (onScreen()) uploading = '';
    }
    connectStream(currentJobId);
  });

  // ---- Live updates ------------------------------------------------------------------
  function openJob(jobId, name, model = '') {
    runToken += 1;
    if (uploading) toast(`${uploading} is still uploading. It will appear in the list.`);
    uploading = '';
    if (currentStream) currentStream.close();
    currentStream = null;
    clearReviewState();
    currentResult = null;
    currentJobName = name || '';
    if (jobAudioURL && jobAudioURL !== audioURL) URL.revokeObjectURL(jobAudioURL);
    jobAudioURL = null;
    jobPeaks = null;
    currentJobId = jobId;
    showRunView(name, model, false);
    progressLabel.textContent = 'Loading…';
    connectStream(jobId, true);
    // The job that just left the screen no longer reports through its stream.
    void refreshStatuses();
  }

  function connectStream(jobId, resumed = false) {
    if (currentStream) currentStream.close();
    currentJobId = jobId;
    setJobReference(jobId);
    cancelJob.hidden = false;
    setBusy(true, 'Transcribing…');
    if (resumed) {
      stopElapsed();
      progressElapsed.textContent = '';
    }
    // Remember the job on screen so a reload brings it back.
    storage.set(sessionStorage, ACTIVE_JOB_KEY, jobId);
    const stream = new EventSource(`/api/jobs/${encodeURIComponent(jobId)}/stream`);
    currentStream = stream;
    updateSubmitState();
    stream.onopen = () => setConnectionState(awaitingReview ? 'Review' : 'Live', 'active');
    stream.addEventListener('progress', (event) => {
      const data = JSON.parse(event.data);
      setProgress(data.fraction);
      progressLabel.textContent = friendlyStage(data.stage);
      const stage = stageFor(data.stage);
      const counts = /(\d+)\/(\d+)/.exec(data.stage);
      if (counts) renderChunks(Number(counts[1]), Number(counts[2]));
      if (stage) setStage(stage, counts ? `${counts[1]} of ${counts[2]}` : undefined);
      setConnectionState('Running', 'active');
      appendLog(data.stage);
    });
    stream.addEventListener('result', (event) => {
      const payload = JSON.parse(event.data);
      currentResult = payload;
      renderResult(payload);
      const review = payload.result.agent_run && payload.result.agent_run.human_review;
      if (review && review.status === 'required') {
        showReview(jobId, review.reasons || []);
        return;
      }
      updateJob(jobId, { status: 'done' });
      finishStages(false);
      setProgress(1);
      closeStream(stream);
      setBusy(false);
      statusCard.hidden = true;
      if (!resumed) toast('Transcript ready');
    });
    stream.addEventListener('review-required', (event) => {
      const data = JSON.parse(event.data);
      showReview(jobId, data.reasons || [], data.review_expires_at);
    });
    stream.addEventListener('review-completed', (event) => completeReview(jobId, JSON.parse(event.data).decision));
    stream.addEventListener('canceled', () => {
      updateJob(jobId, { status: 'canceled' });
      runTitle.textContent = 'Canceled';
      progressLabel.textContent = 'This transcription was canceled.';
      setConnectionState('Canceled', 'error');
      finishStages(true);
      closeStream(stream);
      setBusy(false);
    });
    stream.addEventListener('error-event', (event) => {
      let message = 'Something went wrong.';
      try { message = JSON.parse(event.data).message; } catch (_) { /* keep default */ }
      updateJob(jobId, { status: 'failed' });
      runTitle.textContent = 'Transcription failed';
      progressLabel.textContent = message;
      appendLog(`Error: ${message}`);
      setConnectionState('Failed', 'error');
      finishStages(true);
      closeStream(stream);
      setBusy(false);
    });
    stream.onerror = () => { void handleStreamError(stream, jobId); };
  }

  function closeStream(stream) {
    stream.close();
    if (currentStream === stream) currentStream = null;
    cancelJob.hidden = true;
    stopElapsed();
    updateSubmitState();
  }

  async function handleStreamError(stream, jobId) {
    if (currentStream !== stream || checkingStreams.has(stream)) return;
    checkingStreams.add(stream);
    const onScreen = () => currentStream === stream;
    setConnectionState('Reconnecting', 'pending');
    try {
      const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`);
      if (response.status === 404) {
        forgetJob(jobId);
        if (!onScreen()) return;
        markJobUnavailable(
          jobId,
          awaitingReview ? 'The review window has closed.' : 'This transcript is no longer on the server. Transcripts are kept for a limited time after they finish.',
          awaitingReview ? 'Expired' : 'Unavailable'
        );
        return;
      }
      if (!response.ok) throw new Error(`status check returned ${response.status}`);
      const state = await response.json();
      if (!onScreen()) return;
      if (state.status === 'awaiting_review') {
        showReview(jobId, [], state.review_expires_at);
        return;
      }
      if (state.status === 'expired') {
        markJobUnavailable(jobId, 'The review window has closed.', 'Expired');
        return;
      }
      progressLabel.textContent = 'Connection dropped; reconnecting…';
    } catch (_) {
      if (!onScreen()) return;
      setConnectionState('Offline', 'error');
      progressLabel.textContent = 'Connection dropped; retrying…';
    } finally {
      checkingStreams.delete(stream);
    }
  }

  // ---- Human review ---------------------------------------------------------------
  function clearReviewTimer() {
    if (reviewTimer) window.clearInterval(reviewTimer);
    reviewTimer = null;
    reviewExpiresAt = null;
  }

  function clearReviewState() {
    awaitingReview = false;
    clearReviewTimer();
    reviewPanel.hidden = true;
    reviewActions.hidden = false;
    reviewReasons.textContent = '';
    reviewDeadline.textContent = '';
    reviewDeadline.hidden = true;
    reviewNote.value = '';
    reviewNote.disabled = false;
    acceptReview.disabled = false;
    rejectReview.disabled = false;
  }

  function updateReviewDeadline() {
    if (!reviewExpiresAt || !awaitingReview) return;
    const remaining = reviewExpiresAt.getTime() - Date.now();
    if (remaining <= 0) {
      markJobUnavailable(currentJobId, 'The review window has closed.', 'Expired');
      return;
    }
    const due = reviewExpiresAt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    reviewDeadline.textContent = `Decide by ${due}, ${formatDuration(remaining / 1000)} left`;
    reviewDeadline.hidden = false;
  }

  function setReviewDeadline(value) {
    if (!value) return;
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return;
    if (reviewTimer) window.clearInterval(reviewTimer);
    reviewExpiresAt = parsed;
    updateReviewDeadline();
    reviewTimer = window.setInterval(updateReviewDeadline, 1000);
  }

  function showReview(jobId, reasons, expiresAt) {
    awaitingReview = true;
    updateJob(jobId, { status: 'review' });
    reviewPanel.hidden = false;
    reviewActions.hidden = false;
    reviewNote.disabled = false;
    if (reasons.length || !reviewReasons.childElementCount) {
      const items = reasons.length
        ? reasons
        : ['Some passages were disputed. Check them before accepting this transcript.'];
      reviewReasons.replaceChildren(...items.map((reason) => {
        const item = document.createElement('li');
        item.textContent = reason;
        return item;
      }));
    }
    if (expiresAt) setReviewDeadline(expiresAt);
    cancelJob.hidden = true;
    statusCard.hidden = true;
    finishStages(false);
    stopElapsed();
    setBusy(false);
  }

  function markJobUnavailable(jobId, message, badge = 'Unavailable') {
    const wasAwaitingReview = awaitingReview;
    updateJob(jobId, { status: 'expired' });
    awaitingReview = false;
    clearReviewTimer();
    if (wasAwaitingReview) {
      reviewActions.hidden = true;
      reviewNote.disabled = true;
      reviewDeadline.hidden = false;
      reviewDeadline.textContent = message;
    } else {
      statusCard.hidden = false;
      resultCard.hidden = true;
      runTitle.textContent = 'Not available';
      progressLabel.textContent = message;
      setConnectionState(badge, 'error');
      finishStages(true);
    }
    appendLog(message);
    if (currentStream) closeStream(currentStream);
    storage.remove(sessionStorage, ACTIVE_JOB_KEY);
    stopElapsed();
    setBusy(false);
  }

  // announce is false for decisions replayed from the job's history.
  function completeReview(jobId, decision, announce = false) {
    awaitingReview = false;
    updateJob(jobId, { status: decision === 'accept' ? 'done' : 'rejected' });
    clearReviewTimer();
    reviewPanel.hidden = true;
    if (currentStream) closeStream(currentStream);
    setBusy(false);
    setRowStatus(decision === 'accept' ? 'Done' : 'Rejected', decision === 'accept' ? 'done' : 'rejected');
    if (announce) toast(decision === 'accept' ? 'Transcript accepted' : 'Transcript rejected');
  }

  async function submitReview(decision) {
    const jobId = currentJobId;
    const run = runToken;
    const onScreen = () => run === runToken;
    acceptReview.disabled = true;
    rejectReview.disabled = true;
    try {
      const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/review`, {
        method: 'POST',
        headers: authHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ decision, note: reviewNote.value.trim() }),
      });
      if (!response.ok) {
        const error = new Error(await errorMessage(response));
        error.status = response.status;
        throw error;
      }
      if (onScreen()) {
        completeReview(jobId, decision, true);
      } else {
        updateJob(jobId, { status: decision === 'accept' ? 'done' : 'rejected' });
        toast(decision === 'accept' ? 'Transcript accepted' : 'Transcript rejected');
      }
    } catch (error) {
      if (onScreen() && [404, 409, 410].includes(error.status)) {
        markJobUnavailable(jobId, 'Review is no longer available for this transcript.', 'Expired');
        return;
      }
      toast(`Could not save the review: ${error.message}`);
      if (!onScreen()) void refreshStatuses();
    } finally {
      if (onScreen() && awaitingReview) {
        acceptReview.disabled = false;
        rejectReview.disabled = false;
      }
    }
  }

  cancelJob.addEventListener('click', async () => {
    if (!currentJobId) return;
    const run = runToken;
    cancelJob.disabled = true;
    try {
      const response = await fetch(`/api/jobs/${encodeURIComponent(currentJobId)}/cancel`, {
        method: 'POST',
        headers: authHeaders({ 'Content-Type': 'application/json' }),
        body: '{}',
      });
      if (!response.ok) throw new Error(await errorMessage(response));
      if (run !== runToken) return;
      progressLabel.textContent = 'Canceling…';
      setConnectionState('Canceling', 'pending');
    } catch (error) {
      toast(`Could not cancel: ${error.message}`);
    } finally {
      cancelJob.disabled = false;
    }
  });
  acceptReview.addEventListener('click', () => submitReview('accept'));
  rejectReview.addEventListener('click', () => submitReview('reject'));

  // ---- Result -----------------------------------------------------------------------
  function displayName(speaker) {
    return (speakerNames.get(speaker) || '').trim() || speaker;
  }

  function colorFor(speaker) {
    if (!speakerColors.has(speaker)) speakerColors.set(speaker, SPEAKER_COLORS[speakerColors.size % SPEAKER_COLORS.length]);
    return speakerColors.get(speaker);
  }

  function speakersOf(segments) {
    const seen = [];
    segments.forEach((segment) => {
      if (segment.speaker !== GAP_SPEAKER && !seen.includes(segment.speaker)) seen.push(segment.speaker);
    });
    return seen;
  }

  function initials(name) {
    const words = name.trim().split(/\s+/);
    if (words.length > 1 && /^\d+$/.test(words[words.length - 1])) return words[words.length - 1];
    return words.slice(0, 2).map((word) => [...word][0] || '').join('').toUpperCase();
  }

  function avatar(speaker) {
    const node = el('span', 'avatar', initials(displayName(speaker)));
    node.dataset.avatar = speaker;
    node.style.setProperty('--speaker', colorFor(speaker));
    node.setAttribute('aria-hidden', 'true');
    return node;
  }

  function renderSummary(result) {
    const segments = result.segments || [];
    const metadata = result.metadata || {};
    const review = result.agent_run && result.agent_run.human_review;
    const waiting = Boolean(review && review.status === 'required');
    const rows = [
      ['Status', waiting ? 'Needs review' : 'Done'],
      ['Length', formatDuration(metadata.duration)],
      ['Model', (MODELS[result.model_used] && MODELS[result.model_used].label) || result.model_used],
      ['Took', formatDuration(result.processing_time)],
      ['Words', segments.filter((segment) => segment.speaker !== GAP_SPEAKER).reduce((sum, segment) => sum + (segment.text || '').split(/\s+/).filter(Boolean).length, 0).toLocaleString()],
      ['Reviewed', result.judge_used ? 'By a judge' : ''],
      ['Quality', result.judge_used && result.quality && result.quality.overall_score ? `${Math.round(result.quality.overall_score)} of 100` : ''],
    ].filter(([, value]) => value);
    $('result-summary').replaceChildren(...rows.map(([label, value]) => {
      const row = el('div', 'prop');
      const text = el('span', 'prop-text tabular');
      if (label === 'Status') text.append(statusIcon(waiting ? 'review' : 'done'));
      text.append(el('span', '', value));
      row.append(el('span', 'prop-label', label), text);
      return row;
    }));
    const gaps = segments.filter((segment) => segment.speaker === GAP_SPEAKER);
    const gapJump = $('gap-jump');
    gapJump.hidden = gaps.length === 0;
    gapJump.textContent = gaps.length === 1 ? 'One part has no transcript. Go to it' : `${gaps.length} parts have no transcript. Go to the first`;
    gapJump.onclick = () => focusRow(segments.indexOf(gaps[0]));
    updateJob(currentJobId, { duration: metadata.duration });
  }

  function setRowStatus(label, status) {
    const text = $('result-summary').querySelector('.prop-text');
    if (text) text.replaceChildren(statusIcon(status), el('span', '', label));
  }

  function segmentEnd(segments, index) {
    const segment = segments[index];
    const start = segmentStart(segment);
    if (typeof segment.end === 'number' && segment.end > start) return segment.end;
    const next = segments[index + 1];
    const estimate = start + Math.max(1, (segment.text || '').length / 15);
    return next ? Math.min(estimate, Math.max(segmentStart(next), start + 0.5)) : estimate;
  }

  function renderTimeline(result) {
    const segments = result.segments || [];
    const metadata = result.metadata || {};
    const lastEnd = segments.length ? segmentEnd(segments, segments.length - 1) : 0;
    timelineDuration = Math.max(metadata.duration || 0, lastEnd, 1);
    const lanes = new Map();
    speakersOf(segments).forEach((speaker) => {
      const lane = el('div', 'lane');
      lane.style.setProperty('--speaker', colorFor(speaker));
      lanes.set(speaker, lane);
    });
    let gapLane = null;
    segments.forEach((segment, index) => {
      const start = segmentStart(segment);
      const end = segmentEnd(segments, index);
      const isGap = segment.speaker === GAP_SPEAKER;
      if (isGap && !gapLane) {
        gapLane = el('div', 'lane gaps');
        gapLane.style.setProperty('--speaker', 'var(--gap)');
      }
      const bar = el('span', isGap ? 'bar gap' : 'bar');
      bar.style.left = `${(start / timelineDuration) * 100}%`;
      bar.style.width = `${(Math.max(end - start, 0.2) / timelineDuration) * 100}%`;
      bar.title = `${formatClock(start)} ${isGap ? 'No transcript' : displayName(segment.speaker)}: ${segment.text.slice(0, 120)}`;
      bar.dataset.index = String(index);
      (isGap ? gapLane : lanes.get(segment.speaker)).append(bar);
    });
    timelineLanes.replaceChildren(...lanes.values(), ...(gapLane ? [gapLane] : []));
    timeline.toggleAttribute('data-dense', timelineLanes.childElementCount > 6);
    waveColors = [];
    wave.hidden = !jobPeaks || !jobAudioURL;
    timeline.classList.toggle('has-wave', !wave.hidden);
    if (!wave.hidden) paintPlayer();
    seek.max = String(timelineDuration);
    seek.hidden = !jobAudioURL;
    $('player-total').textContent = formatClock(timelineDuration);
    // A late waveform redraws a player that may already be part-way through.
    if (jobAudioURL && audioPlayer.src === jobAudioURL && audioPlayer.currentTime > 0) {
      placePlayhead();
    } else {
      seek.value = '0';
      seek.setAttribute('aria-valuetext', formatClock(0));
      playerCurrent.textContent = formatClock(0);
      playhead.hidden = true;
    }
    timelineHint.textContent = jobAudioURL
      ? ''
      : 'Click a bar to jump to that line. To play along, transcribe a file in this tab.';
  }

  timelineLanes.addEventListener('click', (event) => {
    if (!currentResult) return;
    const bar = event.target.closest('.bar');
    let seconds;
    if (bar) {
      seconds = segmentStart(currentResult.result.segments[Number(bar.dataset.index)]);
    } else {
      const lane = event.target.closest('.lane');
      if (!lane) return;
      const rect = lane.getBoundingClientRect();
      seconds = ((event.clientX - rect.left) / rect.width) * timelineDuration;
    }
    jumpTo(seconds);
  });

  function rowAt(seconds) {
    let found = null;
    for (const row of segmentRows) {
      if (row.start <= seconds + 0.05) found = row;
      else break;
    }
    return found || segmentRows[0] || null;
  }

  function jumpTo(seconds) {
    if (jobAudioURL) {
      audioPlayer.currentTime = Math.max(0, seconds);
      void audioPlayer.play().catch(() => {});
    }
    const target = rowAt(seconds);
    if (!target) return;
    clearSearch();
    activateTab(tabs[0]);
    target.row.scrollIntoView({ block: 'center', behavior: scrollBehavior() });
    target.row.classList.remove('flash');
    void target.row.offsetWidth;
    target.row.classList.add('flash');
  }

  function renderSpeakers(segments) {
    const talk = new Map();
    let total = 0;
    segments.forEach((segment, index) => {
      if (segment.speaker === GAP_SPEAKER) return;
      const seconds = Math.max(0, segmentEnd(segments, index) - segmentStart(segment));
      talk.set(segment.speaker, (talk.get(segment.speaker) || 0) + seconds);
      total += seconds;
    });
    const speakers = speakersOf(segments);
    $('share-bar').replaceChildren(...speakers.map((speaker) => {
      const part = el('i');
      part.style.setProperty('--speaker', colorFor(speaker));
      part.style.flexGrow = String(talk.get(speaker) || 0.01);
      return part;
    }));
    speakerList.replaceChildren(...speakers.map((speaker) => {
      const item = el('li');
      const button = el('button', 'speaker-button');
      button.type = 'button';
      button.dataset.speaker = speaker;
      button.title = 'Rename';
      const share = total > 0 ? Math.round((talk.get(speaker) / total) * 100) : 0;
      button.setAttribute('aria-label', `Rename ${displayName(speaker)}, ${share}% of talking time`);
      button.append(avatar(speaker), el('span', 'speaker-name', displayName(speaker)), el('span', 'side-meta tabular', `${share}%`));
      button.addEventListener('click', () => editSpeaker(item, speaker));
      item.append(button);
      return item;
    }));
  }

  function editSpeaker(item, speaker) {
    const input = el('input');
    input.type = 'text';
    input.value = speakerNames.get(speaker) || '';
    input.placeholder = speaker;
    input.maxLength = 120;
    input.setAttribute('aria-label', `New name for ${speaker}`);
    item.replaceChildren(input);
    input.focus();
    let done = false;
    const commit = (save, refocus) => {
      if (done) return;
      done = true;
      if (save) {
        speakerNames.set(speaker, input.value.trim());
        refreshSpeakerNames();
      }
      renderSpeakers(currentResult.result.segments || []);
      if (refocus) {
        const button = [...speakerList.querySelectorAll('.speaker-button')].find((node) => node.dataset.speaker === speaker);
        if (button) button.focus();
      }
    };
    input.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') { event.preventDefault(); commit(true, true); }
      if (event.key === 'Escape') { event.preventDefault(); commit(false, true); }
    });
    input.addEventListener('blur', () => commit(true, false));
  }

  function refreshSpeakerNames() {
    transcriptView.querySelectorAll('.segment-speaker[data-speaker]').forEach((node) => { node.textContent = displayName(node.dataset.speaker); });
    transcriptView.querySelectorAll('[data-avatar]').forEach((node) => { node.textContent = initials(displayName(node.dataset.avatar)); });
    if (!panes.subtitles.hidden) void refreshSubtitles();
  }

  function renderTranscript(segments) {
    const fragment = document.createDocumentFragment();
    let previousSpeaker = null;
    segmentRows = segments.map((segment, index) => {
      const row = el('li', 'segment');
      const gap = segment.speaker === GAP_SPEAKER;
      if (gap) row.classList.add('gap');
      else if (segment.speaker === previousSpeaker) row.classList.add('same-speaker');
      previousSpeaker = segment.speaker;
      row.style.setProperty('--speaker', gap ? 'var(--gap)' : colorFor(segment.speaker));
      const start = segmentStart(segment);
      const time = el('button', 'segment-time', formatClock(start));
      time.type = 'button';
      time.disabled = !jobAudioURL;
      time.title = jobAudioURL ? 'Play from here' : '';
      time.addEventListener('click', () => jumpTo(start));
      const speaker = el('span', 'segment-speaker', gap ? 'Gap' : displayName(segment.speaker));
      if (!gap) speaker.dataset.speaker = segment.speaker;
      const text = el('p', 'segment-text', segment.text);
      const copy = el('button', 'icon-button segment-copy');
      copy.type = 'button';
      copy.title = 'Copy line';
      copy.setAttribute('aria-label', 'Copy line');
      copy.append(icon('copy'));
      copy.addEventListener('click', () => copyLine(index));
      const mark = gap ? el('span', 'avatar gap-mark', '!') : avatar(segment.speaker);
      row.append(time, mark, speaker, text, copy);
      fragment.append(row);
      return { row, text, raw: segment.text, start, end: segmentEnd(segments, index) };
    });
    transcriptView.replaceChildren(fragment);
    activeRow = null;
    focusedRow = -1;
    applySearch();
  }

  async function copyText(text, message) {
    try {
      await navigator.clipboard.writeText(text);
      toast(message);
    } catch (_) {
      toast('The browser blocked copying.');
    }
  }

  function copyLine(index) {
    const segment = currentResult.result.segments[index];
    if (segment) void copyText(segment.text, 'Line copied');
  }

  function focusRow(index) {
    if (!segmentRows[index]) return;
    if (segmentRows[index].row.hidden) clearSearch();
    if (segmentRows[focusedRow]) segmentRows[focusedRow].row.classList.remove('focused');
    focusedRow = index;
    activateTab(tabs[0]);
    segmentRows[index].row.classList.add('focused');
    segmentRows[index].row.scrollIntoView({ block: 'nearest', behavior: scrollBehavior() });
  }

  function moveRow(step) {
    const start = focusedRow >= 0 ? focusedRow : segmentRows.findIndex((item) => item.row === activeRow) - (step > 0 ? 0 : -1);
    let index = focusedRow < 0 && !activeRow ? (step > 0 ? 0 : segmentRows.length - 1) : start + step;
    while (segmentRows[index] && segmentRows[index].row.hidden) index += step;
    focusRow(index);
  }


  function applySearch() {
    const query = searchInput.value.trim().toLowerCase();
    let matches = 0;
    let visible = 0;
    segmentRows.forEach(({ row, text, raw }) => {
      text.replaceChildren();
      if (!query) {
        text.textContent = raw;
        row.hidden = false;
        visible += 1;
        return;
      }
      const lower = raw.toLowerCase();
      let from = 0;
      let index = lower.indexOf(query);
      row.hidden = index < 0;
      if (index >= 0) visible += 1;
      while (index >= 0) {
        matches += 1;
        text.append(raw.slice(from, index), el('mark', '', raw.slice(index, index + query.length)));
        from = index + query.length;
        index = lower.indexOf(query, from);
      }
      text.append(raw.slice(from));
    });
    searchCount.textContent = query ? `${matches} match${matches === 1 ? '' : 'es'}` : '';
    const empty = transcriptView.querySelector('.no-results');
    if (query && visible === 0) {
      if (!empty) transcriptView.append(el('li', 'no-results', 'No lines match your search.'));
    } else if (empty) {
      empty.remove();
    }
  }
  searchInput.addEventListener('input', applySearch);

  function clearSearch() {
    if (!searchInput.value) return;
    searchInput.value = '';
    applySearch();
  }

  // timeupdate fires only ~4 times a second, so the playhead follows the audio
  // every frame while playing. Moving it with transform avoids layout work.
  function placePlayhead() {
    if (timelineDuration <= 0) return;
    const fraction = Math.min(1, audioPlayer.currentTime / timelineDuration);
    playhead.style.transform = `translateX(${timeline.clientWidth * fraction}px)`;
    paintPlayer();
    playhead.hidden = false;
    seek.value = String(audioPlayer.currentTime);
    const clock = formatClock(audioPlayer.currentTime);
    if (playerCurrent.textContent !== clock) {
      playerCurrent.textContent = clock;
      seek.setAttribute('aria-valuetext', clock);
    }
  }

  function syncPlayToggle() {
    const playing = !audioPlayer.paused;
    playToggle.classList.toggle('playing', playing);
    playToggle.setAttribute('aria-label', playing ? 'Pause' : 'Play');
  }

  function togglePlay() {
    if (audioPlayer.paused) void audioPlayer.play().catch(() => toast('This browser cannot play the recording.'));
    else audioPlayer.pause();
  }

  // Hovering the waveform shows who says what there.
  timeline.addEventListener('pointermove', (event) => {
    if (wave.hidden || event.pointerType === 'touch') return;
    const box = timeline.getBoundingClientRect();
    const fraction = Math.min(1, Math.max(0, (event.clientX - box.left) / box.width));
    const seconds = fraction * timelineDuration;
    const index = segmentRows.findIndex((row) => seconds >= row.start && seconds <= row.end);
    const segment = index >= 0 ? currentResult.result.segments[index] : null;
    waveTip.replaceChildren(el('span', 'tabular', formatClock(seconds)));
    if (segment) waveTip.append(el('strong', '', segment.speaker === GAP_SPEAKER ? 'Gap' : displayName(segment.speaker)), el('span', 'wave-quote', segment.text));
    waveTip.hidden = false;
    const half = waveTip.offsetWidth / 2;
    waveTip.style.left = `${Math.min(Math.max(half, fraction * box.width), box.width - half)}px`;
  });
  timeline.addEventListener('pointerleave', () => { waveTip.hidden = true; });

  playToggle.addEventListener('click', togglePlay);
  ['pause', 'ended', 'emptied'].forEach((name) => audioPlayer.addEventListener(name, syncPlayToggle));
  seek.addEventListener('input', () => {
    audioPlayer.currentTime = Number(seek.value);
    placePlayhead();
    const target = rowAt(Number(seek.value));
    if (target && followAudio.checked && !target.row.hidden) target.row.scrollIntoView({ block: 'nearest' });
  });
  playbackRate.addEventListener('change', () => { audioPlayer.playbackRate = Number(playbackRate.value); });
  // Loading a new recording resets the rate.
  audioPlayer.addEventListener('loadedmetadata', () => { audioPlayer.playbackRate = Number(playbackRate.value); });

  let playheadFrame = 0;
  function followPlayhead() {
    placePlayhead();
    playheadFrame = audioPlayer.paused ? 0 : window.requestAnimationFrame(followPlayhead);
  }
  audioPlayer.addEventListener('play', () => {
    syncPlayToggle();
    window.cancelAnimationFrame(playheadFrame);
    playheadFrame = window.requestAnimationFrame(followPlayhead);
  });

  audioPlayer.addEventListener('timeupdate', () => {
    const now = audioPlayer.currentTime;
    placePlayhead();
    let current = rowAt(now);
    if (current && now > current.end + 2) current = null;
    const row = current ? current.row : null;
    if (row === activeRow) return;
    if (activeRow) activeRow.classList.remove('playing');
    activeRow = row;
    if (row) {
      row.classList.add('playing');
      if (followAudio.checked && !audioPlayer.paused && !row.hidden) row.scrollIntoView({ block: 'nearest', behavior: scrollBehavior() });
    }
  });

  function renderResult(payload) {
    const result = payload.result;
    resultCard.hidden = false;
    // Leave the form alone if the user is already preparing the next recording.
    const focused = document.activeElement;
    if (!workspace.hidden && (!focused || focused === document.body || statusCard.contains(focused))) resultTitle.focus({ preventScroll: true });
    resultScroll.scrollTop = 0;
    const entry = readHistory().find((item) => item.id === currentJobId);
    $('result-crumb').textContent = entry ? `${dayLabel(entry.at) === 'Earlier' ? new Date(entry.at).toLocaleDateString() : dayLabel(entry.at)} at ${new Date(entry.at).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}` : 'Transcript';
    resultTitle.textContent = (currentJobName || (result.metadata && result.metadata.filename) || 'Transcript').replace(/\.[^.]+$/, '');
    resultTitle.title = resultTitle.textContent;
    const reviewed = Boolean(result.judge_used);
    tabs.forEach((tab) => { if (tab.hasAttribute('data-pipeline-only')) tab.hidden = !reviewed; });
    speakerNames.clear();
    speakerColors.clear();
    searchInput.value = '';

    renderSummary(result);
    renderTimeline(result);
    renderSpeakers(result.segments || []);
    renderTranscript(result.segments || []);

    player.hidden = !jobAudioURL;
    $('player-options').hidden = !jobAudioURL;
    if (jobAudioURL) {
      if (audioPlayer.src !== jobAudioURL) audioPlayer.src = jobAudioURL;
    } else {
      audioPlayer.removeAttribute('src');
    }

    panes.subtitles.textContent = payload.srt || '';
    panes.json.textContent = JSON.stringify(result, null, 2);
    const toolLines = (result.judge_tool_usage || []).map((tool) => `${tool.name}: ${tool.count}`);
    panes.judge.textContent = [
      ...((result.judge_notes || []).length ? result.judge_notes : ['No judge notes.']),
      ...(toolLines.length ? ['', 'Judge tools:', ...toolLines] : []),
    ].join('\n');
    panes.quality.textContent = JSON.stringify(result.quality, null, 2);
    panes.evidence.textContent = JSON.stringify(result.agent_run || {}, null, 2);
    activateTab(tabs[0]);
    renderHistory();
  }

  // ---- Tabs -------------------------------------------------------------------------
  function activateTab(tab, focus = false) {
    tabs.forEach((t) => {
      const active = t === tab;
      t.classList.toggle('active', active);
      t.setAttribute('aria-selected', String(active));
      t.tabIndex = active ? 0 : -1;
      panes[t.dataset.tab].hidden = !active;
    });
    $('search').hidden = tab.dataset.tab !== 'transcript';
    if (focus) tab.focus();
    if (tab.dataset.tab === 'subtitles') void refreshSubtitles();
  }

  tabs.forEach((tab) => {
    tab.addEventListener('click', () => activateTab(tab));
    tab.addEventListener('keydown', (event) => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault();
      const visible = tabs.filter((item) => !item.hidden);
      const index = visible.indexOf(tab);
      let next = index;
      if (event.key === 'ArrowRight') next = (index + 1) % visible.length;
      if (event.key === 'ArrowLeft') next = (index - 1 + visible.length) % visible.length;
      if (event.key === 'Home') next = 0;
      if (event.key === 'End') next = visible.length - 1;
      activateTab(visible[next], true);
    });
  });

  // ---- Export -----------------------------------------------------------------------
  function exportParams(format, timestamps) {
    const params = new URLSearchParams({ format, timestamps: timestamps ? 'true' : 'false' });
    speakerNames.forEach((name, speaker) => {
      if (name.trim()) params.append('speaker', `${speaker}=${name.trim()}`);
    });
    return params;
  }

  async function fetchExport(format, timestamps) {
    if (!currentJobId) throw new Error('Open a finished transcript first.');
    const response = await fetch(`/api/jobs/${encodeURIComponent(currentJobId)}/export?${exportParams(format, timestamps)}`, { cache: 'no-store' });
    if (!response.ok) throw new Error(await errorMessage(response));
    return response;
  }

  async function refreshSubtitles() {
    if (!currentJobId || !currentResult) return;
    const jobId = currentJobId;
    try {
      const text = await (await fetchExport('srt', true)).text();
      // Ignore a preview that finishes after the user opened another job.
      if (jobId === currentJobId) panes.subtitles.textContent = text;
    } catch (_) { /* keep the previous preview */ }
  }

  function renderFormats() {
    $('format-list').replaceChildren(...FORMATS.map((format) => {
      const button = el('button', 'menu-item');
      button.type = 'button';
      button.append(el('span', '', format.label), el('span', 'menu-meta', format.ext));
      button.addEventListener('click', () => {
        $('export-menu').hidePopover();
        void downloadAs(format.id);
      });
      return button;
    }));
  }

  function downloadName(response, format) {
    const header = response.headers.get('Content-Disposition') || '';
    const match = /filename\*=UTF-8''([^;]+)|filename="([^"]+)"/i.exec(header);
    if (match) return match[1] ? decodeURIComponent(match[1]) : match[2];
    return `transcript.${format}`;
  }

  async function downloadAs(format) {
    $('export-button').disabled = true;
    try {
      const response = await fetchExport(format, exportTimestamps.checked);
      const url = URL.createObjectURL(await response.blob());
      const link = el('a');
      link.href = url;
      link.download = downloadName(response, format);
      document.body.append(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
      toast(`Downloaded ${link.download}`);
    } catch (error) {
      toast(`Download failed: ${error.message}`);
    } finally {
      $('export-button').disabled = false;
    }
  }

  async function copyAs(format) {
    try {
      const response = await fetchExport(format, exportTimestamps.checked);
      await copyText(await response.text(), 'Copied to clipboard');
    } catch (error) {
      toast(`Copy failed: ${error.message}`);
    }
  }
  document.querySelectorAll('[data-copy]').forEach((button) => button.addEventListener('click', () => {
    $('copy-menu').hidePopover();
    void copyAs(button.dataset.copy);
  }));

  // ---- Command menu --------------------------------------------------------------
  const isMac = /mac|iphone|ipad/i.test(navigator.platform);
  const keyText = (text) => text.replace('mod', isMac ? '⌘' : 'Ctrl');
  document.querySelectorAll('kbd').forEach((node) => { node.textContent = keyText(node.textContent); });

  let paletteItems = [];
  let paletteIndex = 0;

  function commands() {
    const open = !workspace.hidden && !resultCard.hidden && Boolean(currentResult);
    const list = [
      { group: 'Commands', label: 'New transcription', icon: 'plus', kbd: 'N', run: () => showView('compose', fileTitle) },
      { group: 'Commands', label: 'Choose a recording', icon: 'upload', run: () => { showView('compose'); fileInput.click(); } },
    ];
    if (open) {
      if (jobAudioURL) list.push({ group: 'Commands', label: audioPlayer.paused ? 'Play recording' : 'Pause recording', icon: audioPlayer.paused ? 'play' : 'pause', kbd: 'Space', run: togglePlay });
      list.push({ group: 'Commands', label: 'Copy transcript as text', icon: 'copy', run: () => copyAs('txt') });
      FORMATS.forEach((format) => list.push({ group: 'Commands', label: `Export as ${format.label}`, icon: 'download', meta: format.ext, run: () => downloadAs(format.id) }));
    }
    list.push(
      { group: 'Commands', label: 'Show or hide details', icon: 'panel', kbd: ']', run: toggleDetails },
      { group: 'Commands', label: 'API keys', icon: 'key', run: () => openKeys() },
      { group: 'Commands', label: 'Keyboard shortcuts', icon: 'keyboard', kbd: '?', run: openShortcuts },
      ...Object.entries(THEMES).filter(([id]) => id !== currentTheme()).map(([id, label]) => ({ group: 'Commands', label: `Use ${label.toLowerCase()}`, icon: 'theme', run: () => setTheme(id) })),
    );
    const history = readHistory();
    if (history.some((item) => jobStatus(item) === 'expired')) {
      list.push({ group: 'Commands', label: 'Clear expired transcripts from the list', icon: 'x', run: () => {
        writeHistory(history.filter((item) => jobStatus(item) !== 'expired'));
        renderHistory();
      } });
    }
    history.forEach((item) => list.push({ group: 'Transcripts', label: item.name || 'Recording', status: jobStatus(item), fraction: isLive(item) ? liveFraction : 0, meta: relativeTime(item.at), run: () => openFromHistory(item) }));
    return list;
  }

  function lineMatches(query) {
    if (query.length < 2 || !currentResult || resultCard.hidden) return [];
    return (currentResult.result.segments || [])
      .map((segment, index) => ({ segment, index }))
      .filter(({ segment }) => segment.text.toLowerCase().includes(query))
      .slice(0, 6)
      .map(({ segment, index }) => ({
        group: 'In this transcript', label: segment.text, icon: 'wave', meta: formatClock(segmentStart(segment)),
        run: () => { showView('job'); focusRow(index); },
      }));
  }

  function renderPalette() {
    const query = paletteInput.value.trim().toLowerCase();
    const words = query.split(/\s+/).filter(Boolean);
    const found = commands().filter((item) => words.every((word) => item.label.toLowerCase().includes(word)));
    const order = query ? ['Transcripts', 'In this transcript', 'Commands'] : ['Commands', 'Transcripts'];
    const all = [...found, ...lineMatches(query)];
    paletteItems = order.flatMap((group) => all.filter((item) => item.group === group));
    paletteIndex = 0;
    const nodes = [];
    let group = '';
    paletteItems.forEach((item, index) => {
      if (item.group !== group) {
        group = item.group;
        nodes.push(el('p', 'menu-label', group));
      }
      const row = el('div', 'palette-item');
      row.id = `palette-${index}`;
      row.setAttribute('role', 'option');
      row.append(item.status ? statusIcon(item.status, item.fraction) : icon(item.icon), el('span', 'palette-text', item.label));
      if (item.meta) row.append(el('span', 'menu-meta tabular', item.meta));
      if (item.kbd) row.append(el('kbd', '', keyText(item.kbd)));
      row.addEventListener('pointermove', () => selectPalette(index, false));
      row.addEventListener('click', () => runPalette(index));
      nodes.push(row);
    });
    if (!nodes.length) nodes.push(el('p', 'palette-empty', 'Nothing matches.'));
    paletteList.replaceChildren(...nodes);
    selectPalette(0, true);
  }

  function selectPalette(index, scroll) {
    if (!paletteItems.length) {
      paletteInput.removeAttribute('aria-activedescendant');
      return;
    }
    paletteIndex = (index + paletteItems.length) % paletteItems.length;
    paletteList.querySelectorAll('.palette-item').forEach((row, i) => row.setAttribute('aria-selected', String(i === paletteIndex)));
    const row = $(`palette-${paletteIndex}`);
    paletteInput.setAttribute('aria-activedescendant', row.id);
    if (scroll) row.scrollIntoView({ block: 'nearest' });
  }

  function runPalette(index) {
    const item = paletteItems[index];
    if (!item) return;
    palette.close();
    item.run();
  }

  function openPalette() {
    if (palette.open) {
      palette.close();
      return;
    }
    if (document.querySelector('dialog[open]')) return;
    paletteInput.value = '';
    palette.showModal();
    renderPalette();
    paletteInput.focus();
  }

  function openShortcuts() {
    if (!document.querySelector('dialog[open]')) $('shortcuts-dialog').showModal();
  }

  $('open-palette').addEventListener('click', openPalette);
  $('shortcuts-button').addEventListener('click', openShortcuts);
  [palette, $('shortcuts-dialog')].forEach((dialog) => dialog.addEventListener('click', (event) => {
    if (event.target === dialog) dialog.close();
  }));
  paletteInput.addEventListener('input', renderPalette);
  paletteInput.addEventListener('keydown', (event) => {
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault();
      selectPalette(paletteIndex + (event.key === 'ArrowDown' ? 1 : -1), true);
    } else if (event.key === 'Enter') {
      event.preventDefault();
      runPalette(paletteIndex);
    }
  });

  // ---- Keyboard shortcuts ----------------------------------------------------------
  document.addEventListener('keydown', (event) => {
    if (event.defaultPrevented) return;
    const mod = isMac ? event.metaKey : event.ctrlKey;
    if (mod && event.key.toLowerCase() === 'k') {
      event.preventDefault();
      openPalette();
      return;
    }
    if (document.querySelector('dialog[open]')) return;
    if (mod && event.key === 'Enter' && !compose.hidden) {
      event.preventDefault();
      if (!submitBtn.disabled) form.requestSubmit(submitBtn);
      return;
    }
    if (event.key === 'Escape' && sidebar.classList.contains('open')) closeSidebar();
    if (event.metaKey || event.ctrlKey || event.altKey || isTyping(event.target) || document.querySelector('.menu:popover-open')) return;
    const key = event.key.length === 1 ? event.key.toLowerCase() : event.key;
    if (key === 'n') {
      event.preventDefault();
      showView('compose', fileTitle);
    } else if (key === '?') {
      openShortcuts();
    } else if (key === ']') {
      toggleDetails();
    } else if (!compose.hidden) {
      if (key === ' ' && filePeaks && [document.body, fileTitle].includes(event.target)) {
        event.preventDefault();
        togglePreview();
      }
    } else if (!resultCard.hidden && currentResult) {
      resultKey(event, key);
    }
  });

  function resultKey(event, key) {
    const onControl = event.target.closest('button, a, summary, select, input:not([type="range"]), [role="tab"]');
    // A line the search has hidden is not one to act on.
    const line = segmentRows[focusedRow] && !segmentRows[focusedRow].row.hidden ? segmentRows[focusedRow] : null;
    if (key === '/') {
      event.preventDefault();
      activateTab(tabs[0]);
      searchInput.focus();
    } else if (key === 'j' || key === 'k') {
      moveRow(key === 'j' ? 1 : -1);
    } else if (key === 'c' && line) {
      copyLine(focusedRow);
    } else if (key === 'Enter' && line && !onControl) {
      jumpTo(line.start);
    } else if (key === ' ' && jobAudioURL && !onControl) {
      event.preventDefault();
      togglePlay();
    }
  }

  // ---- Start --------------------------------------------------------------------
  renderFormats();
  renderHistory();
  restoreSettings();
  syncForm();
  navCompose.setAttribute('aria-current', 'page');
  void loadConfig().then(refreshStatuses);
  const resumable = storage.get(sessionStorage, ACTIVE_JOB_KEY, '');
  if (resumable) {
    const entry = readHistory().find((item) => item.id === resumable);
    openJob(resumable, entry ? entry.name : '', entry ? entry.model : '');
  }
})();
