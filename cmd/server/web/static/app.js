(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);

  // ---- Static data -----------------------------------------------------------
  const MODELS = {
    'gemini-3.5-transcribe': { label: 'Gemini 3.5 Transcribe', provider: 'gemini', direct: true, languages: true },
    'MAI-Transcribe-2': { label: 'Microsoft MAI-Transcribe-2', provider: 'microsoft', direct: true, languages: true },
    'muse-voice-transcribe-1.0': { label: 'Meta Muse Voice', provider: 'meta', direct: true, languages: false },
    'gemini-3.8-flash': { label: 'Gemini 3.8 Flash', provider: 'gemini', direct: false, languages: true },
    'gemini-3.5-flash-lite': { label: 'Gemini 3.5 Flash-Lite', provider: 'gemini', direct: false, languages: true },
  };
  const PROVIDERS = { gemini: 'Gemini', microsoft: 'Microsoft', meta: 'Meta' };

  const FORMATS = [
    { id: 'docx', label: 'Word', ext: '.docx', use: 'Share and edit', timestamps: true, binary: true },
    { id: 'txt', label: 'Text', ext: '.txt', use: 'Paste anywhere', timestamps: true },
    { id: 'md', label: 'Markdown', ext: '.md', use: 'Notes and wikis', timestamps: true },
    { id: 'srt', label: 'SubRip', ext: '.srt', use: 'Video subtitles' },
    { id: 'vtt', label: 'WebVTT', ext: '.vtt', use: 'Web captions' },
    { id: 'csv', label: 'CSV', ext: '.csv', use: 'Spreadsheets' },
    { id: 'json', label: 'JSON', ext: '.json', use: 'Full data' },
  ];

  const SPEAKER_COLORS = ['#3b6fd8', '#d2543a', '#2e9d78', '#8e5bd0', '#c7881a', '#2b95b5', '#c24f8a', '#6f8a1f'];
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
  const fileCard = $('file-card');
  const dropzoneEmpty = $('dropzone-empty');
  const keyCallout = $('key-callout');
  const thinkingSelect = form.elements.namedItem('transcription_thinking_level');

  const emptyState = $('empty-state');
  const statusCard = $('status-card');
  const resultCard = $('result-card');
  const progress = statusCard.querySelector('.progress');
  const progressBar = $('progress-bar');
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
  const timelineAxis = $('timeline-axis');
  const timelineHint = $('timeline-hint');
  const playhead = $('playhead');
  const player = $('player');
  const audioPlayer = $('audio-player');
  const followAudio = $('follow-audio');
  const transcriptView = $('transcript-view');
  const searchInput = $('transcript-search');
  const searchCount = $('search-count');
  const speakerChips = $('speaker-chips');
  const formatGrid = $('format-grid');
  const timestampsOption = $('timestamps-option');
  const exportTimestamps = $('export-timestamps');
  const exportDownload = $('export-download');
  const exportCopy = $('export-copy');
  const exportStatus = $('export-status');

  const keysDialog = $('keys-dialog');
  const keysStatus = $('keys-status');
  const authToken = $('auth-token');
  const historyButton = $('history-button');
  const historyPanel = $('history-panel');
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
  let checkingJobStatus = false;
  // True while an upload or transcription is in progress. A transcript that
  // only awaits review is not busy, so a new job can start meanwhile.
  let busy = false;
  let elapsedTimer = null;
  let audioURL = null; // file selected for the next upload
  let jobAudioURL = null; // file behind the transcript on screen
  let segmentRows = [];
  let activeRow = null;
  let timelineDuration = 0;
  let selectedFormat = 'docx';
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
    if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
    if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
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
    if (minutes < 1) return 'just now';
    if (minutes < 60) return `${minutes} min ago`;
    const hours = Math.round(minutes / 60);
    if (hours < 24) return `${hours} h ago`;
    return new Date(date).toLocaleDateString();
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
  const isShown = (node) => !node.hidden && !node.classList.contains('leaving');

  let toastTimer = null;
  function toast(message) {
    toastEl.textContent = message;
    showAnimated(toastEl);
    window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => hideAnimated(toastEl, 150), 2600);
  }

  function selectedModel() {
    const checked = form.querySelector('input[name="model_name"]:checked');
    return checked ? checked.value : 'gemini-3.5-transcribe';
  }

  function isTyping(target) {
    return target instanceof HTMLElement && (target.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName));
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
  $('new-transcription').addEventListener('click', () => {
    document.querySelector('.composer').scrollIntoView({ behavior: scrollBehavior(), block: 'start' });
  });
  $('empty-keys').addEventListener('click', () => openKeys('gemini'));
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

    const minimal = thinkingSelect.querySelector('[data-flash-lite-only]');
    const allowMinimal = model === 'gemini-3.5-flash-lite';
    minimal.disabled = !allowMinimal;
    minimal.hidden = !allowMinimal;
    if (!allowMinimal && thinkingSelect.value === 'minimal') thinkingSelect.value = 'low';

    // With no keys at all, one clear prompt beats a "Needs key" badge on every model.
    const noKeys = Boolean(serverConfig) && !Object.keys(PROVIDERS).some(providerConfigured);
    document.querySelectorAll('[data-key-badge]').forEach((badge) => {
      badge.hidden = !serverConfig || noKeys || providerConfigured(badge.dataset.keyBadge);
    });
    $('empty-keys').hidden = !noKeys;

    const missing = requiredProviders(model, speechReview.checked).filter((p) => !providerConfigured(p));
    keyCallout.hidden = missing.length === 0;
    if (missing.length) {
      $('key-callout-text').textContent = noKeys
        ? 'Add an API key for Gemini, Microsoft, or Meta to start transcribing.'
        : missing.includes(info.provider)
          ? `Add a ${PROVIDERS[info.provider]} API key to use ${info.label}.`
          : 'Double-checking with a judge needs a Gemini API key.';
    }
    updateSubmitState();
    renderKeys();
  }

  function updateSubmitState() {
    const running = busy;
    const missing = requiredProviders(selectedModel(), speechReview.checked).some((p) => !providerConfigured(p));
    submitBtn.disabled = running || missing;
    submitBtn.title = missing ? 'Add the missing API key first' : '';
  }

  form.addEventListener('change', (event) => {
    if (event.target.name === 'model_name' || event.target === speechReview) syncForm();
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

  function setFile(file) {
    if (audioURL && audioURL !== jobAudioURL) URL.revokeObjectURL(audioURL);
    audioURL = null;
    if (!file) {
      fileInput.value = '';
      fileCard.hidden = true;
      dropzoneEmpty.hidden = false;
      dropzone.classList.remove('has-file');
      showFormError('');
      return;
    }
    const limit = serverConfig && serverConfig.max_upload_bytes;
    $('file-name').textContent = file.name;
    $('file-detail').textContent = formatBytes(file.size);
    fileCard.hidden = false;
    dropzoneEmpty.hidden = true;
    dropzone.classList.add('has-file');
    showFormError(limit && file.size > limit ? `This file is ${formatBytes(file.size)}; the server accepts up to ${formatBytes(limit)}.` : '');
    audioURL = URL.createObjectURL(file);
    const probe = new Audio();
    probe.preload = 'metadata';
    probe.addEventListener('loadedmetadata', () => {
      if (Number.isFinite(probe.duration)) $('file-detail').textContent = `${formatClock(probe.duration)} · ${formatBytes(file.size)}`;
    }, { once: true });
    probe.src = audioURL;
  }

  fileInput.addEventListener('change', () => setFile(fileInput.files && fileInput.files[0]));
  $('file-clear').addEventListener('click', () => setFile(null));
  ['dragover', 'dragenter'].forEach((evt) => dropzone.addEventListener(evt, (event) => {
    event.preventDefault();
    dropzone.classList.add('active');
  }));
  ['dragleave', 'drop'].forEach((evt) => dropzone.addEventListener(evt, (event) => {
    event.preventDefault();
    dropzone.classList.remove('active');
  }));
  dropzone.addEventListener('drop', (event) => {
    const files = event.dataTransfer && event.dataTransfer.files;
    if (files && files.length) {
      fileInput.files = files;
      setFile(files[0]);
    }
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

  function renderHistory() {
    const items = readHistory();
    const ttl = (serverConfig && serverConfig.job_ttl_seconds) || 0;
    $('history-list').replaceChildren(...items.map((item) => {
      const li = el('li');
      const button = el('button', 'history-item');
      button.type = 'button';
      // Jobs expire a retention period after they finish; allow for run time.
      const stale = ttl > 0 && Date.now() - item.at > ttl * 1000 * 2;
      button.classList.toggle('expired', stale);
      if (item.id === currentJobId) button.setAttribute('aria-current', 'true');
      button.append(el('span', 'history-name', item.name || 'Recording'));
      const model = MODELS[item.model] ? MODELS[item.model].label : item.model || '';
      button.append(el('span', 'history-detail', [relativeTime(item.at), model, stale ? 'may have expired' : ''].filter(Boolean).join(' · ')));
      button.addEventListener('click', () => {
        closeHistory();
        openJob(item.id, item.name);
      });
      li.append(button);
      return li;
    }));
    $('history-note').textContent = items.length === 0
      ? 'Transcripts you start in this browser appear here.'
      : ttl ? `The server keeps each transcript for ${formatDuration(ttl)} after it finishes.` : '';
  }

  function closeHistory() {
    hideAnimated(historyPanel, 100);
    historyButton.setAttribute('aria-expanded', 'false');
  }

  historyButton.addEventListener('click', () => {
    const open = !isShown(historyPanel);
    renderHistory();
    if (open) showAnimated(historyPanel);
    else hideAnimated(historyPanel, 100);
    historyButton.setAttribute('aria-expanded', String(open));
  });
  document.addEventListener('click', (event) => {
    if (isShown(historyPanel) && !event.target.closest('.popover-anchor')) closeHistory();
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && isShown(historyPanel)) {
      closeHistory();
      historyButton.focus();
    }
  });

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
    stageIndex = Math.max(stageIndex, index);
    stageItems.forEach((item) => {
      const itemIndex = STAGE_ORDER.indexOf(item.dataset.stage);
      item.dataset.state = itemIndex < stageIndex ? 'done' : itemIndex === stageIndex ? 'active' : '';
      const detailEl = item.querySelector('.stage-detail');
      if (itemIndex === stageIndex && detail !== undefined) detailEl.textContent = detail;
      if (itemIndex < stageIndex) detailEl.textContent = '';
    });
  }

  function finishStages(failed) {
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
    });
    stageItems[3].hidden = !showReview;
  }

  function setProgress(fraction) {
    const percent = Math.round(Math.min(1, Math.max(0, fraction)) * 100);
    progressBar.style.transform = `translateX(${percent - 100}%)`;
    progress.setAttribute('aria-valuenow', String(percent));
  }

  function setConnectionState(text, state = 'pending') {
    connectionState.textContent = text;
    connectionState.dataset.state = state;
  }

  function setJobReference(jobId) {
    jobReference.textContent = jobId ? `Job ${jobId}` : '';
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
    logEl.textContent += `${time}  ${line}\n`;
    logEl.scrollTop = logEl.scrollHeight;
  }

  function showRunView(name, model, review) {
    $('app').classList.add('has-job');
    emptyState.hidden = true;
    statusCard.hidden = false;
    resultCard.hidden = true;
    $('run-title').textContent = 'Transcribing';
    $('run-subtitle').textContent = [name, MODELS[model] ? MODELS[model].label : ''].filter(Boolean).join(' · ');
    logEl.textContent = '';
    progressElapsed.textContent = '';
    setProgress(0);
    resetStages(review);
    setJobReference('');
  }

  function setBusy(value, label) {
    busy = value;
    spinner.hidden = !value;
    submitLabel.textContent = label || 'Transcribe';
    updateSubmitState();
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
  async function createJob(formData, idempotencyKey, onProgress) {
    for (let attempt = 0; attempt < 2; attempt += 1) {
      try {
        return await uploadJob(formData, idempotencyKey, onProgress);
      } catch (error) {
        if (attempt > 0) throw error;
        progressLabel.textContent = 'Connection dropped; checking whether the upload arrived…';
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
    const model = selectedModel();
    if (currentStream) {
      currentStream.close();
      currentStream = null;
    }
    if (jobAudioURL && jobAudioURL !== audioURL) URL.revokeObjectURL(jobAudioURL);
    jobAudioURL = audioURL;
    clearReviewState();
    currentResult = null;
    currentJobName = file.name;
    showRunView(file.name, model, reviewEnabled());
    setStage('upload', formatBytes(file.size));
    progressLabel.textContent = 'Uploading…';
    setConnectionState('Uploading', 'pending');
    startElapsed();
    setBusy(true, 'Uploading…');
    submitBtn.disabled = true;
    if (window.matchMedia('(max-width: 960px)').matches) statusCard.scrollIntoView({ behavior: scrollBehavior(), block: 'start' });

    const data = new FormData(form);
    ['use_judge_pipeline', 'agentic_mode', 'auto_format', 'remove_fillers', 'speech_review'].forEach((name) => {
      const input = form.elements.namedItem(name);
      if (input && !input.disabled) data.set(name, input.checked ? 'true' : 'false');
    });

    try {
      const response = await createJob(data, crypto.randomUUID(), (fraction) => {
        setStage('upload', `${Math.round(fraction * 100)}%`);
        setProgress(fraction * 0.08);
      });
      if (response.status < 200 || response.status >= 300) throw new Error(messageFrom(response.text, response.status));
      const started = JSON.parse(response.text);
      currentJobId = started.job_id;
      setJobReference(currentJobId);
      rememberJob({ id: currentJobId, name: file.name, model, at: Date.now() });
      if (started.idempotent_replay) appendLog('Recovered a submission the server had already accepted.');
      appendLog('Upload complete.');
      setStage('prepare');
      progressLabel.textContent = 'Waiting for a free worker…';
      setConnectionState('Queued', 'pending');
    } catch (error) {
      $('run-title').textContent = 'Upload failed';
      progressLabel.textContent = error.message;
      appendLog(`Error: ${error.message}`);
      setConnectionState('Failed', 'error');
      finishStages(true);
      showFormError(error.message);
      stopElapsed();
      setBusy(false);
      return;
    }
    connectStream(currentJobId);
  });

  // ---- Live updates ------------------------------------------------------------------
  function openJob(jobId, name) {
    if (currentStream) currentStream.close();
    currentStream = null;
    clearReviewState();
    currentResult = null;
    currentJobName = name || '';
    if (jobAudioURL && jobAudioURL !== audioURL) URL.revokeObjectURL(jobAudioURL);
    jobAudioURL = null;
    showRunView(name, '', false);
    progressLabel.textContent = 'Loading…';
    connectStream(jobId, true);
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
        showReview(review.reasons || []);
        return;
      }
      finishStages(false);
      setProgress(1);
      closeStream(stream);
      setBusy(false);
      statusCard.hidden = true;
      if (!resumed) toast('Transcript ready');
      resultCard.scrollIntoView({ behavior: scrollBehavior(), block: 'start' });
    });
    stream.addEventListener('review-required', (event) => {
      const data = JSON.parse(event.data);
      showReview(data.reasons || [], data.review_expires_at);
    });
    stream.addEventListener('review-completed', (event) => completeReview(JSON.parse(event.data).decision));
    stream.addEventListener('canceled', () => {
      $('run-title').textContent = 'Canceled';
      progressLabel.textContent = 'This transcription was canceled.';
      setConnectionState('Canceled', 'error');
      finishStages(true);
      closeStream(stream);
      setBusy(false);
    });
    stream.addEventListener('error-event', (event) => {
      let message = 'Something went wrong.';
      try { message = JSON.parse(event.data).message; } catch (_) { /* keep default */ }
      $('run-title').textContent = 'Transcription failed';
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
    if (currentStream !== stream || checkingJobStatus) return;
    checkingJobStatus = true;
    setConnectionState('Reconnecting', 'pending');
    try {
      const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`);
      if (response.status === 404) {
        forgetJob(jobId);
        markJobUnavailable(
          awaitingReview ? 'The review window has closed.' : 'This transcript is no longer on the server. Transcripts are kept for a limited time after they finish.',
          awaitingReview ? 'Expired' : 'Unavailable'
        );
        return;
      }
      if (!response.ok) throw new Error(`status check returned ${response.status}`);
      const state = await response.json();
      if (state.status === 'awaiting_review') {
        showReview([], state.review_expires_at);
        return;
      }
      if (state.status === 'expired') {
        markJobUnavailable('The review window has closed.', 'Expired');
        return;
      }
      progressLabel.textContent = 'Connection dropped; reconnecting…';
    } catch (_) {
      setConnectionState('Offline', 'error');
      progressLabel.textContent = 'Connection dropped; retrying…';
    } finally {
      checkingJobStatus = false;
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
      markJobUnavailable('The review window has closed.', 'Expired');
      return;
    }
    const due = reviewExpiresAt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    reviewDeadline.textContent = `Decide by ${due} · ${formatDuration(remaining / 1000)} left`;
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

  function showReview(reasons, expiresAt) {
    awaitingReview = true;
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

  function markJobUnavailable(message, badge = 'Unavailable') {
    const wasAwaitingReview = awaitingReview;
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
      $('run-title').textContent = 'Not available';
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
  function completeReview(decision, announce = false) {
    awaitingReview = false;
    clearReviewTimer();
    reviewPanel.hidden = true;
    if (currentStream) closeStream(currentStream);
    setBusy(false);
    if (announce) toast(decision === 'accept' ? 'Transcript accepted' : 'Transcript rejected');
  }

  async function submitReview(decision) {
    acceptReview.disabled = true;
    rejectReview.disabled = true;
    try {
      const response = await fetch(`/api/jobs/${encodeURIComponent(currentJobId)}/review`, {
        method: 'POST',
        headers: authHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ decision, note: reviewNote.value.trim() }),
      });
      if (!response.ok) {
        const error = new Error(await errorMessage(response));
        error.status = response.status;
        throw error;
      }
      completeReview(decision, true);
    } catch (error) {
      if ([404, 409, 410].includes(error.status)) {
        markJobUnavailable('Review is no longer available for this transcript.', 'Expired');
        return;
      }
      toast(`Could not save the review: ${error.message}`);
    } finally {
      if (awaitingReview) {
        acceptReview.disabled = false;
        rejectReview.disabled = false;
      }
    }
  }

  cancelJob.addEventListener('click', async () => {
    if (!currentJobId) return;
    cancelJob.disabled = true;
    try {
      const response = await fetch(`/api/jobs/${encodeURIComponent(currentJobId)}/cancel`, {
        method: 'POST',
        headers: authHeaders({ 'Content-Type': 'application/json' }),
        body: '{}',
      });
      if (!response.ok) throw new Error(await errorMessage(response));
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

  function renderSummary(result) {
    const segments = result.segments || [];
    const metadata = result.metadata || {};
    const speakerCount = speakersOf(segments).length;
    const parts = [
      formatDuration(metadata.duration),
      `${speakerCount} speaker${speakerCount === 1 ? '' : 's'}`,
      (MODELS[result.model_used] && MODELS[result.model_used].label) || result.model_used,
      result.judge_used ? 'reviewed by a judge' : '',
      `done in ${formatDuration(result.processing_time)}`,
    ].filter(Boolean);
    const summary = $('result-summary');
    summary.replaceChildren(...parts.map((part) => el('span', '', part)));
    const gaps = segments.filter((segment) => segment.speaker === GAP_SPEAKER).length;
    if (gaps) summary.append(el('span', 'warn', `${gaps} gap${gaps === 1 ? '' : 's'} to check`));
    if (result.judge_used && result.quality && result.quality.overall_score) {
      summary.append(el('span', '', `quality ${Math.round(result.quality.overall_score)}/100`));
    }
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
      const label = el('span', 'lane-label', displayName(speaker));
      label.dataset.speaker = speaker;
      lane.append(label);
      lanes.set(speaker, lane);
    });
    let gapLane = null;
    segments.forEach((segment, index) => {
      const start = segmentStart(segment);
      const end = segmentEnd(segments, index);
      const isGap = segment.speaker === GAP_SPEAKER;
      if (isGap && !gapLane) {
        gapLane = el('div', 'lane gaps');
        gapLane.style.setProperty('--speaker', 'var(--signal)');
        gapLane.append(el('span', 'lane-label', 'Gaps'));
      }
      const bar = el('span', isGap ? 'bar gap' : 'bar');
      bar.style.left = `${(start / timelineDuration) * 100}%`;
      bar.style.width = `${(Math.max(end - start, 0.2) / timelineDuration) * 100}%`;
      bar.title = `${formatClock(start)} · ${isGap ? 'No transcript' : displayName(segment.speaker)}: ${segment.text.slice(0, 120)}`;
      bar.dataset.index = String(index);
      (isGap ? gapLane : lanes.get(segment.speaker)).append(bar);
    });
    timelineLanes.replaceChildren(...lanes.values(), ...(gapLane ? [gapLane] : []));

    const ticks = 5;
    timelineAxis.replaceChildren(...Array.from({ length: ticks + 1 }, (_, i) => {
      const tick = el('span', '', formatClock((timelineDuration * i) / ticks));
      tick.style.left = `${(i / ticks) * 100}%`;
      return tick;
    }));
    playhead.hidden = true;
    timelineHint.textContent = jobAudioURL
      ? 'Click a bar or line to play from there. Space plays and pauses; / searches.'
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
    if (searchInput.value) {
      searchInput.value = '';
      applySearch();
    }
    activateTab(tabs[0]);
    target.row.scrollIntoView({ block: 'center', behavior: scrollBehavior() });
    target.row.classList.remove('flash');
    void target.row.offsetWidth;
    target.row.classList.add('flash');
  }

  function renderChips(segments) {
    speakerChips.replaceChildren(...speakersOf(segments).map((speaker) => {
      const chip = el('button', 'chip');
      chip.type = 'button';
      chip.style.setProperty('--speaker', colorFor(speaker));
      chip.title = `Rename ${speaker}`;
      chip.append(el('span', 'chip-name', displayName(speaker)), el('span', 'chip-edit', 'Rename'));
      chip.addEventListener('click', () => editChip(chip, speaker));
      return chip;
    }));
  }

  function editChip(chip, speaker) {
    if (chip.querySelector('input')) return;
    const input = el('input');
    input.type = 'text';
    input.value = speakerNames.get(speaker) || '';
    input.placeholder = speaker;
    input.maxLength = 120;
    input.setAttribute('aria-label', `New name for ${speaker}`);
    chip.replaceChildren(input);
    input.focus();
    let done = false;
    const commit = (save) => {
      if (done) return;
      done = true;
      if (save) {
        speakerNames.set(speaker, input.value.trim());
        refreshSpeakerNames();
      }
      renderChips(currentResult.result.segments || []);
    };
    input.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') { event.preventDefault(); commit(true); }
      if (event.key === 'Escape') { event.preventDefault(); commit(false); }
    });
    input.addEventListener('blur', () => commit(true));
    input.addEventListener('click', (event) => event.stopPropagation());
  }

  function refreshSpeakerNames() {
    transcriptView.querySelectorAll('.segment-speaker[data-speaker]').forEach((node) => { node.textContent = displayName(node.dataset.speaker); });
    timelineLanes.querySelectorAll('.lane-label[data-speaker]').forEach((node) => { node.textContent = displayName(node.dataset.speaker); });
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
      row.style.setProperty('--speaker', gap ? 'var(--signal)' : colorFor(segment.speaker));
      const start = segmentStart(segment);
      const time = el('button', 'segment-time', formatClock(start));
      time.type = 'button';
      time.disabled = !jobAudioURL;
      time.title = jobAudioURL ? 'Play from here' : '';
      time.addEventListener('click', () => jumpTo(start));
      const speaker = el('span', 'segment-speaker', gap ? 'No transcript for this part' : displayName(segment.speaker));
      if (!gap) speaker.dataset.speaker = segment.speaker;
      const text = el('p', 'segment-text', segment.text);
      row.append(time, speaker, text);
      fragment.append(row);
      return { row, text, raw: segment.text, start, end: segmentEnd(segments, index) };
    });
    transcriptView.replaceChildren(fragment);
    activeRow = null;
    applySearch();
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

  // timeupdate fires only ~4 times a second, so the playhead follows the audio
  // every frame while playing. Moving it with transform avoids layout work.
  function placePlayhead() {
    if (timelineDuration <= 0) return;
    const fraction = Math.min(1, audioPlayer.currentTime / timelineDuration);
    playhead.style.transform = `translateX(${timelineLanes.offsetLeft + timelineLanes.offsetWidth * fraction}px)`;
    playhead.hidden = false;
  }
  let playheadFrame = 0;
  function followPlayhead() {
    placePlayhead();
    playheadFrame = audioPlayer.paused ? 0 : window.requestAnimationFrame(followPlayhead);
  }
  audioPlayer.addEventListener('play', () => {
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
    $('app').classList.add('has-job');
    emptyState.hidden = true;
    resultCard.hidden = false;
    $('result-title').textContent = (currentJobName || (result.metadata && result.metadata.filename) || 'Transcript').replace(/\.[^.]+$/, '');
    const reviewed = Boolean(result.judge_used);
    tabs.forEach((tab) => { if (tab.hasAttribute('data-pipeline-only')) tab.hidden = !reviewed; });
    speakerNames.clear();
    speakerColors.clear();
    searchInput.value = '';

    renderSummary(result);
    renderTimeline(result);
    renderChips(result.segments || []);
    renderTranscript(result.segments || []);

    player.hidden = !jobAudioURL;
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
    exportStatus.textContent = '';
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
    formatGrid.replaceChildren(...FORMATS.map((format) => {
      const button = el('button', 'format-option');
      button.type = 'button';
      button.setAttribute('role', 'radio');
      button.dataset.format = format.id;
      const label = el('span', 'format-label');
      label.append(el('span', 'format-name', format.label), el('span', 'format-use', format.use));
      button.append(label, el('span', 'format-ext', format.ext));
      button.addEventListener('click', () => selectFormat(format.id));
      return button;
    }));
    selectFormat(storage.get(localStorage, 'export-format', 'docx'));
  }

  function selectFormat(id) {
    const format = FORMATS.find((item) => item.id === id) || FORMATS[0];
    selectedFormat = format.id;
    storage.set(localStorage, 'export-format', format.id);
    formatGrid.querySelectorAll('.format-option').forEach((button) => {
      const checked = button.dataset.format === format.id;
      button.classList.toggle('selected', checked);
      button.setAttribute('aria-checked', String(checked));
      button.tabIndex = checked ? 0 : -1;
    });
    timestampsOption.hidden = !format.timestamps;
    exportDownload.textContent = `Download ${format.ext}`;
    exportCopy.hidden = Boolean(format.binary);
  }

  formatGrid.addEventListener('keydown', (event) => {
    if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) return;
    event.preventDefault();
    const index = FORMATS.findIndex((item) => item.id === selectedFormat);
    const step = ['ArrowRight', 'ArrowDown'].includes(event.key) ? 1 : -1;
    const next = FORMATS[(index + step + FORMATS.length) % FORMATS.length];
    selectFormat(next.id);
    formatGrid.querySelector(`[data-format="${next.id}"]`).focus();
  });

  function downloadName(response) {
    const header = response.headers.get('Content-Disposition') || '';
    const match = /filename\*=UTF-8''([^;]+)|filename="([^"]+)"/i.exec(header);
    if (match) return match[1] ? decodeURIComponent(match[1]) : match[2];
    return `transcript.${selectedFormat}`;
  }

  exportDownload.addEventListener('click', async () => {
    exportDownload.disabled = true;
    exportStatus.textContent = 'Preparing…';
    try {
      const response = await fetchExport(selectedFormat, exportTimestamps.checked);
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = el('a');
      link.href = url;
      link.download = downloadName(response);
      document.body.append(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
      exportStatus.textContent = '';
      toast(`Downloaded ${link.download}`);
    } catch (error) {
      exportStatus.textContent = `Download failed: ${error.message}`;
    } finally {
      exportDownload.disabled = false;
    }
  });

  exportCopy.addEventListener('click', async () => {
    exportCopy.disabled = true;
    try {
      const response = await fetchExport(selectedFormat, exportTimestamps.checked);
      await navigator.clipboard.writeText(await response.text());
      exportStatus.textContent = '';
      toast('Copied to clipboard');
    } catch (error) {
      exportStatus.textContent = `Copy failed: ${error.message}`;
    } finally {
      exportCopy.disabled = false;
    }
  });

  // ---- Keyboard shortcuts ----------------------------------------------------------
  document.addEventListener('keydown', (event) => {
    if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.altKey || isTyping(event.target)) return;
    if (resultCard.hidden || keysDialog.open) return;
    if (event.key === '/') {
      event.preventDefault();
      activateTab(tabs[0]);
      searchInput.focus();
    } else if (event.key === ' ' && jobAudioURL && event.target === document.body) {
      event.preventDefault();
      if (audioPlayer.paused) void audioPlayer.play().catch(() => {});
      else audioPlayer.pause();
    }
  });

  // ---- Start --------------------------------------------------------------------
  renderFormats();
  renderHistory();
  restoreSettings();
  syncForm();
  void loadConfig();
  const resumable = storage.get(sessionStorage, ACTIVE_JOB_KEY, '');
  if (resumable) {
    const entry = readHistory().find((item) => item.id === resumable);
    openJob(resumable, entry ? entry.name : '');
  }
})();
