(() => {
  const $ = (id) => document.getElementById(id);

  const form = $('upload-form');
  const submitBtn = $('submit-btn');
  const spinner = submitBtn.querySelector('.spinner');
  const label = submitBtn.querySelector('.label');

  const dropzone = $('dropzone');
  const fileInput = $('audio-input');
  const fileName = $('file-name');
  const statusCard = $('status-card');
  const resultCard = $('result-card');
  const progress = document.querySelector('.progress');
  const progressBar = $('progress-bar');
  const progressLabel = $('progress-label');
  const logEl = $('log');
  const summaryEl = $('result-summary');
  const jobReference = $('job-reference');
  const connectionState = $('connection-state');

  const tabs = document.querySelectorAll('.tab');
  const panes = {
    formatted: $('tab-formatted'),
    srt: $('tab-srt'),
    json: $('tab-json'),
    judge: $('tab-judge'),
    quality: $('tab-quality'),
    evidence: $('tab-evidence'),
  };

  let currentResult = null;
  let currentJobId = null;
  let currentStream = null;
  const authToken = $('auth-token');
  const cancelJob = $('cancel-job');
  const reviewPanel = $('review-panel');
  const reviewActions = $('review-actions');
  const reviewReasons = $('review-reasons');
  const reviewDeadline = $('review-deadline');
  const reviewNote = $('review-note');
  const acceptReview = $('accept-review');
  const rejectReview = $('reject-review');
  let awaitingReview = false;
  let reviewExpiresAt = null;
  let reviewTimer = null;
  let checkingJobStatus = false;

  function authHeaders(extra = {}) {
    const token = authToken.value.trim();
    return token ? { ...extra, Authorization: `Bearer ${token}` } : extra;
  }

  async function postJobAction(action, body) {
    if (!currentJobId) return;
    const response = await fetch(`/api/jobs/${currentJobId}/${action}`, {
      method: 'POST',
      headers: authHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify(body || {}),
    });
    if (!response.ok) {
      const error = new Error(await response.text());
      error.status = response.status;
      throw error;
    }
    return response.json();
  }

  function setConnectionState(text, state = 'pending') {
    connectionState.textContent = text;
    connectionState.dataset.state = state;
  }

  function setJobReference(jobId) {
    jobReference.textContent = jobId ? `Job ID: ${jobId}` : '';
    jobReference.hidden = !jobId;
  }

  function sleep(milliseconds) {
    return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
  }

  async function createJob(formData, idempotencyKey) {
    for (let attempt = 0; attempt < 2; attempt += 1) {
      try {
        return await fetch('/api/jobs', {
          method: 'POST',
          headers: authHeaders({ 'Idempotency-Key': idempotencyKey }),
          body: formData,
        });
      } catch (error) {
        if (attempt > 0) throw error;
        progressLabel.textContent = 'Submission interrupted; confirming acceptance...';
        setConnectionState('Confirming', 'pending');
        await sleep(450);
      }
    }
    throw new Error('job submission failed');
  }

  document.querySelectorAll('.glass-panel, .dropzone, button.primary, button.ghost').forEach((el) => {
    el.addEventListener('pointermove', (event) => {
      const rect = el.getBoundingClientRect();
      el.style.setProperty('--pointer-x', `${event.clientX - rect.left}px`);
      el.style.setProperty('--pointer-y', `${event.clientY - rect.top}px`);
    });
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files && fileInput.files[0]) {
      fileName.textContent = fileInput.files[0].name;
    }
  });

  ['dragover', 'dragenter'].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add('active');
    })
  );
  ['dragleave', 'drop'].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove('active');
    })
  );
  dropzone.addEventListener('drop', (e) => {
    if (e.dataTransfer && e.dataTransfer.files.length > 0) {
      fileInput.files = e.dataTransfer.files;
      fileInput.dispatchEvent(new Event('change'));
    }
  });

  function activateTab(tab, shouldFocus = false) {
    tabs.forEach((t) => {
      const active = t === tab;
      t.classList.toggle('active', active);
      t.setAttribute('aria-selected', active ? 'true' : 'false');
      Object.entries(panes).forEach(([key, pane]) => {
        if (key === t.dataset.tab) {
          pane.hidden = !active;
        }
      });
    });
    if (shouldFocus) {
      tab.focus();
    }
  }

  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => {
      activateTab(tab);
    });
    tab.addEventListener('keydown', (event) => {
      const keys = ['ArrowLeft', 'ArrowRight', 'Home', 'End'];
      if (!keys.includes(event.key)) return;
      event.preventDefault();
      let nextIndex = index;
      if (event.key === 'ArrowRight') nextIndex = (index + 1) % tabs.length;
      if (event.key === 'ArrowLeft') nextIndex = (index - 1 + tabs.length) % tabs.length;
      if (event.key === 'Home') nextIndex = 0;
      if (event.key === 'End') nextIndex = tabs.length - 1;
      activateTab(tabs[nextIndex], true);
    });
  });

  $('download-txt').addEventListener('click', () => download('transcript.txt', 'text/plain', panes.formatted.textContent));
  $('download-srt').addEventListener('click', () => download('transcript.srt', 'application/x-subrip', panes.srt.textContent));
  $('download-json').addEventListener('click', () => download('transcript.json', 'application/json', panes.json.textContent));

  function download(filename, type, contents) {
    const blob = new Blob([contents || ''], { type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!fileInput.files || fileInput.files.length === 0) {
      alert('Please choose an audio file first.');
      return;
    }
    submitBtn.disabled = true;
    spinner.hidden = false;
    label.textContent = 'Working...';
    clearReviewState();
    currentResult = null;
    statusCard.hidden = false;
    resultCard.hidden = true;
    logEl.textContent = '';
    progressBar.style.width = '0%';
    progress.setAttribute('aria-valuenow', '0');
    progressLabel.textContent = 'Submitting...';
    setConnectionState('Submitting', 'pending');
    setJobReference('');

    const fd = new FormData(form);
    // Convert toggle checkboxes to booleans we know how to read on the server.
    ['use_judge_pipeline', 'agentic_mode', 'auto_format', 'remove_fillers'].forEach((name) => {
      if (!fd.has(name)) fd.set(name, 'false');
      else fd.set(name, 'true');
    });

    let jobId;
    try {
      const idempotencyKey = crypto.randomUUID();
      const startResp = await createJob(fd, idempotencyKey);
      if (!startResp.ok) {
        const txt = await startResp.text();
        throw new Error(`server returned ${startResp.status}: ${txt}`);
      }
      const startJSON = await startResp.json();
      jobId = startJSON.job_id;
      currentJobId = jobId;
      setJobReference(jobId);
      setConnectionState('Queued', 'active');
      if (startJSON.idempotent_replay) {
        logEl.textContent += 'Recovered a previously accepted submission.\n';
      }
      sessionStorage.setItem('transcription-job-id', jobId);
    } catch (err) {
      console.error(err);
      progressLabel.textContent = 'Error: ' + err.message;
      logEl.textContent = String(err);
      setConnectionState('Failed', 'error');
      finish();
      return;
    }

    connectStream(jobId);
  });

  function connectStream(jobId) {
    if (currentStream) currentStream.close();
    currentJobId = jobId;
    setJobReference(jobId);
    submitBtn.disabled = true;
    spinner.hidden = false;
    label.textContent = 'Working...';
    cancelJob.hidden = false;
    setConnectionState('Connecting', 'pending');
    const stream = new EventSource(`/api/jobs/${jobId}/stream`);
    currentStream = stream;
    stream.onopen = () => {
      setConnectionState(awaitingReview ? 'Review' : 'Live', 'active');
    };
    stream.addEventListener('progress', (e) => {
      const data = JSON.parse(e.data);
      const percent = Math.round(data.fraction * 100);
      progressBar.style.width = `${percent}%`;
      progress.setAttribute('aria-valuenow', String(percent));
      progressLabel.textContent = data.stage;
      setConnectionState('Running', 'active');
      logEl.textContent += `[${(data.fraction * 100).toFixed(1)}%] ${data.stage}\n`;
      logEl.scrollTop = logEl.scrollHeight;
    });
    stream.addEventListener('result', (e) => {
      const payload = JSON.parse(e.data);
      currentResult = payload;
      renderResult(payload);
      const review = payload.result.agent_run && payload.result.agent_run.human_review;
      if (review && review.status === 'required') {
        showReview(review.reasons || []);
      } else {
        progressBar.style.width = '100%';
        progress.setAttribute('aria-valuenow', '100');
        progressLabel.textContent = 'Transcription complete';
        setConnectionState('Complete', 'active');
        stream.close();
        if (currentStream === stream) currentStream = null;
        sessionStorage.removeItem('transcription-job-id');
        cancelJob.hidden = true;
        finish();
      }
    });
    stream.addEventListener('review-required', (e) => {
      const data = JSON.parse(e.data);
      showReview(data.reasons || [], data.review_expires_at);
    });
    stream.addEventListener('review-completed', (e) => {
      const data = JSON.parse(e.data);
      completeReview(data.decision);
    });
    stream.addEventListener('canceled', () => {
      progressLabel.textContent = 'Canceled';
      setConnectionState('Canceled', 'error');
      cancelJob.hidden = true;
      sessionStorage.removeItem('transcription-job-id');
      stream.close();
      if (currentStream === stream) currentStream = null;
      finish();
    });
    stream.addEventListener('error-event', (e) => {
      let msg = 'Unknown error';
      try {
        msg = JSON.parse(e.data).message;
      } catch (_) {}
      progressLabel.textContent = 'Error: ' + msg;
      logEl.textContent += '\nERROR: ' + msg + '\n';
      setConnectionState('Failed', 'error');
      stream.close();
      if (currentStream === stream) currentStream = null;
      cancelJob.hidden = true;
      sessionStorage.removeItem('transcription-job-id');
      finish();
    });
    stream.onerror = () => {
      void handleStreamError(stream, jobId);
    };
  }

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

  function formatRemaining(milliseconds) {
    const totalSeconds = Math.max(0, Math.ceil(milliseconds / 1000));
    if (totalSeconds >= 3600) {
      const hours = Math.floor(totalSeconds / 3600);
      const minutes = Math.ceil((totalSeconds % 3600) / 60);
      return `${hours}h ${minutes}m remaining`;
    }
    if (totalSeconds >= 60) return `${Math.ceil(totalSeconds / 60)}m remaining`;
    return `${totalSeconds}s remaining`;
  }

  function updateReviewDeadline() {
    if (!reviewExpiresAt || !awaitingReview) return;
    const remaining = reviewExpiresAt.getTime() - Date.now();
    if (remaining <= 0) {
      markJobUnavailable('Human review window expired.', 'Expired');
      return;
    }
    const due = reviewExpiresAt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    reviewDeadline.textContent = `Due ${due} (${formatRemaining(remaining)})`;
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
    const reasonText = reasons.length
      ? reasons.join(' ')
      : reviewReasons.textContent || 'Check the disputed evidence before accepting or rejecting this transcript.';
    if (reviewReasons.textContent !== reasonText && reasons.length) {
      logEl.textContent += `\nREVIEW: ${reasonText}\n`;
    }
    reviewReasons.textContent = reasonText;
    if (expiresAt) setReviewDeadline(expiresAt);
    cancelJob.hidden = true;
    progressLabel.textContent = 'Human review required';
    setConnectionState('Review', 'pending');
    submitBtn.disabled = true;
    spinner.hidden = true;
    label.textContent = 'Review pending';
  }

  function markJobUnavailable(message, badge = 'Unavailable') {
    const wasAwaitingReview = awaitingReview;
    awaitingReview = false;
    clearReviewTimer();
    if (wasAwaitingReview) {
      reviewPanel.hidden = false;
      reviewActions.hidden = true;
      reviewNote.disabled = true;
      reviewDeadline.hidden = false;
      reviewDeadline.textContent = message;
    }
    progressLabel.textContent = message;
    logEl.textContent += `\n${message}\n`;
    setConnectionState(badge, 'error');
    cancelJob.hidden = true;
    sessionStorage.removeItem('transcription-job-id');
    if (currentStream) currentStream.close();
    currentStream = null;
    finish();
  }

  function completeReview(decision) {
    awaitingReview = false;
    clearReviewTimer();
    reviewPanel.hidden = true;
    reviewActions.hidden = true;
    cancelJob.hidden = true;
    const accepted = decision === 'accept';
    progressLabel.textContent = accepted ? 'Review accepted' : 'Review rejected';
    setConnectionState(accepted ? 'Complete' : 'Rejected', accepted ? 'active' : 'error');
    sessionStorage.removeItem('transcription-job-id');
    if (currentStream) currentStream.close();
    currentStream = null;
    finish();
  }

  async function handleStreamError(stream, jobId) {
    if (currentStream !== stream || checkingJobStatus) return;
    checkingJobStatus = true;
    setConnectionState('Reconnecting', 'pending');
    progressLabel.textContent = 'Connection interrupted; checking job state...';
    try {
      const response = await fetch(`/api/jobs/${jobId}`);
      if (response.status === 404) {
        markJobUnavailable(
          awaitingReview ? 'Human review window expired.' : 'Job is no longer available.',
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
        markJobUnavailable('Human review window expired.', 'Expired');
        return;
      }
      progressLabel.textContent = 'Connection interrupted; reconnecting...';
    } catch (error) {
      console.error(error);
      setConnectionState('Offline', 'error');
      progressLabel.textContent = 'Connection interrupted; retrying...';
    } finally {
      checkingJobStatus = false;
    }
  }

  async function submitReview(decision) {
    acceptReview.disabled = true;
    rejectReview.disabled = true;
    progressLabel.textContent = 'Saving review decision...';
    setConnectionState('Saving', 'pending');
    try {
      await postJobAction('review', { decision, note: reviewNote.value.trim() });
      completeReview(decision);
    } catch (error) {
      if ([404, 409, 410].includes(error.status)) {
        markJobUnavailable('Human review is no longer available for this job.', 'Expired');
        return;
      }
      progressLabel.textContent = 'Review failed: ' + error.message;
      setConnectionState('Review', 'error');
    } finally {
      if (awaitingReview) {
        acceptReview.disabled = false;
        rejectReview.disabled = false;
      }
    }
  }

  cancelJob.addEventListener('click', async () => {
    cancelJob.disabled = true;
    try {
      await postJobAction('cancel');
      progressLabel.textContent = 'Cancel requested...';
      setConnectionState('Canceling', 'pending');
    } catch (err) {
      progressLabel.textContent = 'Cancel failed: ' + err.message;
    } finally {
      cancelJob.disabled = false;
    }
  });

  acceptReview.addEventListener('click', async () => {
    await submitReview('accept');
  });

  rejectReview.addEventListener('click', async () => {
    await submitReview('reject');
  });

  function finish() {
    submitBtn.disabled = false;
    spinner.hidden = true;
    label.textContent = 'Transcribe';
  }

  function renderResult(payload) {
    resultCard.hidden = false;
    panes.formatted.textContent = payload.formatted_text || '';
    panes.srt.textContent = payload.srt || '';
    panes.json.textContent = JSON.stringify(payload.result, null, 2);
    const toolUsage = payload.result.judge_tool_usage || [];
    const toolLines = toolUsage.map((tool) => `${tool.name}: ${tool.count}`);
    panes.judge.textContent = [
      ...((payload.result.judge_notes || []).length ? payload.result.judge_notes : ['No judge notes.']),
      ...(toolLines.length ? ['', 'Judge tools:', ...toolLines] : []),
    ].join('\n');
    panes.quality.textContent = JSON.stringify(payload.result.quality, null, 2);
    panes.evidence.textContent = JSON.stringify(payload.result.agent_run || {}, null, 2);
    summaryEl.replaceChildren();
    const metadata = payload.result.metadata || {};
    const stats = [
      ['Quality', `${(payload.result.quality.overall_score || 0).toFixed(1)} / 100`],
      ['Segments', String((payload.result.segments || []).length)],
      ['Duration', `${(metadata.duration || 0).toFixed(1)}s`],
      ['Processing', `${(payload.result.processing_time || 0).toFixed(2)}s`],
      ['Strategy', payload.result.candidate_strategy || 'single_gemini'],
    ];
    if (metadata.needs_chunking) {
      stats.push(['Chunks', String(metadata.chunk_count || (metadata.chunks || []).length || 0)]);
      stats.push(['Planner', metadata.chunk_strategy || 'fixed']);
    }
    if (payload.result.judge_used) {
      stats.push(['Judge', payload.result.judge_model_used || '—']);
    }
    if (payload.result.agent_run) {
      stats.push(['Run', payload.result.agent_run.status || 'unknown']);
      stats.push(['Agent steps', String((payload.result.agent_run.steps || []).length)]);
      stats.push(['Disputes', String((payload.result.agent_run.disputed_spans || []).length)]);
    }
    if (toolUsage.length) {
      const totalTools = toolUsage.reduce((sum, tool) => sum + (tool.count || 0), 0);
      stats.push(['Judge tools', String(totalTools)]);
    }
    stats.forEach(([lbl, val]) => {
      const stat = document.createElement('div');
      stat.className = 'stat';
      const statLabel = document.createElement('div');
      statLabel.className = 'label';
      statLabel.textContent = lbl;
      const statValue = document.createElement('div');
      statValue.className = 'value';
      statValue.textContent = val;
      stat.append(statLabel, statValue);
      summaryEl.appendChild(stat);
    });
  }

  const resumableJob = sessionStorage.getItem('transcription-job-id');
  if (resumableJob) {
    statusCard.hidden = false;
    progressLabel.textContent = 'Resuming job...';
    connectStream(resumableJob);
  }
})();
