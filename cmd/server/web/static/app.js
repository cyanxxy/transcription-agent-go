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

  const tabs = document.querySelectorAll('.tab');
  const panes = {
    formatted: $('tab-formatted'),
    srt: $('tab-srt'),
    json: $('tab-json'),
    judge: $('tab-judge'),
    quality: $('tab-quality'),
  };

  let currentResult = null;

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
    statusCard.hidden = false;
    resultCard.hidden = true;
    logEl.textContent = '';
    progressBar.style.width = '0%';
    progress.setAttribute('aria-valuenow', '0');
    progressLabel.textContent = 'Submitting...';

    const fd = new FormData(form);
    // Convert toggle checkboxes to booleans we know how to read on the server.
    ['use_judge_pipeline', 'auto_format', 'remove_fillers'].forEach((name) => {
      if (!fd.has(name)) fd.set(name, 'false');
      else fd.set(name, 'true');
    });

    let jobId;
    try {
      const startResp = await fetch('/api/jobs', { method: 'POST', body: fd });
      if (!startResp.ok) {
        const txt = await startResp.text();
        throw new Error(`server returned ${startResp.status}: ${txt}`);
      }
      const startJSON = await startResp.json();
      jobId = startJSON.job_id;
    } catch (err) {
      console.error(err);
      progressLabel.textContent = 'Error: ' + err.message;
      logEl.textContent = String(err);
      finish();
      return;
    }

    const stream = new EventSource(`/api/jobs/${jobId}/stream`);
    stream.addEventListener('progress', (e) => {
      const data = JSON.parse(e.data);
      const percent = Math.round(data.fraction * 100);
      progressBar.style.width = `${percent}%`;
      progress.setAttribute('aria-valuenow', String(percent));
      progressLabel.textContent = data.stage;
      logEl.textContent += `[${(data.fraction * 100).toFixed(1)}%] ${data.stage}\n`;
      logEl.scrollTop = logEl.scrollHeight;
    });
    stream.addEventListener('result', (e) => {
      const payload = JSON.parse(e.data);
      currentResult = payload;
      renderResult(payload);
      stream.close();
      finish();
    });
    stream.addEventListener('error-event', (e) => {
      let msg = 'Unknown error';
      try {
        msg = JSON.parse(e.data).message;
      } catch (_) {}
      progressLabel.textContent = 'Error: ' + msg;
      logEl.textContent += '\nERROR: ' + msg + '\n';
      stream.close();
      finish();
    });
    stream.onerror = () => {
      // Browser auto-reconnects; surface the issue once and stop.
      stream.close();
      finish();
    };
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
    summaryEl.innerHTML = '';
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
    if (toolUsage.length) {
      const totalTools = toolUsage.reduce((sum, tool) => sum + (tool.count || 0), 0);
      stats.push(['Judge tools', String(totalTools)]);
    }
    stats.forEach(([lbl, val]) => {
      const stat = document.createElement('div');
      stat.className = 'stat';
      stat.innerHTML = `<div class="label">${lbl}</div><div class="value">${val}</div>`;
      summaryEl.appendChild(stat);
    });
  }
})();
