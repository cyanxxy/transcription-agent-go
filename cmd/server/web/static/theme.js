// Loaded before the stylesheet so a saved theme never flashes.
try {
  const theme = localStorage.getItem('transcription-theme');
  if (theme === 'light' || theme === 'dark') document.documentElement.dataset.theme = theme;
} catch (_) { /* storage unavailable: follow the system */ }
