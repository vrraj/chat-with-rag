// Embed loader script
// Usage on host site:
// <div id="support-chat"></div>
// <script src="https://your-app.com/static/embed-loader.js"
//         data-target="#support-chat"
//         data-top_k="8"
//         data-temperature="0.4"
//         data-namespace="docs-help">
// </script>

(function () {
  const script = document.currentScript;
  if (!script) return;

  const dataset = script.dataset || {};
  const targetSelector = dataset.target || '#embed-chat-root';
  const targetEl = document.querySelector(targetSelector);
  if (!targetEl) {
    console.error('Embed loader: target element not found for selector', targetSelector);
    return;
  }

  // Build query string from data-* attributes, excluding layout keys
  const sp = new URLSearchParams();
  const layoutKeys = new Set(['target', 'width', 'height']);

  Object.keys(dataset).forEach((key) => {
    if (layoutKeys.has(key)) return;
    const val = dataset[key];
    if (val != null && val !== '') {
      sp.set(key, val);
    }
  });

  // Compute chat-embed.html URL relative to this script
  let embedUrl;
  try {
    const base = new URL('../chat-embed.html', script.src);
    embedUrl = base.toString();
  } catch (e) {
    embedUrl = '/chat-embed.html';
  }

  const qs = sp.toString();
  if (qs) {
    embedUrl += (embedUrl.indexOf('?') === -1 ? '?' : '&') + qs;
  }

  const iframe = document.createElement('iframe');
  iframe.src = embedUrl;
  iframe.style.border = '0';
  iframe.style.width = dataset.width || '100%';
  iframe.style.height = dataset.height || '400px';
  iframe.setAttribute('allow', 'clipboard-read; clipboard-write');

  targetEl.appendChild(iframe);
})();
