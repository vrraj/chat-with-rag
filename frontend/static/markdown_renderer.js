(function () {
  let _purifier = null;
  const VERSION = '6';
  let _lastError = null;

  function isEnabled() {
    try {
      return !!(window && window.__ENABLE_MARKDOWN_RENDERING);
    } catch (e) {
      return false;
    }
  }

  function canParseMarkdown() {
    try {
      const m = (typeof window !== 'undefined') ? window.marked : null;
      if (!m) return false;
      if (typeof m === 'function') return true;
      if (typeof m.parse === 'function') return true;
      if (typeof m.marked === 'function') return true;
      if (m.marked && typeof m.marked.parse === 'function') return true;
    } catch (e) {
    }
    return false;
  }

  function getPurifier() {
    try {
      if (_purifier && typeof _purifier.sanitize === 'function') return _purifier;

      const dp = (typeof window !== 'undefined') ? window.DOMPurify : null;
      if (!dp) return null;
      if (typeof dp.sanitize === 'function') {
        _purifier = dp;
        return _purifier;
      }
      if (typeof dp === 'function') {
        try {
          const inst = dp(window);
          if (inst && typeof inst.sanitize === 'function') {
            _purifier = inst;
            return _purifier;
          }
        } catch (e) {
        }
      }
    } catch (e) {
    }
    return null;
  }

  function canRender() {
    try {
      const purifier = getPurifier();
      return isEnabled() && typeof window !== 'undefined' && canParseMarkdown() && !!purifier;
    } catch (e) {
      return false;
    }
  }

  function _wrapTables(rootEl) {
    try {
      if (!rootEl || !rootEl.querySelectorAll) return;
      const tables = rootEl.querySelectorAll('table');
      tables.forEach((t) => {
        const parent = t.parentElement;
        if (parent && parent.classList && parent.classList.contains('md-table-wrap')) return;
        const wrap = document.createElement('div');
        wrap.className = 'md-table-wrap';
        t.replaceWith(wrap);
        wrap.appendChild(t);
      });
    } catch (e) {
    }
  }

  function _hardenLinks(rootEl) {
    try {
      if (!rootEl || !rootEl.querySelectorAll) return;
      const links = rootEl.querySelectorAll('a[href]');
      links.forEach((a) => {
        a.setAttribute('target', '_blank');
        a.setAttribute('rel', 'noopener noreferrer');
      });
    } catch (e) {
    }
  }

  function renderMarkdownToSafeHtml(markdownText) {
    const src = (markdownText == null) ? '' : String(markdownText);

    _lastError = null;

    if (!canRender()) {
      _lastError = { where: 'canRender', enabled: isEnabled(), markedType: (typeof window !== 'undefined' ? typeof window.marked : 'n/a'), hasPurifier: !!getPurifier(), canParse: canParseMarkdown() };
      return { ok: false, html: '' };
    }

    try {
      const purifier = getPurifier();
      if (!purifier || typeof purifier.sanitize !== 'function') {
        _lastError = { where: 'purifier', msg: 'No sanitizer available' };
        return { ok: false, html: '' };
      }

      if (!canParseMarkdown()) {
        _lastError = { where: 'marked', msg: 'No parse function available' };
        return { ok: false, html: '' };
      }

      const opts = {
        gfm: true,
        breaks: true,
        headerIds: false,
        mangle: false
      };

      let rawHtml = '';
      try {
        const m = window.marked;
        if (typeof m === 'function') rawHtml = m(src, opts);
        else if (m && typeof m.parse === 'function') rawHtml = m.parse(src, opts);
        else if (m && typeof m.marked === 'function') rawHtml = m.marked(src, opts);
        else if (m && m.marked && typeof m.marked.parse === 'function') rawHtml = m.marked.parse(src, opts);
      } catch (e) {
        _lastError = { where: 'render', msg: String(e && e.message ? e.message : e) };
        return { ok: false, html: '' };
      }

      let cleanHtml = '';
      try {
        cleanHtml = purifier.sanitize(rawHtml);
      } catch (e) {
        _lastError = { where: 'sanitize', msg: String(e && e.message ? e.message : e) };
        try {
          cleanHtml = purifier.sanitize(rawHtml, { USE_PROFILES: { html: true } });
        } catch (e2) {
          _lastError = { where: 'sanitize', msg: String(e2 && e2.message ? e2.message : e2) };
          cleanHtml = '';
        }
      }
      if (typeof cleanHtml !== 'string') {
        cleanHtml = '';
      }
      return { ok: true, html: cleanHtml };
    } catch (e) {
      _lastError = { where: 'render', msg: String(e && e.message ? e.message : e) };
      return { ok: false, html: '' };
    }
  }

  function decorateRenderedMessage(containerEl) {
    _wrapTables(containerEl);
    _hardenLinks(containerEl);
  }

  window.MarkdownRenderer = {
    renderMarkdownToSafeHtml,
    decorateRenderedMessage
  };

  try {
    if (typeof window !== 'undefined') {
      window.__MARKDOWN_RENDERER_VERSION = VERSION;
      window.__MARKDOWN_RENDERER_LAST_ERROR = () => _lastError;
    }
  } catch (e) {
  }
})();
