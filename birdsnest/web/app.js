/* ═══════════════════════════════════════════════════════════════
   Bird's Nest — Chat UI Controller
   WebSocket streaming, model management, settings
   ═══════════════════════════════════════════════════════════════ */

// ── State ────────────────────────────────────────────────────
let ws = null;
let isGenerating = false;
let currentStreamEl = null;
let currentNickname = 'Bird\'s Nest';
let archCategories = {};
let chatHistory = []; // {role, text, nickname, stats?}

// ── Init ─────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    restoreHistory();
    connectWebSocket();
    loadModels();
    setupSettings();
    updateRAGStatus();
    loadTools();
    document.getElementById('chatInput').focus();
});

// ── WebSocket ────────────────────────────────────────────────
function connectWebSocket() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${proto}//${location.host}/ws/chat`);

    ws.onopen = () => {
        setStatus('Connected', 'green');
        document.getElementById('sendBtn').disabled = false;
    };

    ws.onclose = () => {
        setStatus('Disconnected', 'red');
        document.getElementById('sendBtn').disabled = true;
        setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = () => setStatus('Connection error', 'red');

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);

        switch (data.type) {
            case 'start':
                isGenerating = true;
                currentNickname = data.nickname || data.model || 'Bird\'s Nest';
                currentStreamEl = addMessage('assistant', '', true);
                setStatus(`Generating...`, 'yellow');
                break;

            case 'token':
                if (currentStreamEl) {
                    const textEl = currentStreamEl.querySelector('.message-text');
                    const cursor = textEl.querySelector('.typing-indicator');
                    if (cursor) cursor.remove();
                    textEl.textContent += data.content;
                    textEl.innerHTML += '<span class="typing-indicator"></span>';
                    scrollToBottom();
                }
                break;

            case 'done':
                isGenerating = false;
                if (currentStreamEl) {
                    const textEl = currentStreamEl.querySelector('.message-text');
                    const cursor = textEl.querySelector('.typing-indicator');
                    if (cursor) cursor.remove();

                    // Detect if this was a tool-only response (has tool elements, no streamed text)
                    const hasToolResult = textEl.querySelector('.tool-result') !== null;
                    const finalText = textEl.textContent.trim();

                    if (hasToolResult) {
                        // Tool-only response — save with flag so restoreHistory skips it
                        chatHistory.push({ role: 'assistant', text: '', nickname: currentNickname, isTool: true });
                    } else {
                        // Regular model response — save the text
                        chatHistory.push({ role: 'assistant', text: finalText, nickname: currentNickname, stats: data.stats });
                    }
                    saveHistory();

                    if (data.stats) {
                        const statsEl = document.createElement('div');
                        statsEl.className = 'message-stats';
                        statsEl.textContent = `${data.stats.tokens} tokens • ${data.stats.tok_s} tok/s • ${data.stats.time}s`;
                        currentStreamEl.querySelector('.message-content').appendChild(statsEl);
                        document.getElementById('tokStats').textContent = `${data.stats.tok_s} tok/s`;
                    }
                }
                currentStreamEl = null;
                setStatus('Ready', 'green');
                document.getElementById('chatInput').focus();
                break;

            case 'tool_call':
                if (currentStreamEl) {
                    const textEl = currentStreamEl.querySelector('.message-text');
                    const cursor = textEl.querySelector('.typing-indicator');
                    if (cursor) cursor.remove();

                    const toolCallEl = document.createElement('div');
                    toolCallEl.className = 'tool-call';
                    const argsStr = Object.entries(data.args || {}).map(([k, v]) => `${k}="${v}"`).join(', ');
                    toolCallEl.innerHTML = `<span class="tool-call-icon">🔧</span> <span class="tool-call-name">${data.name}</span>${argsStr ? `<span class="tool-call-args">(${argsStr})</span>` : ''}`;
                    textEl.appendChild(toolCallEl);

                    // Live timer for slow tools
                    const slowToolMessages = {
                        'generate_image': '🎨 Generating image',
                        'screenshot': '📸 Taking screenshot',
                        'translate': '🌐 Translating',
                        'youtube_transcript': '📺 Fetching transcript',
                        'search_web': '🔍 Searching',
                        'fetch_url': '🌍 Fetching URL',
                        'weather': '🌤️ Getting weather',
                    };
                    const slowMsg = slowToolMessages[data.name];
                    if (slowMsg) {
                        const timerEl = document.createElement('div');
                        timerEl.className = 'tool-progress';
                        timerEl.id = 'toolProgressTimer';
                        timerEl.innerHTML = `<span class="tool-progress-spinner"></span> ${slowMsg}... <span class="tool-progress-time">0s</span>`;
                        textEl.appendChild(timerEl);
                        let secs = 0;
                        window._toolTimer = setInterval(() => {
                            secs++;
                            const timeEl = document.getElementById('toolProgressTimer')?.querySelector('.tool-progress-time');
                            if (timeEl) timeEl.textContent = `${secs}s`;
                        }, 1000);
                    }

                    textEl.insertAdjacentHTML('beforeend', '<span class="typing-indicator"></span>');
                    scrollToBottom();
                    setStatus(slowMsg ? `${slowMsg}...` : 'Executing tool...', 'yellow');
                }
                break;

            case 'tool_result':
                // Clear any running timer
                if (window._toolTimer) { clearInterval(window._toolTimer); window._toolTimer = null; }
                const progressEl = document.getElementById('toolProgressTimer');
                if (progressEl) progressEl.remove();

                if (currentStreamEl) {
                    const textEl = currentStreamEl.querySelector('.message-text');
                    const cursor = textEl.querySelector('.typing-indicator');
                    if (cursor) cursor.remove();

                    const resultEl = document.createElement('div');
                    resultEl.className = 'tool-result';

                    // Check if result contains an image URL
                    const imgMatch = data.result && data.result.match(/URL:\s*(\/workspace\/\S+)/);
                    let imageHtml = '';
                    if (imgMatch) {
                        imageHtml = `<img src="${imgMatch[1]}" alt="Generated image" style="max-width:100%;border-radius:8px;margin-top:8px;">`;
                    }

                    resultEl.innerHTML = `<div class="tool-result-header"><span class="tool-result-icon">📋</span> ${data.name} result</div><pre class="tool-result-output">${data.result}</pre>${imageHtml}`;
                    textEl.appendChild(resultEl);
                    textEl.innerHTML += '<span class="typing-indicator"></span>';
                    scrollToBottom();
                    setStatus('Generating...', 'yellow');
                }
                break;

            case 'error':
                isGenerating = false;
                addSystemMessage(data.content, 'error');
                setStatus('Error', 'red');
                break;
        }
    };
}

// ── Messages ─────────────────────────────────────────────────
function addMessage(role, text, streaming = false) {
    const messages = document.getElementById('messages');
    const welcome = messages.querySelector('.welcome-message');
    if (welcome) welcome.remove();

    const el = document.createElement('div');
    el.className = `message ${role}`;

    const avatar = role === 'user' ? '👤' : '🪹';
    const sender = role === 'user' ? 'You' : currentNickname;

    el.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            <div class="message-sender">${sender}</div>
            <div class="message-text">${text}${streaming ? '<span class="typing-indicator"></span>' : ''}</div>
        </div>
    `;

    messages.appendChild(el);
    scrollToBottom();
    return el;
}

function addSystemMessage(text, type = 'info') {
    const messages = document.getElementById('messages');
    const el = document.createElement('div');
    el.className = 'message system';
    el.innerHTML = `
        <div class="message-content" style="text-align: center; width: 100%;">
            <div class="message-text" style="color: var(--${type === 'error' ? 'red' : 'text-muted'}); font-size: 12px;">${text}</div>
        </div>
    `;
    messages.appendChild(el);
    scrollToBottom();
}

function scrollToBottom() {
    const area = document.getElementById('chatArea');
    area.scrollTop = area.scrollHeight;
}

// ── Send Message ─────────────────────────────────────────────
function sendMessage() {
    const input = document.getElementById('chatInput');
    const text = input.value.trim();
    if (!text || isGenerating || !ws || ws.readyState !== WebSocket.OPEN) return;

    addMessage('user', text);
    chatHistory.push({ role: 'user', text });
    saveHistory();
    input.value = '';
    input.style.height = '24px';

    ws.send(JSON.stringify({
        message: text,
        temperature: parseFloat(document.getElementById('temperature').value),
        top_p: parseFloat(document.getElementById('topP').value),
        max_tokens: parseInt(document.getElementById('maxTokens').value),
    }));
}

function handleKeydown(e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
}

function autoResize(el) {
    el.style.height = '24px';
    el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

// ── Model Management ─────────────────────────────────────────
async function loadModels() {
    try {
        const res = await fetch('/api/models');
        const data = await res.json();
        window._cachedLocalModels = data.local || [];

        archCategories = data.categories || {};

        // Update header model badge
        if (data.loaded_nickname) {
            document.getElementById('modelLabel').textContent = data.loaded_nickname;
            document.getElementById('modelDot').className = 'model-dot active';
            currentNickname = data.loaded_nickname;
        } else {
            document.getElementById('modelLabel').textContent = 'No model loaded';
            document.getElementById('modelDot').className = 'model-dot';
        }

        // Disk usage
        document.getElementById('diskUsage').textContent =
            `${data.disk_usage.total_gb} GB • ${data.disk_usage.model_count} models`;

        // Loaded section
        const loadedSection = document.getElementById('loadedModelSection');
        if (data.loaded) {
            const model = data.local.find(m => m.name === data.loaded);
            loadedSection.className = '';
            loadedSection.innerHTML = model ? `
                <div class="model-card">
                    <div class="model-card-header">
                        <span class="model-card-name">${model.display_name || model.name}</span>
                        <div class="model-card-badges">
                            <span class="badge badge-arch">${model.architecture}</span>
                            <span class="badge badge-size">${model.size_gb} GB</span>
                        </div>
                    </div>
                    <div class="model-card-actions">
                        <button class="btn btn-secondary" onclick="unloadModel()">Unload</button>
                    </div>
                </div>
            ` : `<div class="model-card"><span class="model-card-name">${data.loaded_nickname || data.loaded}</span></div>`;
        } else {
            loadedSection.className = 'empty-state';
            loadedSection.textContent = 'No model loaded';
        }

        // ── Downloaded models grouped by architecture ──
        const localList = document.getElementById('localModelsList');
        const unloaded = data.local.filter(m => m.name !== data.loaded);

        if (unloaded.length === 0) {
            localList.innerHTML = '<div class="empty-state">No downloaded models</div>';
        } else {
            localList.innerHTML = renderGroupedModels(unloaded, 'local');
        }

        // ── Available models grouped by architecture ──
        const catalogList = document.getElementById('catalogModelsList');
        const notDownloaded = data.catalog.filter(m => !m.downloaded);

        if (notDownloaded.length === 0) {
            catalogList.innerHTML = '<div class="empty-state">All catalog models downloaded!</div>';
        } else {
            catalogList.innerHTML = renderGroupedModels(notDownloaded, 'catalog');
        }

    } catch (e) {
        console.error('Failed to load models:', e);
    }
}

function renderGroupedModels(models, mode) {
    // Group by architecture
    const groups = {};
    models.forEach(m => {
        const arch = m.architecture || 'unknown';
        if (!groups[arch]) groups[arch] = [];
        groups[arch].push(m);
    });

    let html = '';
    for (const [arch, items] of Object.entries(groups)) {
        const cat = archCategories[arch] || { label: arch, type: '', desc: '', color: '#888' };

        html += `
            <div class="arch-group">
                <div class="arch-header">
                    <span class="arch-dot" style="background: ${cat.color}"></span>
                    <span class="arch-label">${cat.label}</span>
                    <span class="arch-type">${cat.type}</span>
                </div>
                <div class="arch-desc">${cat.desc}</div>
                ${items.map(m => renderModelCard(m, mode)).join('')}
            </div>
        `;
    }
    return html;
}

function renderModelCard(m, mode) {
    const name = m.display_name || m.name;
    const desc = m.description || '';

    if (mode === 'local') {
        return `
            <div class="model-card">
                <div class="model-card-header">
                    <span class="model-card-name">${name}</span>
                    <span class="badge badge-size">${m.size_gb} GB</span>
                </div>
                ${desc ? `<div class="model-card-desc">${desc}</div>` : ''}
                <div class="model-card-actions">
                    <button class="btn btn-primary" onclick="loadModel('${m.name}')">Load</button>
                    <button class="btn btn-danger" onclick="deleteModel('${m.filename}')">Delete</button>
                </div>
            </div>`;
    } else {
        return `
            <div class="model-card">
                <div class="model-card-header">
                    <span class="model-card-name">${name}</span>
                    <span class="badge badge-size">${m.size_gb} GB</span>
                </div>
                ${desc ? `<div class="model-card-desc">${desc}</div>` : ''}
                <div class="model-card-actions">
                    <button class="btn btn-download" onclick="downloadModel('${m.id}', this)">
                        Download (${m.size_gb} GB)
                    </button>
                </div>
            </div>`;
    }
}

let _loadingTimer = null;

function showLoadingOverlay(title, detail) {
    const overlay = document.getElementById('loadingOverlay');
    document.getElementById('loadingTitle').textContent = title;
    document.getElementById('loadingDetail').textContent = detail;
    document.getElementById('loadingTimer').textContent = '0.0s';
    overlay.classList.add('active');

    const t0 = performance.now();
    _loadingTimer = setInterval(() => {
        const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
        document.getElementById('loadingTimer').textContent = elapsed + 's';
    }, 100);
}

function hideLoadingOverlay() {
    clearInterval(_loadingTimer);
    _loadingTimer = null;
    document.getElementById('loadingOverlay').classList.remove('active');
}

async function loadModel(name) {
    // Find model info for display
    const localModels = window._cachedLocalModels || [];
    const modelInfo = localModels.find(m => m.name === name);
    const displayName = modelInfo ? modelInfo.display_name : name;
    const sizeText = modelInfo ? `${modelInfo.size_gb} GB • ${modelInfo.architecture.toUpperCase()}` : '';

    showLoadingOverlay('Loading ' + displayName, sizeText + ' — Initializing on GPU...');
    closeAllPanels();

    setStatus('Loading model...', 'yellow');
    document.getElementById('modelDot').className = 'model-dot loading';
    document.getElementById('modelLabel').textContent = 'Loading...';

    try {
        const res = await fetch('/api/models/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_name: name }),
        });
        const data = await res.json();

        if (res.ok) {
            setStatus('Ready', 'green');
            addSystemMessage(`Loaded ${data.info.version} ${data.info.size} — ${data.info.device.toUpperCase()} • ${data.info.load_time}s`);
            loadModels();
        } else {
            throw new Error(data.detail || 'Load failed');
        }
    } catch (e) {
        document.getElementById('modelDot').className = 'model-dot';
        document.getElementById('modelLabel').textContent = 'Load failed';
        setStatus('Error', 'red');
        addSystemMessage(`Failed to load: ${e.message}`, 'error');
    } finally {
        hideLoadingOverlay();
    }
}

async function unloadModel() {
    const currentLabel = document.getElementById('modelLabel').textContent;
    showLoadingOverlay('Unloading Model', currentLabel + ' — Freeing GPU memory...');
    closeAllPanels();

    try {
        await fetch('/api/models/unload', { method: 'POST' });
        document.getElementById('modelLabel').textContent = 'No model loaded';
        document.getElementById('modelDot').className = 'model-dot';
        currentNickname = 'Bird\'s Nest';
        setStatus('No model', 'yellow');
        loadModels();
    } catch (e) {
        addSystemMessage(`Failed to unload: ${e.message}`, 'error');
    } finally {
        hideLoadingOverlay();
    }
}

async function deleteModel(filename) {
    if (!confirm(`Delete ${filename}?`)) return;
    try {
        const res = await fetch(`/api/models/${encodeURIComponent(filename)}`, { method: 'DELETE' });
        const data = await res.json();
        if (res.ok) {
            addSystemMessage(`Deleted — freed ${data.freed_gb} GB`);
            loadModels();
        } else {
            addSystemMessage(data.detail || 'Delete failed', 'error');
        }
    } catch (e) {
        addSystemMessage(`Delete failed: ${e.message}`, 'error');
    }
}

async function downloadModel(catalogId, btn) {
    btn.disabled = true;
    btn.textContent = 'Downloading...';

    try {
        const res = await fetch('/api/models/download', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ catalog_id: catalogId }),
        });
        const data = await res.json();

        if (res.ok) {
            addSystemMessage(`Downloaded ${catalogId} (${data.size_gb} GB in ${data.time}s)`);
            loadModels();
        } else {
            throw new Error(data.detail || 'Download failed');
        }
    } catch (e) {
        addSystemMessage(`Download failed: ${e.message}`, 'error');
        btn.disabled = false;
        btn.textContent = 'Retry';
    }
}

// ── Reset ────────────────────────────────────────────────────
async function resetChat() {
    try { await fetch('/api/models/reset', { method: 'POST' }); } catch { }
    chatHistory = [];
    saveHistory();
    document.getElementById('messages').innerHTML = `
        <div class="welcome-message">
            <div class="welcome-icon">🪹</div>
            <h2>Conversation Reset</h2>
            <p>Memory cleared. Start fresh!</p>
        </div>
    `;
}

// ── History Persistence ─────────────────────────────────────
function saveHistory() {
    try {
        localStorage.setItem('birdsnest_history', JSON.stringify(chatHistory));
    } catch { }
}

function restoreHistory() {
    try {
        const saved = localStorage.getItem('birdsnest_history');
        if (!saved) return;
        chatHistory = JSON.parse(saved);
        if (!chatHistory.length) return;

        const messages = document.getElementById('messages');
        const welcome = messages.querySelector('.welcome-message');
        if (welcome) welcome.remove();

        chatHistory.forEach(entry => {
            // Skip tool-only responses — they were one-shot results
            if (entry.isTool) return;

            if (entry.role === 'user') {
                currentNickname = 'Bird\'s Nest'; // temporarily
                addMessage('user', entry.text);
            } else {
                currentNickname = entry.nickname || 'Bird\'s Nest';
                if (!entry.text) return; // skip empty entries
                const el = addMessage('assistant', entry.text);
                if (entry.stats && el) {
                    const statsEl = document.createElement('div');
                    statsEl.className = 'message-stats';
                    statsEl.textContent = `${entry.stats.tokens} tokens • ${entry.stats.tok_s} tok/s • ${entry.stats.time}s`;
                    el.querySelector('.message-content').appendChild(statsEl);
                }
            }
        });
    } catch { }
}

// ── Panels ───────────────────────────────────────────────────
function toggleModelPanel() {
    const panel = document.getElementById('modelPanel');
    const overlay = document.getElementById('panelOverlay');
    document.getElementById('settingsPanel').classList.remove('active');
    document.getElementById('docsPanel').classList.remove('active');
    panel.classList.toggle('active');
    overlay.classList.toggle('active', panel.classList.contains('active'));
    if (panel.classList.contains('active')) {
        loadModels();
    }
}

function toggleSettings() {
    const panel = document.getElementById('settingsPanel');
    const overlay = document.getElementById('panelOverlay');
    document.getElementById('modelPanel').classList.remove('active');
    document.getElementById('docsPanel').classList.remove('active');
    panel.classList.toggle('active');
    overlay.classList.toggle('active', panel.classList.contains('active'));
}

function toggleDocsPanel() {
    const panel = document.getElementById('docsPanel');
    const overlay = document.getElementById('panelOverlay');
    document.getElementById('modelPanel').classList.remove('active');
    document.getElementById('settingsPanel').classList.remove('active');
    panel.classList.toggle('active');
    overlay.classList.toggle('active', panel.classList.contains('active'));
    if (panel.classList.contains('active')) {
        loadRAGDocs();
    }
}

function closeAllPanels() {
    document.getElementById('modelPanel').classList.remove('active');
    document.getElementById('settingsPanel').classList.remove('active');
    document.getElementById('docsPanel').classList.remove('active');
    document.getElementById('panelOverlay').classList.remove('active');
}

// ── Settings ─────────────────────────────────────────────────
function setupSettings() {
    [
        { id: 'temperature', display: 'tempValue' },
        { id: 'topP', display: 'topPValue' },
        { id: 'maxTokens', display: 'maxTokensValue' },
    ].forEach(c => {
        const el = document.getElementById(c.id);
        el.addEventListener('input', () => {
            document.getElementById(c.display).textContent = el.value;
        });
    });
}

// ── Status ───────────────────────────────────────────────────
function setStatus(text, color) {
    const el = document.getElementById('statusText');
    el.textContent = text;
    el.style.color = color === 'green' ? 'var(--green)' :
        color === 'red' ? 'var(--red)' :
            color === 'yellow' ? 'var(--yellow)' : 'var(--text-muted)';
}

// ── RAG ──────────────────────────────────────────────────────
async function toggleRAG(enabled) {
    try {
        await fetch('/api/rag/toggle', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ enabled }),
        });
        addSystemMessage(enabled ? '📄 RAG enabled — documents will be used as context' : 'RAG disabled');
    } catch { }
}

async function uploadDocument(input) {
    if (!input.files.length) return;
    const file = input.files[0];
    const formData = new FormData();
    formData.append('file', file);

    addSystemMessage(`Uploading ${file.name}...`);

    try {
        const res = await fetch('/api/rag/upload', { method: 'POST', body: formData });
        const data = await res.json();
        if (res.ok) {
            addSystemMessage(`📄 Indexed "${data.filename}" — ${data.chunks} chunks, ${data.characters} chars (${data.time}s)`);
            loadRAGDocs();
            updateRAGStatus();
        } else {
            addSystemMessage(`Upload failed: ${data.detail}`, 'error');
        }
    } catch (e) {
        addSystemMessage(`Upload failed: ${e.message}`, 'error');
    }
    input.value = '';
}

async function loadRAGDocs() {
    try {
        const res = await fetch('/api/rag/documents');
        const data = await res.json();
        const list = document.getElementById('ragDocsList');
        const countEl = document.getElementById('docsCount');

        const docs = data.documents || [];
        if (countEl) {
            countEl.textContent = docs.length > 0 ? `${docs.length} doc${docs.length > 1 ? 's' : ''} • ${data.total_chunks || 0} chunks` : '';
        }

        if (docs.length === 0) {
            list.innerHTML = '<div class="empty-state" style="font-size:12px">No documents indexed yet.<br>Upload files to give your model context.</div>';
            return;
        }

        list.innerHTML = docs.map(d => `
            <div class="model-card" style="padding: 8px 12px;">
                <div class="model-card-header">
                    <span class="model-card-name" style="font-size:13px">📄 ${d.filename}</span>
                    <span class="badge badge-size">${d.chunk_count} chunks</span>
                </div>
                <div class="model-card-actions">
                    <button class="btn btn-danger" onclick="deleteRAGDoc('${d.doc_id}')">Remove</button>
                </div>
            </div>
        `).join('');
    } catch { }
}

async function deleteRAGDoc(docId) {
    try {
        const res = await fetch(`/api/rag/documents/${docId}`, { method: 'DELETE' });
        const data = await res.json();
        if (res.ok) {
            addSystemMessage(`Removed "${data.filename}" (${data.chunks_removed} chunks)`);
            loadRAGDocs();
            updateRAGStatus();
        }
    } catch { }
}

async function updateRAGStatus() {
    try {
        const res = await fetch('/api/rag/status');
        const data = await res.json();
        const el = document.getElementById('ragStatusText');
        if (el) {
            el.textContent = data.stats.total_documents > 0
                ? `${data.stats.total_documents} docs • ${data.stats.total_chunks} chunks indexed`
                : 'No documents indexed';
        }
        document.getElementById('ragToggle').checked = data.rag_enabled;
    } catch { }
}

// ── Tools ─────────────────────────────────────────────────────
async function toggleTools(enabled) {
    try {
        await fetch('/api/tools/toggle', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ enabled }),
        });
        addSystemMessage(enabled ? '🔧 Tools enabled — model can call functions' : '🔧 Tools disabled');
    } catch { }
}

async function toggleSingleTool(name, enabled) {
    try {
        await fetch('/api/tools/toggle', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ tool_name: name, tool_enabled: enabled }),
        });
    } catch { }
}

async function loadTools() {
    try {
        const res = await fetch('/api/tools');
        const data = await res.json();
        const list = document.getElementById('toolsList');
        if (!list) return;

        const toggle = document.getElementById('toolsToggle');
        if (toggle) toggle.checked = data.tools_enabled;

        if (!data.tools || data.tools.length === 0) {
            list.innerHTML = '<div class="empty-state" style="font-size:12px">No tools registered.</div>';
            return;
        }

        list.innerHTML = data.tools.map(t => `
            <div class="setting-row" style="padding: 4px 0;">
                <label style="font-size:12px;" title="${t.description}">
                    <span style="margin-right:4px;">🔧</span>${t.name}
                </label>
                <label class="toggle-switch" style="transform: scale(0.8);">
                    <input type="checkbox" ${t.enabled ? 'checked' : ''} onchange="toggleSingleTool('${t.name}', this.checked)">
                    <span class="toggle-slider"></span>
                </label>
            </div>
        `).join('');

        // Update sidebar count
        const countEl = document.getElementById('toolCount');
        if (countEl) countEl.textContent = data.tools.length;
    } catch { }
}

// ── Tools Sidebar ────────────────────────────────────────────────
function toggleToolsSidebar() {
    const sidebar = document.getElementById('toolsSidebar');
    const chatArea = document.getElementById('chatArea');
    const inputArea = document.querySelector('.input-area');
    const isOpen = sidebar.classList.toggle('open');

    if (chatArea) chatArea.classList.toggle('sidebar-open', isOpen);
    if (inputArea) inputArea.classList.toggle('sidebar-open', isOpen);

    // Close other panels when sidebar opens
    if (isOpen) {
        document.getElementById('settingsPanel')?.classList.remove('open');
        document.getElementById('docsPanel')?.classList.remove('open');
        document.getElementById('panelOverlay')?.classList.remove('active');
    }
}

function fireToolCard(card) {
    const type = card.dataset.tool;

    if (type === 'instant') {
        // Instant tool — send the prompt immediately
        const prompt = card.dataset.prompt;
        if (prompt) _sendAsUser(prompt);
        return;
    }

    if (type === 'input') {
        // Toggle active state
        const wasActive = card.classList.contains('active');

        // Close all other active cards
        document.querySelectorAll('.tool-card.active').forEach(c => c.classList.remove('active'));

        if (!wasActive) {
            card.classList.add('active');
            // Focus the input after animation
            const input = card.querySelector('.tool-input');
            if (input) {
                setTimeout(() => input.focus(), 200);
            }
        }
        return;
    }
}

function handleToolInput(event, input) {
    if (event.key !== 'Enter') return;
    event.preventDefault();
    event.stopPropagation();

    const value = input.value.trim();
    if (!value) return;

    const card = input.closest('.tool-card');
    const prefix = card.dataset.prefix || '';
    const action = card.dataset.action || '';

    let message;
    if (action === 'file') {
        message = `read file ${value}`;
    } else if (action === 'python') {
        message = `run python ${value}`;
    } else if (action === 'database') {
        // "path: SQL" format → "query database path: SQL"
        message = `query database ${value}`;
    } else if (prefix === 'translate ') {
        // "translate hello to spanish" — value should contain "text to lang"
        message = `translate ${value}`;
    } else {
        message = prefix + value;
    }

    _sendAsUser(message);
    input.value = '';
    card.classList.remove('active');
}

function sendQuickAction(prompt) {
    _sendAsUser(prompt);
}

function _sendAsUser(text) {
    const input = document.getElementById('chatInput');
    if (!input) return;
    input.value = text;
    sendMessage();
}

