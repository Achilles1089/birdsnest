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
    loadImageModels();
    setupDragDrop();
    document.getElementById('chatInput').focus();
});

// ── WebSocket ────────────────────────────────────────────────
let _updateChecked = false;

function connectWebSocket() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${proto}//${location.host}/ws/chat`);

    ws.onopen = () => {
        setStatus('Connected', 'green');
        document.getElementById('sendBtn').disabled = false;
        checkForUpdates();
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
                window._thinkState = { active: false, buffer: '' }; // Reset for new message
                setStatus(`Generating...`, 'yellow');
                document.getElementById('sendBtn').style.display = 'none';
                document.getElementById('stopBtn').style.display = '';
                break;

            case 'token':
                if (currentStreamEl) {
                    const textEl = currentStreamEl.querySelector('.message-text');
                    const cursor = textEl.querySelector('.typing-indicator');
                    if (cursor) cursor.remove();

                    const content = data.content;
                    const ts = window._thinkState;

                    // Accumulate into buffer
                    ts.buffer += content;

                    if (!ts.active) {
                        // Not yet in thinking mode — check if buffer contains <think>
                        const thinkIdx = ts.buffer.indexOf('<think>');
                        if (thinkIdx !== -1) {
                            // Enter thinking mode
                            ts.active = true;
                            const thinkBlock = document.createElement('details');
                            thinkBlock.className = 'think-block';
                            thinkBlock.open = true;
                            thinkBlock.innerHTML = '<summary class="think-header">🧠 Thinking…</summary><div class="think-content"></div>';
                            textEl.appendChild(thinkBlock);
                            // Keep only text after <think> in buffer for processing
                            ts.buffer = ts.buffer.slice(thinkIdx + 7);
                            // Fall through to process buffer as thinking content
                        } else if (ts.buffer.length > 10 && !ts.buffer.includes('<')) {
                            // Definitely not a think tag — flush buffer as normal text
                            textEl.appendChild(document.createTextNode(ts.buffer));
                            ts.buffer = '';
                        }
                        // else keep buffering — might still be building "<think>"
                    }

                    if (ts.active) {
                        // In thinking mode — check buffer for </think>
                        const closeIdx = ts.buffer.indexOf('</think>');
                        if (closeIdx !== -1) {
                            // Found close tag — split content
                            const thinkText = ts.buffer.slice(0, closeIdx);
                            const afterText = ts.buffer.slice(closeIdx + 8).replace(/^\n/, '');
                            ts.buffer = '';
                            ts.active = false;

                            const thinkBlock = textEl.querySelector('.think-block');
                            if (thinkBlock) {
                                if (thinkText) {
                                    thinkBlock.querySelector('.think-content').textContent += thinkText;
                                }
                                thinkBlock.querySelector('.think-header').textContent = '🧠 Thought process';
                                thinkBlock.open = false;
                            }
                            // Render response text after think block
                            if (afterText) {
                                textEl.appendChild(document.createTextNode(afterText));
                            }
                        } else {
                            // Still thinking — flush safe content (keep last 8 chars in buffer for boundary detection)
                            const thinkContent = textEl.querySelector('.think-block .think-content');
                            if (thinkContent && ts.buffer.length > 8) {
                                const safe = ts.buffer.slice(0, -8);
                                ts.buffer = ts.buffer.slice(-8);
                                thinkContent.textContent += safe;
                            }
                        }
                    }

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

                    // Flush any remaining think buffer
                    const ts = window._thinkState;
                    if (ts && ts.buffer) {
                        if (ts.active) {
                            // Still in thinking mode — flush to think block and close it
                            const thinkContent = textEl.querySelector('.think-block .think-content');
                            const thinkBlock = textEl.querySelector('.think-block');
                            const cleaned = ts.buffer.replace(/<\/?think>/g, '');
                            if (thinkContent && cleaned) thinkContent.textContent += cleaned;
                            if (thinkBlock) {
                                thinkBlock.querySelector('.think-header').textContent = '🧠 Thought process';
                                thinkBlock.open = false;
                            }
                        } else {
                            // Non-think buffer — flush as normal text
                            textEl.appendChild(document.createTextNode(ts.buffer));
                        }
                        ts.buffer = '';
                        ts.active = false;
                    }

                    // Detect if this was a tool-only response (has tool elements, no streamed text)
                    const hasToolResult = textEl.querySelector('.tool-result') !== null;
                    const finalText = textEl.textContent.trim();

                    if (hasToolResult) {
                        // Tool-only response — save with flag so restoreHistory skips it
                        chatHistory.push({ role: 'assistant', text: '', nickname: currentNickname, isTool: true });
                    } else if (finalText && finalText.length > 1 && !/^\d+$/.test(finalText)) {
                        // Regular model response — save the text (skip trivially short or digit-only junk)
                        chatHistory.push({ role: 'assistant', text: finalText, nickname: currentNickname, stats: data.stats });
                    }
                    saveHistory();

                    if (data.stats && data.stats.tokens > 0) {
                        const statsEl = document.createElement('div');
                        statsEl.className = 'message-stats';
                        statsEl.textContent = `${data.stats.tokens} tokens • ${data.stats.tok_s} tok/s • ${data.stats.time}s`;
                        currentStreamEl.querySelector('.message-content').appendChild(statsEl);
                        document.getElementById('tokStats').textContent = `${data.stats.tok_s} tok/s`;
                    }
                }
                currentStreamEl = null;
                setStatus('Ready', 'green');
                document.getElementById('sendBtn').style.display = '';
                document.getElementById('stopBtn').style.display = 'none';
                document.getElementById('chatInput').focus();
                break;

            case 'cancelled':
                isGenerating = false;
                if (currentStreamEl) {
                    const textEl = currentStreamEl.querySelector('.message-text');
                    const cursor = textEl.querySelector('.typing-indicator');
                    if (cursor) cursor.remove();

                    // Add stopped indicator
                    const stoppedEl = document.createElement('div');
                    stoppedEl.className = 'message-stats';
                    stoppedEl.style.color = 'var(--text-muted)';
                    if (data.stats && data.stats.tokens > 0) {
                        stoppedEl.textContent = `⏹ Stopped — ${data.stats.tokens} tokens • ${data.stats.tok_s} tok/s • ${data.stats.time}s`;
                        document.getElementById('tokStats').textContent = `${data.stats.tok_s} tok/s`;
                    } else {
                        stoppedEl.textContent = `⏹ Stopped`;
                    }
                    currentStreamEl.querySelector('.message-content').appendChild(stoppedEl);

                    // Save partial response
                    const partialText = textEl.textContent.trim();
                    if (partialText && partialText.length > 1) {
                        chatHistory.push({ role: 'assistant', text: partialText, nickname: currentNickname, stopped: true });
                        saveHistory();
                    }
                }
                currentStreamEl = null;
                setStatus('Ready', 'green');
                document.getElementById('sendBtn').style.display = '';
                document.getElementById('stopBtn').style.display = 'none';
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
                        'upscale_image': '🔍 Upscaling image',
                        'edit_image': '✂️ Editing image',
                        'search_files': '🔎 Searching files',
                        'screenshot': '📸 Taking screenshot',
                        'generate_music': '🎵 Generating music',
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

            case 'tool_result': {
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
                    const imgMatch = data.result && data.result.match(/URL:\s*(\/workspace\/\S+\.(?:png|jpg|jpeg|webp))/i);
                    let mediaHtml = '';

                    // Before/After for upscale/edit tools
                    if ((data.name === 'upscale_image' || data.name === 'edit_image') && imgMatch) {
                        const origMatch = data.result.match(/Original:\s*(\S+)/i);
                        const origName = origMatch ? origMatch[1] : null;
                        let origUrl = null;
                        if (origName) {
                            // Try both images/ and uploads/ paths
                            origUrl = `/workspace/images/${origName}`;
                        }
                        if (origUrl) {
                            mediaHtml = `<div class="image-comparison">
                                <div class="image-comparison-item">
                                    <img class="chat-image" src="${origUrl}" alt="Original" onclick="viewFullImage('${origUrl}')">
                                    <span class="image-comparison-label">Original</span>
                                </div>
                                <div class="image-comparison-item">
                                    <img class="chat-image" src="${imgMatch[1]}" alt="Upscaled" onclick="viewFullImage('${imgMatch[1]}')">
                                    <span class="image-comparison-label">Upscaled</span>
                                </div>
                            </div>`;
                        } else {
                            mediaHtml = `<img class="chat-image" src="${imgMatch[1]}" alt="Upscaled image" onclick="viewFullImage('${imgMatch[1]}')">`;
                        }
                    } else if (imgMatch) {
                        mediaHtml = `<img class="chat-image" src="${imgMatch[1]}" alt="Generated image" onclick="viewFullImage('${imgMatch[1]}')">`;
                    }

                    // Check if result contains an audio URL
                    const audioMatch = data.result && data.result.match(/Audio URL:\s*(\/workspace\/\S+\.wav)/i);
                    if (audioMatch) {
                        mediaHtml = `<div class="audio-player" style="margin-top:8px">
                            <audio controls preload="auto" style="width:100%;border-radius:8px;">
                                <source src="${audioMatch[1]}" type="audio/wav">
                            </audio>
                        </div>`;
                    }

                    // Check if result is structured JSON (image search)
                    try {
                        const parsed = JSON.parse(data.result);
                        if (parsed.type === 'image_results' && parsed.images) {
                            const gridHtml = renderImageGrid(parsed);
                            resultEl.innerHTML = `<div class="tool-result-header"><span class="tool-result-icon">🖼️</span> Image search: ${parsed.query}</div>${gridHtml}`;
                            textEl.appendChild(resultEl);
                            textEl.insertAdjacentHTML('beforeend', '<span class="typing-indicator"></span>');
                            scrollToBottom();
                            setStatus('Generating...', 'yellow');
                            break;
                        }
                    } catch (e) { /* Not JSON, continue with normal rendering */ }

                    // Linkify URLs in the result text
                    const linkified = (data.result || '').replace(
                        /(https?:\/\/[^\s<]+)/g,
                        '<a href="$1" target="_blank" rel="noopener" class="result-link">$1</a>'
                    );

                    resultEl.innerHTML = `<div class="tool-result-header"><span class="tool-result-icon">📋</span> ${data.name} result</div><pre class="tool-result-output">${linkified}</pre>${mediaHtml}`;
                    textEl.appendChild(resultEl);
                    textEl.insertAdjacentHTML('beforeend', '<span class="typing-indicator"></span>');
                    scrollToBottom();
                    setStatus('Generating...', 'yellow');
                }
                break;
            }

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

// ── Image Search Grid ─────────────────────────────────────────
function renderImageGrid(data) {
    const cards = data.images.map(img => {
        const thumb = img.thumbnail || img.image;
        const title = (img.title || '').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        const source = (img.source || '').replace(/</g, '&lt;');
        return `<a class="image-search-card" href="${img.image}" target="_blank" rel="noopener" title="${title}">
            <img src="${thumb}" alt="${title}" loading="lazy" onerror="this.closest('.image-search-card').style.display='none'">
            <div class="image-search-overlay">
                <span class="image-search-title">${title}</span>
                <span class="image-search-source">${source}</span>
            </div>
        </a>`;
    }).join('');
    return `<div class="image-search-grid">${cards}</div>
            <div class="image-search-meta">${data.count} images found for "${data.query}"</div>`;
}

// ── Web Search Result Cards ───────────────────────────────────
function renderSearchCards(data) {
    const cards = data.results.map(r => {
        const title = (r.title || '').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        const snippet = (r.snippet || '').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        const domain = (r.domain || '').replace(/</g, '&lt;');
        const favicon = r.favicon || '';
        return `<a class="search-result-card" href="${r.url}" target="_blank" rel="noopener">
            <div class="search-result-header">
                ${favicon ? `<img class="search-result-favicon" src="${favicon}" alt="" width="16" height="16" onerror="this.style.display='none'">` : ''}
                <span class="search-result-domain">${domain}</span>
            </div>
            <div class="search-result-title">${title}</div>
            ${snippet ? `<div class="search-result-snippet">${snippet}</div>` : ''}
        </a>`;
    }).join('');
    return `<div class="search-results-list">${cards}</div>`;
}

// ── Auto-Update Check ────────────────────────────────────────
async function checkForUpdates() {
    if (_updateChecked) return;
    _updateChecked = true;
    try {
        const res = await fetch('/api/update-check');
        const data = await res.json();
        if (data.update_available) {
            const messages = document.getElementById('messages');
            const banner = document.createElement('div');
            banner.className = 'update-banner';
            banner.innerHTML = `
                <span>🔄 Bird's Nest v${data.latest_version} is available!</span>
                ${data.download_url
                    ? `<a href="${data.download_url}" target="_blank" rel="noopener">Download</a>`
                    : ''}
                <button onclick="this.parentElement.remove()">✕</button>
            `;
            messages.insertBefore(banner, messages.firstChild);
        }
    } catch (e) { /* silent fail */ }
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

    // Build WS payload with optional image context
    const payload = {
        message: text,
        temperature: parseFloat(document.getElementById('temperature').value),
        top_p: parseFloat(document.getElementById('topP').value),
        max_tokens: parseInt(document.getElementById('maxTokens').value),
    };

    if (_pendingImage) {
        payload.image_path = _pendingImage.path;
        payload.image_url = _pendingImage.url;
        payload.image_filename = _pendingImage.filename;
        _pendingImage = null;
        input.placeholder = 'Type a message...';
    }

    ws.send(JSON.stringify(payload));
}

function handleKeydown(e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
}

function stopGeneration() {
    if (!isGenerating || !ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ type: 'cancel' }));
    setStatus('Stopping...', 'yellow');
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
            document.getElementById('modelLabel').textContent = 'No model';
            document.getElementById('modelDot').className = 'model-dot';
        }

        // Tab badge: total local models
        const aiBadge = document.getElementById('badgeAi');
        if (aiBadge) aiBadge.textContent = (data.local || []).length || '';

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
                <div class="arch-desc">${cat.desc}</div>`;

        if (arch === 'rwkv') {
            // Split RWKV into Thinking (G1) and Non-Thinking sub-groups
            const thinking = items.filter(m => m.thinking).sort((a, b) => a.size_gb - b.size_gb);
            const nonThinking = items.filter(m => !m.thinking).sort((a, b) => a.size_gb - b.size_gb);

            if (thinking.length > 0) {
                html += `
                    <details class="model-subgroup" open>
                        <summary class="subgroup-header">🧠 Thinking Models <span class="subgroup-count">${thinking.length}</span></summary>
                        <div class="subgroup-content">
                            ${thinking.map(m => renderModelCard(m, mode)).join('')}
                        </div>
                    </details>`;
            }
            if (nonThinking.length > 0) {
                html += `
                    <details class="model-subgroup">
                        <summary class="subgroup-header">💬 Chat Models <span class="subgroup-count">${nonThinking.length}</span></summary>
                        <div class="subgroup-content">
                            ${nonThinking.map(m => renderModelCard(m, mode)).join('')}
                        </div>
                    </details>`;
            }
        } else {
            // Non-RWKV architectures — render flat, sorted by size
            const sorted = items.sort((a, b) => a.size_gb - b.size_gb);
            html += sorted.map(m => renderModelCard(m, mode)).join('');
        }

        html += `</div>`;
    }
    return html;
}

function renderModelCard(m, mode) {
    const name = m.display_name || m.name;
    const desc = m.description || '';
    const thinkBadge = m.thinking ? '<span class="badge badge-think">🧠 Thinking</span>' : '';

    if (mode === 'local') {
        return `
            <div class="model-card">
                <div class="model-card-header">
                    <span class="model-card-name">${name}</span>
                    <span class="badge badge-size">${m.size_gb} GB</span>
                    ${thinkBadge}
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
                    ${thinkBadge}
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
        document.getElementById('modelLabel').textContent = 'No model';
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
                if (!entry.text || entry.text.length <= 1 || /^\d+$/.test(entry.text)) return; // skip empty/junk entries
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
function toggleModelPanel(tabId) {
    const panel = document.getElementById('modelPanel');
    const overlay = document.getElementById('panelOverlay');
    document.getElementById('settingsPanel').classList.remove('active');
    document.getElementById('docsPanel').classList.remove('active');
    panel.classList.toggle('active');
    overlay.classList.toggle('active', panel.classList.contains('active'));
    if (panel.classList.contains('active')) {
        if (tabId) {
            // Map tabId to the tab data attribute
            const tabMap = { tabAi: 'ai', tabImage: 'image', tabTranslation: 'translation', tabMusic: 'music', tabEmbed: 'embed' };
            const tab = tabMap[tabId] || 'ai';
            switchModelTab(tab);
        }
        loadModels();
        loadImageModels();
        loadTranslationModels();
        loadMusicModels();
        loadEmbeddingModels();
    }
}

// ── Tab Switching ─────────────────────────────────────
function switchModelTab(tab) {
    // Update tab buttons
    document.querySelectorAll('.model-tab').forEach(t => t.classList.remove('active'));
    document.querySelector(`.model-tab[data-tab="${tab}"]`).classList.add('active');

    // Update tab content
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    const tabId = 'tab' + tab.charAt(0).toUpperCase() + tab.slice(1);
    document.getElementById(tabId).classList.add('active');
}

// ── Image Models (mflux) ──────────────────────────────
// Image model catalog and state now come from server
let activeImageModel = localStorage.getItem('birdsnest_image_model') || '';

// Helper: build size badge showing download + quantized RAM
function imgSizeBadge(m) {
    if (m.q_size_gb && m.q_size_gb < m.size_gb) {
        return `<span class="badge badge-size">${m.size_gb} GB</span><span class="badge badge-size" style="color:#4ade80">~${m.q_size_gb} GB int4</span>`;
    }
    return `<span class="badge badge-size">${m.size_gb} GB</span>`;
}

async function loadImageModels() {
    let catalog = [];
    let installedRaw = [];
    try {
        const res = await fetch('/api/image-models');
        const data = await res.json();
        catalog = data.catalog || [];
        installedRaw = data.installed_raw || [];
        if (data.active) {
            activeImageModel = data.active;
            localStorage.setItem('birdsnest_image_model', data.active);
            const imgInfo = catalog.find(m => m.id === data.active);
            const imgName = imgInfo ? imgInfo.name : data.active.replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
            document.getElementById('imgModelLabel').textContent = imgName;
            document.getElementById('imgModelDot').className = 'model-dot active';
        }
    } catch { }

    // Tab badge
    const downloadedCount = catalog.filter(m => m.installed).length;
    const imgBadge = document.getElementById('badgeImage');
    if (imgBadge) imgBadge.textContent = downloadedCount || '';

    // ── Engine group config ──
    const engineGroups = {
        diffusers: { label: 'SDXL', icon: '⚡', type: 'Diffusers', desc: 'Ultra-fast distilled models via PyTorch MPS', color: '#f59e0b' },
        mflux: { label: 'FLUX / MLX', icon: '🔥', type: 'mflux', desc: 'High-quality Apple Silicon native models', color: '#7c6aef' },
    };

    // ── Helper: render one image model card ──
    function imgCard(m, mode) {
        const legacyBadge = m.legacy ? '<span class="badge" style="background:#f59e0b;color:#000;font-size:9px">LEGACY</span>' : '';
        const typeBadge = m.type === 'upscaler' ? '<span class="badge" style="background:#8b5cf6;color:#fff;font-size:9px">UPSCALER</span>' : '';

        let actions = '';
        if (mode === 'downloaded') {
            const rawMatch = installedRaw.find(r => {
                const repoLower = (m.hf_repo || '').toLowerCase();
                const idLower = (r.id || '').toLowerCase();
                return repoLower.includes(idLower) || idLower.includes(repoLower.split('/').pop());
            });
            const deleteBtn = rawMatch
                ? `<button class="btn btn-danger" onclick="deleteHfModel('image', '${rawMatch.dir_name}', '${m.name}')">Delete</button>`
                : '';
            actions = `
                <button class="btn btn-primary" onclick="selectImageModel('${m.id}')">Select</button>
                ${deleteBtn}`;
        } else {
            actions = `
                <button class="btn btn-download" onclick="downloadImageModel('${m.id}', this)">
                    Download (${m.size_gb} GB)
                </button>`;
        }

        return `
            <div class="model-card">
                <div class="model-card-header">
                    <span class="model-card-name">${m.name}</span>
                    <div class="model-card-badges">
                        <span class="badge badge-arch">${m.params || ''}</span>
                        <span class="badge badge-size">${m.steps} steps</span>
                        ${imgSizeBadge(m)}
                        ${legacyBadge}${typeBadge}
                    </div>
                </div>
                <div class="model-card-desc">${m.desc}</div>
                <div class="model-card-actions">${actions}</div>
            </div>`;
    }

    // ── Helper: render grouped list ──
    function renderGrouped(models, mode) {
        const groups = {};
        models.forEach(m => {
            const eng = m.engine || 'mflux';
            if (!groups[eng]) groups[eng] = [];
            groups[eng].push(m);
        });

        let html = '';
        // Show diffusers first (speed models), then mflux
        for (const eng of ['diffusers', 'mflux']) {
            const items = groups[eng];
            if (!items || items.length === 0) continue;
            const g = engineGroups[eng] || engineGroups.mflux;

            html += `
                <div class="arch-group">
                    <div class="arch-header">
                        <span class="arch-dot" style="background: ${g.color}"></span>
                        <span class="arch-label">${g.label}</span>
                        <span class="arch-type">${g.type}</span>
                    </div>
                    <div class="arch-desc">${g.desc}</div>
                    <details class="model-subgroup" open>
                        <summary class="subgroup-header">${g.icon} ${g.label} Models <span class="subgroup-count">${items.length}</span></summary>
                        <div class="subgroup-content">
                            ${items.map(m => imgCard(m, mode)).join('')}
                        </div>
                    </details>
                </div>`;
        }
        return html;
    }

    // ── LOADED section — only show if model is both selected AND installed ──
    const activeSection = document.getElementById('activeImageModel');
    const active = catalog.find(m => m.id === activeImageModel && m.installed);
    if (active) {

        // Build resolution picker from model's supported resolutions
        const resolutions = active.resolutions || [];
        const curW = parseInt(localStorage.getItem('birdsnest_img_width') || '1024');
        const curH = parseInt(localStorage.getItem('birdsnest_img_height') || '1024');
        let resolutionHtml = '';
        if (resolutions.length > 0) {
            const presetBtns = resolutions.map(([w, h]) => {
                const label = w === h ? `${w}²` : `${w}×${h}`;
                const isActive = w === curW && h === curH ? 'active' : '';
                return `<button class="img-preset ${isActive}" onclick="setImagePreset(${w},${h})" data-w="${w}" data-h="${h}">${label}</button>`;
            }).join('');
            resolutionHtml = `
                <div class="img-setting-group" style="margin-top:8px">
                    <label class="img-setting-label">📐 Resolution</label>
                    <div class="img-preset-grid">${presetBtns}</div>
                </div>`;
        }

        activeSection.className = '';
        activeSection.innerHTML = `
            <div class="model-card">
                <div class="model-card-header">
                    <span class="model-card-name">${active.name}</span>
                    <div class="model-card-badges">
                        <span class="badge badge-arch">${active.params || ''}</span>
                        <span class="badge badge-size">${active.steps} steps</span>
                        ${imgSizeBadge(active)}
                    </div>
                </div>
                <div class="model-card-desc">${active.desc}</div>
                ${resolutionHtml}
                <div class="model-card-actions">
                    <button class="warm-btn" id="warmImageBtn" onclick="warmImageEngine()"
                        title="Pre-load model into GPU for instant generation">🔥 Warm Engine</button>
                    <button class="btn btn-secondary" onclick="unloadImageModel()">Unload</button>
                </div>
            </div>`;
    } else {
        activeSection.className = 'empty-state';
        activeSection.textContent = 'No image model loaded';
    }

    // ── DOWNLOADED section — collapsible by engine ──
    const installedSection = document.getElementById('imageInstalledList');
    const downloaded = catalog.filter(m => m.installed && m.id !== activeImageModel);
    if (downloaded.length === 0) {
        installedSection.innerHTML = '<div class="empty-state">No downloaded image models</div>';
    } else {
        installedSection.innerHTML = renderGrouped(downloaded, 'downloaded');
    }

    // ── AVAILABLE TO DOWNLOAD — collapsible by engine ──
    const list = document.getElementById('imageModelsList');
    const notDownloaded = catalog.filter(m => !m.installed);
    if (notDownloaded.length === 0) {
        list.innerHTML = '<div class="empty-state">All image models downloaded!</div>';
    } else {
        list.innerHTML = renderGrouped(notDownloaded, 'available');
    }
}

async function downloadImageModel(modelId, btn) {
    btn.disabled = true;
    btn.textContent = 'Downloading...';

    try {
        const res = await fetch('/api/image-models/download', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_id: modelId }),
        });
        const data = await res.json();

        if (res.ok) {
            addSystemMessage(`Downloaded ${modelId} (${data.size_gb} GB in ${data.time}s)`);
            loadImageModels();
        } else {
            throw new Error(data.detail || 'Download failed');
        }
    } catch (e) {
        addSystemMessage(`Image model download failed: ${e.message}`, 'error');
        btn.disabled = false;
        btn.textContent = 'Retry';
    }
}

async function selectImageModel(id) {
    activeImageModel = id;
    localStorage.setItem('birdsnest_image_model', id);
    await fetch('/api/image-models/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: id }),
    });
    // Update header badge
    const label = id.replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    document.getElementById('imgModelLabel').textContent = label;
    document.getElementById('imgModelDot').className = 'model-dot active';

    // Auto-set resolution to model's default
    try {
        const res = await fetch('/api/image-models');
        const data = await res.json();
        const entry = (data.catalog || []).find(m => m.id === id);
        if (entry && entry.default_resolution) {
            const [w, h] = entry.default_resolution;
            setImagePreset(w, h);
        }
    } catch { }

    addSystemMessage(`Image model set to ${id}`);
    loadImageModels();
}

async function unloadImageModel() {
    activeImageModel = '';
    localStorage.removeItem('birdsnest_image_model');
    await fetch('/api/image-models/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: '' }),
    }).catch(() => { });
    // Update header badge
    document.getElementById('imgModelLabel').textContent = 'No image model';
    document.getElementById('imgModelDot').className = 'model-dot dim';
    addSystemMessage('Image model unloaded');
    loadImageModels();
}

// ── Image Performance Settings ──────────────────────────────────
function setImagePreset(w, h) {
    localStorage.setItem('birdsnest_img_width', w);
    localStorage.setItem('birdsnest_img_height', h);
    // Update active button
    document.querySelectorAll('.img-preset').forEach(btn => {
        btn.classList.toggle('active', parseInt(btn.dataset.w) === w && parseInt(btn.dataset.h) === h);
    });
    // Sync to server
    fetch('/api/image-settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ width: w, height: h }),
    }).catch(() => { });
    addSystemMessage(`Image resolution set to ${w}×${h}`);
}

function loadImageSettings() {
    // Legacy stub — resolution is now per-model in the loaded card
}

async function warmImageEngine() {
    const btn = document.getElementById('warmImageBtn');
    if (btn) { btn.disabled = true; btn.textContent = '🔥 Loading...'; }
    try {
        const resp = await fetch('/api/image-engine/warm', { method: 'POST' });
        const data = await resp.json();
        if (data.status === 'warm') {
            const model = data.model || 'unknown';
            const elapsed = data.elapsed ? `${data.elapsed}s` : '';
            const loadTime = data.engine?.load_time ? ` (loaded in ${data.engine.load_time}s)` : '';
            addSystemMessage(`🔥 Image engine ready — ${model} Q4 in GPU memory${loadTime}. Generation will be fast.`);
        } else {
            addSystemMessage(`⚠️ Image warm failed: ${data.error || 'Unknown error'}`);
        }
    } catch (e) {
        addSystemMessage('⚠️ Could not warm image engine');
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = '🔥 Warm Image Engine'; }
    }
}

async function unloadMusicModel() {
    activeMusicModel = '';
    localStorage.removeItem('birdsnest_music_model');
    await fetch('/api/music-models/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: '' }),
    }).catch(() => { });
    addSystemMessage('Music model unloaded');
    loadMusicModels();
}

function unloadEmbedModel() {
    activeEmbedModel = 'tfidf';
    localStorage.removeItem('birdsnest_embed_model');
    addSystemMessage('Embed model reset to default (TF-IDF)');
    loadEmbeddingModels();
}

// ── Unified HF Model Delete ─────────────────────────
async function deleteHfModel(type, dirName, displayName) {
    if (!confirm(`Delete ${displayName}?\nThis will remove it from the HuggingFace cache and free disk space.`)) return;
    try {
        const res = await fetch(`/api/${type}-models/${encodeURIComponent(dirName)}`, { method: 'DELETE' });
        const data = await res.json();
        if (data.success) {
            addSystemMessage(`Deleted ${displayName} — freed ${data.freed_gb} GB`);
            // Reload the appropriate tab
            if (type === 'image') loadImageModels();
            else if (type === 'music') loadMusicModels();
            else if (type === 'embed') loadEmbeddingModels();
        } else {
            addSystemMessage(data.error || 'Delete failed', 'error');
        }
    } catch (e) {
        addSystemMessage(`Delete failed: ${e.message}`, 'error');
    }
}

// ── Translation Models (Opus-MT) ──────────────────────
const TRANSLATION_CATALOG = [
    { pair: 'en-es', from: 'English', to: 'Spanish', flag: '🇪🇸' },
    { pair: 'en-fr', from: 'English', to: 'French', flag: '🇫🇷' },
    { pair: 'en-de', from: 'English', to: 'German', flag: '🇩🇪' },
    { pair: 'en-it', from: 'English', to: 'Italian', flag: '🇮🇹' },
    { pair: 'en-pt', from: 'English', to: 'Portuguese', flag: '🇵🇹' },
    { pair: 'en-ru', from: 'English', to: 'Russian', flag: '🇷🇺' },
    { pair: 'en-zh', from: 'English', to: 'Chinese', flag: '🇨🇳' },
    { pair: 'en-ja', from: 'English', to: 'Japanese', flag: '🇯🇵' },
    { pair: 'en-ko', from: 'English', to: 'Korean', flag: '🇰🇷' },
    { pair: 'en-ar', from: 'English', to: 'Arabic', flag: '🇸🇦' },
    { pair: 'en-nl', from: 'English', to: 'Dutch', flag: '🇳🇱' },
    { pair: 'en-tr', from: 'English', to: 'Turkish', flag: '🇹🇷' },
    { pair: 'en-pl', from: 'English', to: 'Polish', flag: '🇵🇱' },
    { pair: 'en-sv', from: 'English', to: 'Swedish', flag: '🇸🇪' },
    { pair: 'en-hi', from: 'English', to: 'Hindi', flag: '🇮🇳' },
    { pair: 'es-en', from: 'Spanish', to: 'English', flag: '🇬🇧' },
];

async function loadTranslationModels() {
    try {
        const res = await fetch('/api/translation-models');
        const data = await res.json();
        const installed = new Set(data.installed || []);

        const installedList = document.getElementById('installedTranslations');
        const availableList = document.getElementById('availableTranslations');

        const installedPairs = TRANSLATION_CATALOG.filter(m => installed.has(m.pair));
        const availablePairs = TRANSLATION_CATALOG.filter(m => !installed.has(m.pair));

        // Tab badge
        const langBadge = document.getElementById('badgeLang');
        if (langBadge) langBadge.textContent = installedPairs.length || '';

        installedList.innerHTML = installedPairs.length > 0
            ? installedPairs.map(m => `
                <div class="model-card">
                    <div class="model-card-header">
                        <span class="model-card-name">${m.flag} ${m.from} → ${m.to}</span>
                        <span class="badge badge-size" style="color:#4ade80">Installed</span>
                    </div>
                    <div class="model-card-actions">
                        <button class="btn btn-danger" onclick="deleteTranslationModel('${m.pair}', '${m.from} → ${m.to}')">Delete</button>
                    </div>
                </div>
            `).join('')
            : '<div class="empty-state">No translation pairs installed yet</div>';

        availableList.innerHTML = availablePairs.map(m => `
            <div class="model-card">
                <div class="model-card-header">
                    <span class="model-card-name">${m.flag} ${m.from} → ${m.to}</span>
                    <span class="badge badge-size">~300 MB</span>
                </div>
                <div class="model-card-actions">
                    <button class="btn btn-download" onclick="downloadTranslationModel('${m.pair}', this)">
                        Download
                    </button>
                </div>
            </div>
        `).join('');

    } catch {
        document.getElementById('installedTranslations').innerHTML =
            '<div class="empty-state">Translation API loading...</div>';
    }
}

async function downloadTranslationModel(pair, btn) {
    btn.disabled = true;
    btn.textContent = 'Downloading...';
    try {
        const res = await fetch('/api/translation-models/download', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ pair }),
        });
        const data = await res.json();
        if (data.success) {
            loadTranslationModels();
        } else {
            btn.textContent = 'Error';
        }
    } catch {
        btn.textContent = 'Error';
    }
}

async function deleteTranslationModel(pair, displayName) {
    if (!confirm(`Delete translation model: ${displayName}?\nThis will free disk space.`)) return;
    try {
        const res = await fetch(`/api/translation-models/${encodeURIComponent(pair)}`, { method: 'DELETE' });
        const data = await res.json();
        if (data.success) {
            addSystemMessage(`Deleted ${displayName} — freed ${data.freed_gb} GB`);
            loadTranslationModels();
        } else {
            addSystemMessage(data.error || 'Delete failed', 'error');
        }
    } catch (e) {
        addSystemMessage(`Delete failed: ${e.message}`, 'error');
    }
}

// ── Music Models (Stable Audio + Riffusion) ─────────

const MUSIC_ENGINE_GROUPS = {
    'stable-audio': { label: 'Stable Audio', icon: '🎵', color: '#818cf8', desc: 'High-quality music & sound effects' },
    'riffusion': { label: 'Riffusion', icon: '🎸', color: '#f472b6', desc: 'Spectrogram-based creative audio' },
};

let activeMusicModel = localStorage.getItem('birdsnest_music_model') || '';

async function loadMusicModels() {
    let catalog = [], installed = [], activeId = '';
    try {
        const res = await fetch('/api/music-models');
        const data = await res.json();
        catalog = data.catalog || [];
        installed = data.installed || [];
        activeId = data.active || '';
    } catch { }

    // Sync local state with server
    activeMusicModel = activeId;
    if (activeId) localStorage.setItem('birdsnest_music_model', activeId);

    // Tab badge — count downloaded
    const badge = document.getElementById('badgeMusic');
    const downloadedCount = catalog.filter(m => m.installed).length;
    if (badge) badge.textContent = downloadedCount || '';

    // Helper: render a music card
    function musicCard(m, mode) {
        const sizeBadge = m.size_gb ? `<span class="badge">${m.size_gb} GB</span>` : '';
        const durBadge = m.max_duration ? `<span class="badge badge-size">${m.max_duration}s max</span>` : '';

        let actions = '';
        if (mode === 'loaded') {
            actions = `<button class="btn btn-secondary" onclick="unloadMusicModel()">Unload</button>`;
        } else if (mode === 'downloaded') {
            actions = `
                <button class="btn btn-primary" onclick="selectMusicModel('${m.id}')">Select</button>
                <button class="btn btn-danger" onclick="deleteMusicModel('${m.id}')">Delete</button>`;
        } else {
            actions = `<button class="btn btn-download" onclick="downloadMusicModel('${m.id}')">Download (${m.size_gb || '?'} GB)</button>`;
        }

        return `
            <div class="model-card">
                <div class="model-card-header">
                    <span class="model-card-name">${m.name}</span>
                    <div class="model-card-badges">${durBadge}${sizeBadge}</div>
                </div>
                <div class="model-card-desc">${m.description || ''}</div>
                <div class="model-card-actions">${actions}</div>
            </div>`;
    }

    // Helper: group cards by engine into collapsible <details>
    function renderGroups(models, mode) {
        const byEngine = {};
        models.forEach(m => {
            const eng = m.engine || 'unknown';
            if (!byEngine[eng]) byEngine[eng] = [];
            byEngine[eng].push(m);
        });

        let html = '';
        for (const [eng, items] of Object.entries(byEngine)) {
            const g = MUSIC_ENGINE_GROUPS[eng] || { label: eng, icon: '🎵', color: '#888', desc: '' };
            html += `
                <div class="arch-group">
                    <div class="arch-header">
                        <span class="arch-dot" style="background:${g.color}"></span>
                        <span class="arch-label">${g.label}</span>
                        <span class="arch-type">${eng.toUpperCase()}</span>
                    </div>
                    <div class="arch-desc">${g.desc}</div>
                    <details class="model-subgroup" open>
                        <summary class="subgroup-header">${g.icon} ${g.label} Models <span class="subgroup-count">${items.length}</span></summary>
                        <div class="subgroup-content">
                            ${items.map(m => musicCard(m, mode)).join('')}
                        </div>
                    </details>
                </div>`;
        }
        return html;
    }

    // ── LOADED section — only show if model is both selected AND installed ──
    const activeSection = document.getElementById('activeMusicModel');
    const active = catalog.find(m => m.id === activeId && m.installed);
    if (active) {
        activeSection.className = '';
        activeSection.innerHTML = `
            <div class="model-card">
                <div class="model-card-header">
                    <span class="model-card-name">${active.name}</span>
                    <div class="model-card-badges">
                        ${active.max_duration ? `<span class="badge badge-size">${active.max_duration}s max</span>` : ''}
                        ${active.size_gb ? `<span class="badge">${active.size_gb} GB</span>` : ''}
                    </div>
                </div>
                <div class="model-card-desc">${active.description || ''}</div>
                <div class="model-card-actions">
                    <button class="btn btn-secondary" onclick="unloadMusicModel()">Unload</button>
                </div>
            </div>`;
    } else {
        activeSection.className = 'empty-state';
        activeSection.textContent = 'No music model loaded';
    }

    // ── DOWNLOADED section ──
    const downloadedModels = catalog.filter(m => m.installed && m.id !== activeId);
    const dlSection = document.getElementById('musicInstalledList');
    if (downloadedModels.length > 0) {
        dlSection.innerHTML = renderGroups(downloadedModels, 'downloaded');
    } else {
        dlSection.innerHTML = '<div class="empty-state">No downloaded music models</div>';
    }

    // ── AVAILABLE section ──
    const availableModels = catalog.filter(m => !m.installed);
    const avSection = document.getElementById('musicModelsList');
    if (availableModels.length > 0) {
        avSection.innerHTML = renderGroups(availableModels, 'available');
    } else {
        avSection.innerHTML = '<div class="empty-state">All music models downloaded!</div>';
    }
}

async function selectMusicModel(id) {
    activeMusicModel = id;
    localStorage.setItem('birdsnest_music_model', id);
    await fetch('/api/music-models/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: id }),
    }).catch(() => { });
    loadMusicModels();
}

async function downloadMusicModel(id) {
    const btn = event.target;
    btn.disabled = true;
    btn.textContent = 'Downloading...';
    try {
        const res = await fetch('/api/music-models/download', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: id }),
        });
        const data = await res.json();
        if (data.status === 'downloaded') {
            addSystemMessage(`✅ Music model downloaded in ${data.time}s`);
        } else {
            addSystemMessage(`⚠️ Download failed: ${data.message}`);
        }
    } catch (e) {
        addSystemMessage(`⚠️ Download error: ${e.message}`);
    }
    loadMusicModels();
}

async function deleteMusicModel(id) {
    // Find the dir_name in installed list from last fetch
    try {
        const res = await fetch('/api/music-models');
        const data = await res.json();
        const entry = (data.catalog || []).find(m => m.id === id);
        const installed = data.installed || [];
        const match = installed.find(m => m.id.includes(entry?.hf_repo?.split('/')[1] || ''));
        if (match) {
            await fetch(`/api/music-models/${match.dir_name}`, { method: 'DELETE' });
            addSystemMessage(`🗑️ Deleted music model: ${id}`);
        }
    } catch { }
    loadMusicModels();
}

// ── Embedding Models (RAG) ───────────────────────────
const EMBED_MODEL_CATALOG = [
    { id: 'tfidf', name: 'TF-IDF (Built-in)', size: '0 MB', quality: '⭐⭐', desc: 'Fast keyword matching, no download' },
    { id: 'all-MiniLM-L6-v2', name: 'MiniLM-L6', size: '80 MB', quality: '⭐⭐⭐⭐', desc: 'Best balance of speed & quality' },
    { id: 'nomic-embed-text-v1.5', name: 'Nomic Embed', size: '275 MB', quality: '⭐⭐⭐⭐⭐', desc: 'Top-tier semantic search' },
    { id: 'BAAI/bge-small-en-v1.5', name: 'BGE Small', size: '130 MB', quality: '⭐⭐⭐⭐', desc: 'High quality, small footprint' },
];

let activeEmbedModel = localStorage.getItem('birdsnest_embed_model') || 'tfidf';

async function loadEmbeddingModels() {
    // Fetch installed from server
    let installed = [];
    try {
        const res = await fetch('/api/embed-models');
        const data = await res.json();
        installed = data.installed || [];
    } catch { }

    // Tab badge
    const badge = document.getElementById('badgeEmbed');
    if (badge) badge.textContent = installed.length || '';

    // Active model display
    const active = EMBED_MODEL_CATALOG.find(m => m.id === activeEmbedModel) || EMBED_MODEL_CATALOG[0];
    document.getElementById('activeEmbedModel').innerHTML = `
        <div class="model-card">
            <div class="model-card-header">
                <span class="model-card-name">${active.name}</span>
                <div class="model-card-badges">
                    <span class="badge badge-arch">${active.quality}</span>
                    <span class="badge badge-size">${active.size}</span>
                </div>
            </div>
            <div class="model-card-desc">${active.desc}</div>
            <div class="model-card-actions">
                <button class="btn btn-danger" onclick="unloadEmbedModel()">Unload</button>
            </div>
        </div>
    `;
    document.getElementById('activeEmbedModel').className = '';

    // Installed from HF cache (with delete)
    const installedSection = document.getElementById('embedInstalledList');
    if (installedSection) {
        installedSection.innerHTML = installed.length > 0
            ? installed.map(m => `
                <div class="model-card">
                    <div class="model-card-header">
                        <span class="model-card-name">${m.id}</span>
                        <span class="badge badge-size" style="color:#4ade80">${m.size_gb} GB</span>
                    </div>
                    <div class="model-card-actions">
                        <button class="btn btn-danger" onclick="deleteHfModel('embed', '${m.dir_name}', '${m.id}')">Delete</button>
                    </div>
                </div>
            `).join('')
            : '<div class="empty-state">No embedding models cached</div>';
    }

    // Available catalog
    const list = document.getElementById('embedModelsList');
    const others = EMBED_MODEL_CATALOG.filter(m => m.id !== activeEmbedModel);
    list.innerHTML = others.map(m => `
        <div class="model-card">
            <div class="model-card-header">
                <span class="model-card-name">${m.name}</span>
                <div class="model-card-badges">
                    <span class="badge badge-arch">${m.quality}</span>
                    <span class="badge badge-size">${m.size}</span>
                </div>
            </div>
            <div class="model-card-desc">${m.desc}</div>
            <div class="model-card-actions">
                <button class="btn btn-primary" onclick="selectEmbedModel('${m.id}')">Select</button>
            </div>
        </div>
    `).join('');
}

function selectEmbedModel(id) {
    activeEmbedModel = id;
    localStorage.setItem('birdsnest_embed_model', id);
    loadEmbeddingModels();
}

// ── System Monitor ──────────────────────────────────
let _systemMonitorInterval = null;

function toggleSystemMonitor() {
    const statsEl = document.getElementById('systemStats');
    const btn = document.getElementById('systemMonitorBtn');
    const isVisible = statsEl.classList.contains('visible');

    if (isVisible) {
        statsEl.classList.remove('visible');
        btn.classList.remove('active');
        if (_systemMonitorInterval) {
            clearInterval(_systemMonitorInterval);
            _systemMonitorInterval = null;
        }
    } else {
        statsEl.classList.add('visible');
        btn.classList.add('active');
        fetchSystemStats();
        _systemMonitorInterval = setInterval(fetchSystemStats, 5000);
    }
}

async function fetchSystemStats() {
    try {
        const res = await fetch('/api/system-stats');
        const data = await res.json();
        const statsEl = document.getElementById('systemStats');
        statsEl.innerHTML = `
            <span><span class="stat-label">RAM</span> <span class="stat-value">${data.ram_gb} GB</span></span>
            <span class="stat-divider">│</span>
            <span><span class="stat-label">Disk</span> <span class="stat-value">${data.disk_gb} GB</span></span>
            <span class="stat-divider">│</span>
            <span><span class="stat-label">Models</span> <span class="stat-value">${data.model_count}</span></span>
            ${data.loaded ? `<span class="stat-divider">│</span><span><span class="stat-value" style="color:#8b9cf7">● ${data.loaded.split('-').pop()}</span></span>` : ''}
        `;
    } catch { }
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
    document.getElementById('imageLibraryPanel').classList.remove('active');
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

const TOOL_DISPLAY_NAMES = {
    get_current_time: { label: 'Clock & Time', icon: '🕐' },
    calculate: { label: 'Calculator', icon: '🧮' },
    get_system_info: { label: 'System Info', icon: '💻' },
    search_web: { label: 'Web Search', icon: '🔍' },
    fetch_url: { label: 'Read Webpage', icon: '🌐' },
    read_file: { label: 'Read File', icon: '📄' },
    write_file: { label: 'Write File', icon: '✏️' },
    run_python: { label: 'Run Python', icon: '🐍' },
    list_directory: { label: 'Browse Files', icon: '📁' },
    run_shell: { label: 'Terminal', icon: '💲' },
    youtube_transcript: { label: 'YouTube', icon: '📺' },
    memory: { label: 'Memory', icon: '🧠' },
    weather: { label: 'Weather', icon: '🌤️' },
    clipboard: { label: 'Clipboard', icon: '📋' },
    screenshot: { label: 'Screenshot', icon: '📸' },
    todo: { label: 'Task List', icon: '✅' },
    translate: { label: 'Translate', icon: '🌍' },
    generate_image: { label: 'Image Gen', icon: '🎨' },
    search_images: { label: 'Image Search', icon: '🖼️' },
    query_database: { label: 'Database', icon: '🗄️' },
    generate_music: { label: 'Music Gen', icon: '🎵' },
};

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

        list.innerHTML = data.tools.map(t => {
            const display = TOOL_DISPLAY_NAMES[t.name] || { label: t.name, icon: '🔧' };
            return `
                <div class="tool-card" title="${t.description}">
                    <div class="tool-card-top">
                        <span class="tool-icon">${display.icon}</span>
                        <span class="tool-label">${display.label}</span>
                    </div>
                    <label class="toggle-switch toggle-sm">
                        <input type="checkbox" ${t.enabled ? 'checked' : ''} onchange="toggleSingleTool('${t.name}', this.checked)">
                        <span class="toggle-slider"></span>
                    </label>
                </div>`;
        }).join('');

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
        // Populate chat input with the prefix and focus it
        const prefix = card.dataset.prefix || '';
        const chatInput = document.getElementById('chatInput');
        if (chatInput) {
            chatInput.value = prefix;
            chatInput.focus();
            // Move cursor to end
            chatInput.setSelectionRange(prefix.length, prefix.length);
            autoResize(chatInput);
            // Enable send button
            document.getElementById('sendBtn').disabled = false;
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


// ── Drag & Drop Image Upload ─────────────────────────────────────

let _pendingImage = null; // {url, path, filename}

function setupDragDrop() {
    const chatArea = document.getElementById('chatArea');
    const dropOverlay = document.getElementById('dropOverlay');
    let dragCounter = 0;

    document.body.addEventListener('dragenter', (e) => {
        e.preventDefault();
        if (e.dataTransfer.types.includes('Files')) {
            dragCounter++;
            dropOverlay.classList.add('active');
        }
    });

    document.body.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dragCounter--;
        if (dragCounter <= 0) {
            dragCounter = 0;
            dropOverlay.classList.remove('active');
        }
    });

    document.body.addEventListener('dragover', (e) => {
        e.preventDefault();
    });

    document.body.addEventListener('drop', async (e) => {
        e.preventDefault();
        dragCounter = 0;
        dropOverlay.classList.remove('active');

        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            await handleImageUpload(file);
        }
    });
}

function handleImageFileSelect(input) {
    const file = input.files[0];
    if (file && file.type.startsWith('image/')) {
        handleImageUpload(file);
    }
    input.value = ''; // Reset so same file can be selected again
}

async function handleImageUpload(file) {
    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch('/api/upload-image', {
            method: 'POST',
            body: formData,
        });
        const data = await res.json();

        if (!res.ok) {
            addSystemMessage(`Upload failed: ${data.detail || 'Unknown error'}`, 'error');
            return;
        }

        // Store pending image
        _pendingImage = {
            url: data.url,
            path: data.path,
            filename: data.filename,
        };

        // Show in user message with thumbnail
        addMessage('user',
            `<div class="chat-image-pending">
                <img src="${data.url}" alt="${data.filename}">
                <span>📎 ${data.filename} (${data.size_kb} KB) — attached to next message</span>
            </div>`
        );

        // Focus chat input so user can type caption/instruction
        document.getElementById('chatInput').focus();
        document.getElementById('chatInput').placeholder = 'Describe what to do with this image, or just send...';

        addSystemMessage(`Image attached. Type your message or press Enter to send.`);

    } catch (e) {
        addSystemMessage(`Upload failed: ${e.message}`, 'error');
    }
}

// ── Image Library ────────────────────────────────────────────────

let _libraryImages = [];
let _libraryFilter = 'all';

function toggleImageLibrary() {
    const panel = document.getElementById('imageLibraryPanel');
    const isOpen = panel.classList.contains('active');
    closeAllPanels();
    if (!isOpen) {
        panel.classList.add('active');
        document.getElementById('panelOverlay').classList.add('active');
        loadImageLibrary();
    }
}

async function loadImageLibrary() {
    try {
        const res = await fetch('/api/image-library');
        const data = await res.json();
        _libraryImages = data.images || [];

        document.getElementById('imageLibraryCount').textContent =
            `${data.total} images`;

        renderLibraryGrid();
    } catch (e) {
        console.error('Failed to load image library:', e);
    }
}

function filterLibrary(filter, btn) {
    _libraryFilter = filter;
    // Update active state
    document.querySelectorAll('.filter-pill').forEach(p => p.classList.remove('active'));
    if (btn) btn.classList.add('active');
    renderLibraryGrid();
}

function renderLibraryGrid() {
    const grid = document.getElementById('imageLibraryGrid');
    const filtered = _libraryFilter === 'all'
        ? _libraryImages
        : _libraryImages.filter(img => img.source === _libraryFilter);

    if (filtered.length === 0) {
        grid.innerHTML = '<div class="image-library-empty">No images yet. Generate or upload some!</div>';
        return;
    }

    grid.innerHTML = filtered.map(img => {
        const badge = img.source === 'generated' ? '🎨' : '📎';
        return `
            <div class="image-library-card" onclick="viewFullImage('${img.url}')">
                <img src="${img.url}" alt="${img.filename}" loading="lazy">
                <span class="image-library-card-badge">${badge}</span>
                <div class="image-library-card-overlay">
                    <button onclick="event.stopPropagation(); insertImageToChat('${img.url}', '${img.path}', '${img.filename}')" title="Use in chat">💬</button>
                    <button onclick="event.stopPropagation(); downloadLibraryImage('${img.url}', '${img.filename}')" title="Download">⬇️</button>
                    <button onclick="event.stopPropagation(); deleteLibraryImage('${img.filename}')" title="Delete">🗑️</button>
                </div>
            </div>`;
    }).join('');
}

function insertImageToChat(url, path, filename) {
    _pendingImage = { url, path, filename };
    addMessage('user',
        `<div class="chat-image-pending">
            <img src="${url}" alt="${filename}">
            <span>📎 ${filename} — attached to next message</span>
        </div>`
    );
    document.getElementById('chatInput').focus();
    document.getElementById('chatInput').placeholder = 'Describe what to do with this image...';
    // Close panel
    document.getElementById('imageLibraryPanel').classList.remove('active');
    document.getElementById('panelOverlay').classList.remove('active');
}

function downloadLibraryImage(url, filename) {
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
}

async function deleteLibraryImage(filename) {
    if (!confirm(`Delete ${filename}?`)) return;
    try {
        const res = await fetch(`/api/image-library/${encodeURIComponent(filename)}`, { method: 'DELETE' });
        const data = await res.json();
        if (res.ok) {
            addSystemMessage(`Deleted ${filename} (${data.freed_kb} KB freed)`);
            loadImageLibrary();
        } else {
            addSystemMessage(data.detail || 'Delete failed', 'error');
        }
    } catch (e) {
        addSystemMessage(`Delete failed: ${e.message}`, 'error');
    }
}

// ── Fullsize Image Modal ─────────────────────────────────────────

function viewFullImage(url) {
    const modal = document.getElementById('imageModal');
    const img = document.getElementById('imageModalImg');
    img.src = url;
    modal.classList.add('active');
}

function closeImageModal() {
    document.getElementById('imageModal').classList.remove('active');
}

// Escape key to close modal
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeImageModal();
    }
});

// ── Style Presets ────────────────────────────────────────────────

let _activeStylePreset = localStorage.getItem('birdsnest_style_preset') || 'none';
let _styleIntensity = parseInt(localStorage.getItem('birdsnest_style_intensity') || '2');

function setStylePreset(style, btn) {
    _activeStylePreset = style;
    localStorage.setItem('birdsnest_style_preset', style);
    // Update chip UI
    document.querySelectorAll('.style-chip').forEach(c => c.classList.remove('active'));
    if (btn) btn.classList.add('active');
    saveStyleSettings();
}

function saveStyleSettings() {
    _styleIntensity = parseInt(document.getElementById('styleIntensity')?.value || '2');
    localStorage.setItem('birdsnest_style_intensity', _styleIntensity);
    // Persist to workspace so tools.py can read it
    fetch('/api/image-settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            quantize: localStorage.getItem('birdsnest_img_quant') || '8',
            low_ram: localStorage.getItem('birdsnest_img_lowram') === 'true',
            style_preset: _activeStylePreset,
            style_intensity: _styleIntensity,
        }),
    }).catch(() => { });
}

function restoreStyleSettings() {
    // Restore chip active state
    const chips = document.querySelectorAll('.style-chip');
    chips.forEach(c => {
        c.classList.toggle('active', c.dataset.style === _activeStylePreset);
    });
    // Restore intensity slider
    const slider = document.getElementById('styleIntensity');
    if (slider) {
        slider.value = _styleIntensity;
        const label = document.getElementById('styleIntensityLabel');
        if (label) label.textContent = ['Subtle', 'Normal', 'Strong'][_styleIntensity - 1];
    }
}

// Call restore when image tab loads
const _origLoadImageModels = loadImageModels;
loadImageModels = async function () {
    await _origLoadImageModels();
    restoreStyleSettings();
};
