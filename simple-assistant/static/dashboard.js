/**
 * Voice Assistant Dashboard JavaScript
 * Handles all UI interactions, tab switching, and API communication
 */

// Global state
let statusInterval;
let logsInterval;
let startTime = new Date();

// Logs management state
let currentSelectedContainer = 'ollama';
let autoRefreshEnabled = true;

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
});

function initializeDashboard() {
    refreshStatus();
    updateTime();
    
    // Auto-refresh every 5 seconds
    statusInterval = setInterval(refreshStatus, 5000);
    setInterval(updateTime, 1000);
    
    // Initialize logs view
    initializeLogsView();
}

// ============= TIME AND STATUS FUNCTIONS =============

function updateTime() {
    const now = new Date();
    const currentTimeEl = document.getElementById('currentTime');
    const uptimeEl = document.getElementById('uptime');
    
    if (currentTimeEl) {
        currentTimeEl.textContent = now.toLocaleTimeString();
    }
    
    if (uptimeEl) {
        const uptime = Math.floor((now - startTime) / 1000);
        const hours = Math.floor(uptime / 3600);
        const minutes = Math.floor((uptime % 3600) / 60);
        const seconds = uptime % 60;
        uptimeEl.textContent = 
            `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    }
}

async function refreshStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        updateStatusDisplay(data);
    } catch (error) {
        console.error('Failed to refresh status:', error);
    }
}

function updateStatusDisplay(data) {
    // Update Ollama status
    const ollamaStatusEl = document.getElementById('ollamaStatus');
    const ollamaStatus = data.ollama_status;
    if (ollamaStatusEl) {
        ollamaStatusEl.textContent = ollamaStatus === 'connected' ? 'ONLINE' : 'OFFLINE';
        ollamaStatusEl.className = ollamaStatus === 'connected' ? 
            'text-lg mt-1 terminal-text status-green' : 
            'text-lg mt-1 terminal-text status-red';
    }
    
    // Update model count
    const modelCountEl = document.getElementById('modelCount');
    if (modelCountEl) {
        modelCountEl.textContent = data.model_count || 0;
    }
    
    // Update audio devices
    const audioDevicesEl = document.getElementById('audioDevices');
    if (audioDevicesEl) {
        audioDevicesEl.textContent = data.audio_devices || 0;
    }
    
    // Update assistant status
    const assistantStatusEl = document.getElementById('assistantStatus');
    if (assistantStatusEl) {
        if (data.assistant_running) {
            assistantStatusEl.textContent = 'ACTIVE';
            assistantStatusEl.className = 'text-lg mt-1 terminal-text status-green';
            updateAssistantControlStatus(true);
        } else {
            assistantStatusEl.textContent = 'INACTIVE';
            assistantStatusEl.className = 'text-lg mt-1 terminal-text status-red';
            updateAssistantControlStatus(false);
        }
    }
    
    // Update container status
    if (data.containers) {
        updateContainerStatus(data.containers);
    }
}

function updateContainerStatus(containers) {
    Object.keys(containers).forEach(service => {
        const statusEl = document.getElementById(service + 'Container');
        if (statusEl && containers[service]) {
            const status = containers[service];
            const running = status.running;
            const text = running ? 'ACTIVE' : 'INACTIVE';
            statusEl.textContent = text;
            statusEl.className = `terminal-text font-medium ${running ? 'status-green' : 'status-red'}`;
            
            // Update parent container styling
            const parentDiv = statusEl.closest('.flex');
            if (parentDiv) {
                parentDiv.className = `flex justify-between items-center p-3 hacker-box ${running ? 'border-l-4 border-green-400' : 'border-l-4 border-red-400'}`;
            }
        }
    });
}

function updateAssistantControlStatus(running) {
    const statusEl = document.getElementById('assistantControlStatus');
    if (!statusEl) return;
    
    const indicator = statusEl.querySelector('.w-3');
    const text = statusEl.querySelector('span:last-child');
    
    if (indicator && text) {
        if (running) {
            indicator.className = 'inline-block w-3 h-3 bg-green-500 mr-3 blink';
            text.textContent = 'ASSISTANT: ACTIVE';
        } else {
            indicator.className = 'inline-block w-3 h-3 bg-red-500 mr-3 blink';
            text.textContent = 'ASSISTANT: INACTIVE';
        }
    }
}

// ============= ASSISTANT CONTROL FUNCTIONS =============

async function startAssistant() {
    try {
        setButtonLoading('startAssistantBtn', true);
        const response = await fetch('/api/assistant/start', { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            showAlert('success', 'ASSISTANT_STARTED');
            refreshStatus();
        } else {
            showAlert('error', 'START_FAILED');
        }
    } catch (error) {
        showAlert('error', 'START_ERROR: ' + error.message);
    } finally {
        setButtonLoading('startAssistantBtn', false);
    }
}

async function stopAssistant() {
    try {
        setButtonLoading('stopAssistantBtn', true);
        const response = await fetch('/api/assistant/stop', { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            showAlert('success', 'ASSISTANT_STOPPED');
            refreshStatus();
        } else {
            showAlert('error', 'STOP_FAILED');
        }
    } catch (error) {
        showAlert('error', 'STOP_ERROR: ' + error.message);
    } finally {
        setButtonLoading('stopAssistantBtn', false);
    }
}

// ============= AUDIO TEST FUNCTIONS =============

async function testTTS() {
    const button = event.target;
    try {
        setButtonLoading(button, true, 'SPEAKING...');
        
        const response = await fetch('/api/test/tts', { method: 'POST' });
        const data = await response.json();
        
        showTestResult('ttsTestResult', data);
    } catch (error) {
        showTestResult('ttsTestResult', { success: false, error: error.message });
    } finally {
        setButtonLoading(button, false, 'TEST_TTS (WHISPER)');
    }
}

// Whisper Service Management Functions
async function startWhisper() {
    try {
        const response = await fetch('/api/whisper/start', { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            showAlert('success', 'WHISPER_STARTED');
            refreshStatus();
        } else {
            showAlert('error', 'WHISPER_START_FAILED: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        showAlert('error', 'WHISPER_START_ERROR: ' + error.message);
    }
}

async function stopWhisper() {
    try {
        const response = await fetch('/api/whisper/stop', { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            showAlert('success', 'WHISPER_STOPPED');
            refreshStatus();
        } else {
            showAlert('error', 'WHISPER_STOP_FAILED: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        showAlert('error', 'WHISPER_STOP_ERROR: ' + error.message);
    }
}

async function toggleWhisper() {
    try {
        const button = document.getElementById('whisperEnableBtn');
        const isEnabled = button.textContent.includes('AUTO: ON');
        
        const endpoint = isEnabled ? '/api/whisper/disable' : '/api/whisper/enable';
        const response = await fetch(endpoint, { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            showAlert('success', isEnabled ? 'WHISPER_DISABLED' : 'WHISPER_ENABLED');
            refreshStatus();
        } else {
            showAlert('error', 'WHISPER_TOGGLE_FAILED: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        showAlert('error', 'WHISPER_TOGGLE_ERROR: ' + error.message);
    }
}

async function testSpeaker() {
    const button = event.target;
    try {
        setButtonLoading(button, true, 'PLAYING...');
        
        const response = await fetch('/api/test/speaker', { method: 'POST' });
        const data = await response.json();
        
        showTestResult('speakerResult', data);
    } catch (error) {
        showTestResult('speakerResult', { success: false, error: error.message });
    } finally {
        setButtonLoading(button, false, 'TEST_SPEAKER');
    }
}

async function testOllama() {
    const button = event.target;
    try {
        setButtonLoading(button, true, 'TESTING...');
        
        const response = await fetch('/api/test/ollama', { method: 'POST' });
        const data = await response.json();
        
        showTestResult('ollamaTestResult', data);
    } catch (error) {
        showTestResult('ollamaTestResult', { success: false, error: error.message });
    } finally {
        setButtonLoading(button, false, 'TEST_OLLAMA');
    }
}

// ============= TAB SWITCHING FUNCTIONS =============

function switchTab(tabName) {
    console.log('Switching to tab:', tabName);
    
    // Hide all views
    const dashboardView = document.getElementById('dashboardView');
    const logsView = document.getElementById('logsView');
    
    if (dashboardView) dashboardView.classList.add('hidden');
    if (logsView) logsView.classList.add('hidden');
    
    // Show selected view
    if (tabName === 'dashboard') {
        if (dashboardView) dashboardView.classList.remove('hidden');
    } else if (tabName === 'logs') {
        if (logsView) {
            logsView.classList.remove('hidden');
            refreshCurrentLogs(); // Refresh logs when switching to logs tab
        }
    }
    
    // Update tab button styling
    updateTabButtonStates(tabName);
}

function updateTabButtonStates(activeTab) {
    const dashboardTab = document.getElementById('dashboardTab');
    const logsTab = document.getElementById('logsTab');
    
    if (dashboardTab && logsTab) {
        if (activeTab === 'dashboard') {
            dashboardTab.className = 'px-6 py-2 border border-green-400 text-green-400 bg-green-400 bg-opacity-20 terminal-text hover:bg-opacity-30';
            logsTab.className = 'px-6 py-2 border border-gray-400 text-gray-400 terminal-text hover:bg-gray-400 hover:bg-opacity-20';
        } else if (activeTab === 'logs') {
            dashboardTab.className = 'px-6 py-2 border border-gray-400 text-gray-400 terminal-text hover:bg-gray-400 hover:bg-opacity-20';
            logsTab.className = 'px-6 py-2 border border-green-400 text-green-400 bg-green-400 bg-opacity-20 terminal-text hover:bg-opacity-30';
        }
    }
}

// ============= LOGS MANAGEMENT FUNCTIONS =============

function selectContainer(containerName) {
    console.log('Selecting container:', containerName);
    currentSelectedContainer = containerName;
    
    // Update container button styling
    const buttons = ['ollama', 'whisper', 'piper', 'assistant'];
    buttons.forEach(btn => {
        const button = document.getElementById(`container-${btn}`);
        if (button) {
            if (btn === containerName) {
                button.className = 'w-full text-left px-4 py-3 border border-green-400 bg-green-400 bg-opacity-20 text-green-400 terminal-text hover:bg-opacity-30';
            } else {
                button.className = 'w-full text-left px-4 py-3 border border-gray-600 text-gray-400 terminal-text hover:border-green-400 hover:text-green-400';
            }
        }
    });
    
    // Update container title
    const titleMap = {
        'ollama': 'OLLAMA',
        'whisper': 'WHISPER_STT',
        'piper': 'PIPER_TTS',
        'assistant': 'VOICE_ASSISTANT'
    };
    const titleEl = document.getElementById('currentContainerTitle');
    if (titleEl) {
        titleEl.textContent = titleMap[containerName] || containerName.toUpperCase();
    }
    
    // Refresh logs for selected container
    refreshCurrentLogs();
}

async function refreshLogs() {
    try {
        await refreshCurrentLogs();
    } catch (error) {
        console.error('Failed to refresh logs:', error);
    }
}

async function refreshCurrentLogs() {
    console.log('Refreshing logs for container:', currentSelectedContainer);
    
    try {
        const lines = document.getElementById('logLinesSelect')?.value || 50;
        const url = `/api/logs?service=${currentSelectedContainer}&lines=${lines}`;
        console.log('Fetching logs from:', url);
        
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const logsText = await response.text();
        displayLogs(logsText);
        
        // Update last update time
        const lastUpdateEl = document.getElementById('lastLogUpdate');
        if (lastUpdateEl) {
            lastUpdateEl.textContent = new Date().toLocaleTimeString();
        }
        
    } catch (error) {
        console.error('Error refreshing logs:', error);
        displayLogs(`[ERROR] Failed to fetch logs: ${error.message}`);
    }
}

function displayLogs(logsText) {
    const logsDisplay = document.getElementById('logsDisplay');
    if (!logsDisplay) return;
    
    // Clear existing logs
    logsDisplay.innerHTML = '';
    
    // Split logs into lines and process them
    const lines = logsText.split('\n').filter(line => line.trim());
    
    if (lines.length === 0) {
        logsDisplay.innerHTML = '<div class="text-yellow-400">[SYSTEM] No logs available for the selected container.</div>';
        return;
    }
    
    // Add logs with proper formatting
    lines.forEach(line => {
        const logEntry = document.createElement('div');
        logEntry.className = 'terminal-text';
        
        // Color code logs based on content
        if (line.toLowerCase().includes('error') || line.toLowerCase().includes('exception')) {
            logEntry.className = 'terminal-text error-text';
        } else if (line.toLowerCase().includes('warning') || line.toLowerCase().includes('warn')) {
            logEntry.className = 'terminal-text warning-text';
        } else {
            logEntry.className = 'terminal-text';
        }
        
        logEntry.textContent = line;
        logsDisplay.appendChild(logEntry);
    });
    
    // Scroll to bottom
    const logsContainer = document.getElementById('logsDisplayContainer');
    if (logsContainer) {
        logsContainer.scrollTop = logsContainer.scrollHeight;
    }
}

function clearLogsDisplay() {
    const logsDisplay = document.getElementById('logsDisplay');
    if (logsDisplay) {
        logsDisplay.innerHTML = '<div class="text-green-400">[SYSTEM] Logs cleared. Click REFRESH to reload.</div>';
    }
}

function toggleAutoRefresh() {
    autoRefreshEnabled = !autoRefreshEnabled;
    const autoRefreshBtn = document.getElementById('autoRefreshBtn');
    
    if (autoRefreshBtn) {
        if (autoRefreshEnabled) {
            autoRefreshBtn.innerHTML = 'AUTO: ON';
            autoRefreshBtn.className = 'w-full bg-transparent border border-green-400 text-green-400 px-4 py-2 terminal-text text-sm hover:bg-green-400 hover:text-black';
        } else {
            autoRefreshBtn.innerHTML = 'AUTO: OFF';
            autoRefreshBtn.className = 'w-full bg-transparent border border-red-400 text-red-400 px-4 py-2 terminal-text text-sm hover:bg-red-400 hover:text-black';
        }
    }
}

// Initialize logs view
function initializeLogsView() {
    selectContainer('ollama');
}

// ============= UTILITY FUNCTIONS =============

function setButtonLoading(button, loading, loadingText = 'LOADING...') {
    if (typeof button === 'string') {
        button = document.querySelector(`[onclick="${button}"]`);
    }
    
    if (button) {
        button.disabled = loading;
        if (loading) {
            button.dataset.originalText = button.textContent;
            button.innerHTML = `<span class="loading"></span> ${loadingText}`;
        } else {
            button.innerHTML = button.dataset.originalText || button.textContent;
        }
    }
}

function showTestResult(elementId, result) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    element.classList.remove('hidden');
    element.className = `mt-4 p-4 hacker-box terminal-text ${result.success ? 'border-l-4 border-green-400' : 'border-l-4 border-red-400'}`;
    
    if (result.success) {
        let output = `
            <div class="text-green-400 font-bold mb-2">✓ TEST_PASSED</div>
            ${result.message ? `<div class="text-gray-300 mb-2">${escapeHtml(result.message)}</div>` : ''}
            ${result.execution_time ? `<div class="text-yellow-400 mb-2">EXEC_TIME: ${result.execution_time}s</div>` : ''}
        `;
        
        // AI Test Results
        if (result.test_response) {
            output += `<div class="mt-2 text-yellow-400"><strong>AI_RESPONSE:</strong><br>"${escapeHtml(result.test_response)}"</div>`;
        }
        if (result.ai_stats) {
            const stats = result.ai_stats;
            output += `<div class="mt-2 text-blue-400"><strong>AI_STATS:</strong><br>
                WORDS: ${stats.word_count} | TOKENS: ${stats.total_tokens} | TPS: ${stats.tokens_per_second}<br>
                QUALITY: ${stats.response_quality}
            </div>`;
        }
        
        // STT Test Results
        if (result.transcription) {
            output += `<div class="mt-2 text-cyan-400"><strong>TRANSCRIPTION:</strong><br>"${escapeHtml(result.transcription)}"</div>`;
        }
        if (result.transcription_stats) {
            const stats = result.transcription_stats;
            output += `<div class="mt-2 text-cyan-400"><strong>STT_STATS:</strong><br>
                WORDS: ${stats.word_count} | CONFIDENCE: ${(stats.confidence_score * 100).toFixed(1)}% | WPS: ${stats.words_per_second}
            </div>`;
        }
        
        // Audio Test Results
        if (result.audio_stats) {
            const stats = result.audio_stats;
            output += `<div class="mt-2 text-purple-400"><strong>AUDIO_STATS:</strong><br>
                DURATION: ${stats.duration_seconds}s | SAMPLES: ${stats.sample_count} | QUALITY: ${stats.quality_score || 'N/A'}%
            </div>`;
        }
        
        if (result.played_message) {
            output += `<div class="mt-2 text-orange-400"><strong>RESULT:</strong> ${escapeHtml(result.played_message)}</div>`;
        }
        
        if (result.recorded_message) {
            output += `<div class="mt-2 text-orange-400"><strong>RESULT:</strong> ${escapeHtml(result.recorded_message)}</div>`;
        }
        
        // Model info
        if (result.models) {
            output += `<div class="mt-2 text-green-400"><strong>MODELS:</strong> ${result.models.join(', ')}</div>`;
        }
        
        element.innerHTML = output;
    } else {
        element.innerHTML = `
            <div class="text-red-400 font-bold mb-2">✗ TEST_FAILED</div>
            <div class="text-red-400 mb-2">${escapeHtml(result.error)}</div>
            ${result.execution_time ? `<div class="text-yellow-400">EXEC_TIME: ${result.execution_time}s</div>` : ''}
        `;
    }
}

function showAlert(type, message) {
    const alertClass = type === 'success' ? 'border-green-400 text-green-400' : 'border-red-400 text-red-400';
    
    // Create a temporary alert element
    const alertDiv = document.createElement('div');
    alertDiv.className = `fixed top-4 right-4 border-2 ${alertClass} bg-black px-6 py-3 terminal-text z-50`;
    alertDiv.innerHTML = `[${type.toUpperCase()}] ${message}`;
    
    document.body.appendChild(alertDiv);
    
    // Remove after 3 seconds
    setTimeout(() => {
        alertDiv.style.opacity = '0';
        setTimeout(() => {
            if (document.body.contains(alertDiv)) {
                document.body.removeChild(alertDiv);
            }
        }, 300);
    }, 3000);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}