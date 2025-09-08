/**
 * Multi-Modal Input Processing Frontend
 * Handles file uploads, voice recording, and all user interactions
 */

// Global variables
let currentTab = 'file-upload';
let mediaRecorder = null;
let audioChunks = [];
let recordingStartTime = null;
let recordingInterval = null;
let capabilities = {};

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeInterface();
    loadCapabilities();
    setupEventListeners();
});

/**
 * Initialize the interface
 */
function initializeInterface() {
    // Dark mode toggle
    const darkModeToggle = document.getElementById('darkModeToggle');
    const html = document.documentElement;
    
    // Check for saved dark mode preference
    if (localStorage.getItem('darkMode') === 'true' || 
        (!localStorage.getItem('darkMode') && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
        html.classList.add('dark');
    }
    
    darkModeToggle.addEventListener('click', () => {
        html.classList.toggle('dark');
        localStorage.setItem('darkMode', html.classList.contains('dark'));
    });

    // Tab functionality
    const tabBtns = document.querySelectorAll('.input-tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');
            switchTab(targetTab);
        });
    });
}

/**
 * Switch between input method tabs
 */
function switchTab(targetTab) {
    currentTab = targetTab;
    
    // Update button styles
    const tabBtns = document.querySelectorAll('.input-tab-btn');
    tabBtns.forEach(btn => {
        if (btn.getAttribute('data-tab') === targetTab) {
            btn.className = 'input-tab-btn active px-6 py-3 mx-2 mb-2 text-lg font-medium rounded-lg bg-primary-500 text-white hover:bg-primary-600 transition-colors';
        } else {
            btn.className = 'input-tab-btn px-6 py-3 mx-2 mb-2 text-lg font-medium rounded-lg bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors';
        }
    });
    
    // Show target tab
    const tabContents = document.querySelectorAll('.tab-content');
    tabContents.forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(`${targetTab}-tab`).classList.add('active');
}

/**
 * Setup all event listeners
 */
function setupEventListeners() {
    // File upload events
    setupFileUploadEvents();
    
    // Voice input events
    setupVoiceEvents();
    
    // Text analysis form
    const textForm = document.getElementById('textAnalysisForm');
    if (textForm) {
        textForm.addEventListener('submit', handleTextAnalysis);
    }
    
    // URL analysis form
    const urlForm = document.getElementById('urlAnalysisForm');
    if (urlForm) {
        urlForm.addEventListener('submit', handleUrlAnalysis);
    }
}

/**
 * Setup file upload drag-and-drop and click events
 */
function setupFileUploadEvents() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    
    // Click to browse
    dropZone.addEventListener('click', () => {
        fileInput.click();
    });
    
    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
    
    // Drag and drop events
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });
    
    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
    });
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });
}

/**
 * Setup voice recording events
 */
function setupVoiceEvents() {
    // Audio file input
    const audioInput = document.getElementById('audioInput');
    if (audioInput) {
        audioInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleAudioFile(e.target.files[0]);
            }
        });
    }
}

/**
 * Handle file selection and preview
 */
function handleFileSelect(file) {
    const filePreview = document.getElementById('filePreview');
    const previewContent = document.getElementById('previewContent');
    const processingOptions = document.getElementById('processingOptions');
    const analyzeBtn = document.getElementById('analyzeFileBtn');
    
    // Store file reference
    window.selectedFile = file;
    
    // Show preview section
    filePreview.classList.remove('hidden');
    processingOptions.classList.remove('hidden');
    analyzeBtn.disabled = false;
    
    // Create file info display
    const fileInfo = document.createElement('div');
    fileInfo.className = 'flex items-center space-x-4';
    
    const fileIcon = getFileIcon(file.type);
    const fileSize = formatFileSize(file.size);
    
    fileInfo.innerHTML = `
        <div class="text-4xl">${fileIcon}</div>
        <div class="flex-1">
            <p class="font-medium text-gray-900 dark:text-white">${file.name}</p>
            <p class="text-sm text-gray-500 dark:text-gray-400">
                ${file.type || 'Unknown type'} • ${fileSize}
            </p>
        </div>
    `;
    
    previewContent.innerHTML = '';
    previewContent.appendChild(fileInfo);
    
    // Add image preview if it's an image
    if (file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = document.createElement('img');
            img.src = e.target.result;
            img.className = 'file-preview mt-4 rounded-lg border border-gray-200 dark:border-gray-600';
            previewContent.appendChild(img);
        };
        reader.readAsDataURL(file);
    }
}

/**
 * Handle audio file selection
 */
function handleAudioFile(file) {
    const audioPreview = document.getElementById('audioPreview');
    const audioPlayer = document.getElementById('audioPlayer');
    
    // Store file reference
    window.selectedAudioFile = file;
    
    // Create audio preview
    const url = URL.createObjectURL(file);
    audioPlayer.src = url;
    audioPreview.classList.remove('hidden');
}

/**
 * Clear selected file
 */
function clearFile() {
    const filePreview = document.getElementById('filePreview');
    const processingOptions = document.getElementById('processingOptions');
    const analyzeBtn = document.getElementById('analyzeFileBtn');
    const fileInput = document.getElementById('fileInput');
    
    filePreview.classList.add('hidden');
    processingOptions.classList.add('hidden');
    analyzeBtn.disabled = true;
    fileInput.value = '';
    window.selectedFile = null;
}

/**
 * Analyze uploaded file
 */
async function analyzeFile() {
    if (!window.selectedFile) {
        showNotification('No file selected', 'error');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', window.selectedFile);
    formData.append('enable_ocr', document.getElementById('enableOCR').checked);
    formData.append('enhanced_processing', document.getElementById('enhancedProcessing').checked);
    
    await submitMultiModalAnalysis(formData, 'file');
}

/**
 * Toggle voice recording
 */
async function toggleRecording() {
    const recordBtn = document.getElementById('recordBtn');
    const recordIcon = document.getElementById('recordIcon');
    const recordingStatus = document.getElementById('recordingStatus');
    
    if (!mediaRecorder || mediaRecorder.state === 'inactive') {
        // Start recording
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];
            
            mediaRecorder.ondataavailable = (event) => {
                audioChunks.push(event.data);
            };
            
            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                const audioUrl = URL.createObjectURL(audioBlob);
                
                // Show audio preview
                const audioPreview = document.getElementById('audioPreview');
                const audioPlayer = document.getElementById('audioPlayer');
                audioPlayer.src = audioUrl;
                audioPreview.classList.remove('hidden');
                
                // Store for analysis
                window.recordedAudio = audioBlob;
                
                // Stop all tracks
                stream.getTracks().forEach(track => track.stop());
            };
            
            mediaRecorder.start();
            recordingStartTime = Date.now();
            
            // Update UI
            recordBtn.classList.add('recording');
            recordIcon.innerHTML = `
                <svg class="w-12 h-12 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 10h6v4H9z"></path>
                </svg>
                <span class="text-sm font-medium">Stop</span>
            `;
            recordingStatus.classList.remove('hidden');
            
            // Start timer
            recordingInterval = setInterval(updateRecordingTime, 1000);
            
        } catch (error) {
            showNotification('Could not access microphone: ' + error.message, 'error');
        }
    } else {
        // Stop recording
        mediaRecorder.stop();
        clearInterval(recordingInterval);
        
        // Reset UI
        recordBtn.classList.remove('recording');
        recordIcon.innerHTML = `
            <svg class="w-12 h-12 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"></path>
            </svg>
            <span class="text-sm font-medium">Record</span>
        `;
        recordingStatus.classList.add('hidden');
    }
}

/**
 * Update recording time display
 */
function updateRecordingTime() {
    if (recordingStartTime) {
        const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        const timeString = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        document.getElementById('recordingTime').textContent = timeString;
    }
}

/**
 * Analyze recorded or uploaded audio
 */
async function analyzeAudio() {
    let audioFile = window.recordedAudio || window.selectedAudioFile;
    
    if (!audioFile) {
        showNotification('No audio to analyze', 'error');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', audioFile, 'audio.wav');
    
    await submitMultiModalAnalysis(formData, 'audio');
}

/**
 * Handle text analysis form submission
 */
async function handleTextAnalysis(e) {
    e.preventDefault();
    
    const textInput = document.getElementById('textInput');
    const contentType = document.getElementById('contentType');
    
    const formData = new FormData();
    formData.append('text', textInput.value);
    formData.append('prediction_type', contentType.value);
    
    await submitMultiModalAnalysis(formData, 'text');
}

/**
 * Handle URL analysis form submission
 */
async function handleUrlAnalysis(e) {
    e.preventDefault();
    
    const urlInput = document.getElementById('urlInputField');
    const captureScreenshot = document.getElementById('captureScreenshot');
    const ocrScreenshot = document.getElementById('ocrScreenshot');
    
    const formData = new FormData();
    formData.append('url', urlInput.value);
    formData.append('capture_screenshot', captureScreenshot.checked);
    formData.append('ocr_screenshot', ocrScreenshot.checked);
    
    await submitMultiModalAnalysis(formData, 'url');
}

/**
 * Submit multi-modal analysis request
 */
async function submitMultiModalAnalysis(formData, inputType) {
    showLoading(true, `Analyzing ${inputType}...`, 'Processing your input using advanced AI models');
    
    try {
        const response = await fetch('/predict/multimodal/', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        showResults(data, inputType);
        
    } catch (error) {
        showNotification('Network error: ' + error.message, 'error');
    } finally {
        showLoading(false);
    }
}

/**
 * Start batch processing
 */
async function startBatchProcessing() {
    const fileInput = document.getElementById('batchFileInput');
    const urlList = document.getElementById('urlList');
    const captureScreenshots = document.getElementById('batchScreenshots');
    
    const files = Array.from(fileInput.files);
    const urls = urlList.value.split('\n').filter(url => url.trim()).map(url => url.trim());
    
    if (files.length === 0 && urls.length === 0) {
        showNotification('Please select files or enter URLs for batch processing', 'error');
        return;
    }
    
    const formData = new FormData();
    
    // Add files
    files.forEach(file => {
        formData.append('files', file);
    });
    
    // Add URLs and options
    formData.append('urls', JSON.stringify(urls));
    formData.append('capture_screenshots', captureScreenshots.checked);
    
    showBatchProgress(true);
    
    try {
        const response = await fetch('/predict/batch/', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        showBatchResults(data);
        
    } catch (error) {
        showNotification('Batch processing error: ' + error.message, 'error');
    } finally {
        showBatchProgress(false);
    }
}

/**
 * Show/hide loading indicator
 */
function showLoading(show, title = 'Processing...', subtitle = '') {
    const loading = document.getElementById('loadingIndicator');
    const loadingText = document.getElementById('loadingText');
    const loadingSubtext = document.getElementById('loadingSubtext');
    
    if (show) {
        loadingText.textContent = title;
        loadingSubtext.textContent = subtitle;
        loading.style.display = 'flex';
    } else {
        loading.style.display = 'none';
    }
}

/**
 * Show/hide batch progress
 */
function showBatchProgress(show) {
    const progress = document.getElementById('batchProgress');
    
    if (show) {
        progress.classList.remove('hidden');
        updateBatchProgress(0, 0);
    } else {
        progress.classList.add('hidden');
    }
}

/**
 * Update batch progress
 */
function updateBatchProgress(completed, total) {
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    
    const percentage = total > 0 ? (completed / total) * 100 : 0;
    progressBar.style.width = `${percentage}%`;
    progressText.textContent = `${completed}/${total}`;
}

/**
 * Display analysis results
 */
function showResults(data, inputType) {
    const resultsContainer = document.getElementById('resultsContainer');
    const resultsContent = document.getElementById('resultsContent');
    
    if (!data.success) {
        resultsContent.innerHTML = createErrorResult(data.error, data.processing_info);
        resultsContainer.classList.remove('hidden');
        return;
    }
    
    const result = data.result;
    const inputProcessing = data.input_processing;
    const enhancedAnalysis = data.enhanced_analysis;
    
    // Determine result styling
    const isThreats = ['PHISHING', 'SPAM', 'MALICIOUS'].includes(result.prediction);
    const resultColor = isThreats ? 'red' : 'green';
    const resultIcon = isThreats ? '⚠️' : '✅';
    const resultBg = isThreats ? 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800' : 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800';
    
    resultsContent.innerHTML = `
        <div class="result-card ${resultBg} border rounded-lg p-6 mb-6">
            <div class="flex items-center justify-center mb-4">
                <span class="text-4xl mr-3">${resultIcon}</span>
                <h4 class="text-2xl font-bold text-${resultColor}-600 dark:text-${resultColor}-400">${result.prediction}</h4>
            </div>
            <div class="text-center space-y-3">
                <div>
                    <p class="text-gray-600 dark:text-gray-300 mb-2">
                        Confidence: <span class="font-semibold text-lg">${result.confidence_score.toFixed(1)}%</span>
                        <span class="text-sm text-gray-500 ml-2">(${result.confidence_level})</span>
                    </p>
                    <div class="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-3">
                        <div class="bg-blue-500 h-3 rounded-full transition-all duration-500" style="width: ${result.confidence_score}%"></div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Input Processing Details -->
        <div class="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 mb-6">
            <h5 class="font-semibold text-gray-900 dark:text-white mb-3">📋 Processing Details</h5>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                <div>
                    <span class="font-medium text-gray-700 dark:text-gray-300">Input Type:</span>
                    <span class="text-gray-600 dark:text-gray-400 ml-2">${inputProcessing.input_type}</span>
                </div>
                <div>
                    <span class="font-medium text-gray-700 dark:text-gray-300">Prediction Type:</span>
                    <span class="text-gray-600 dark:text-gray-400 ml-2">${result.prediction_type}</span>
                </div>
                <div>
                    <span class="font-medium text-gray-700 dark:text-gray-300">Text Extracted:</span>
                    <span class="text-gray-600 dark:text-gray-400 ml-2">${inputProcessing.extracted_text_length} characters</span>
                </div>
                <div>
                    <span class="font-medium text-gray-700 dark:text-gray-300">Analysis Method:</span>
                    <span class="text-gray-600 dark:text-gray-400 ml-2">${enhancedAnalysis.analysis_method}</span>
                </div>
            </div>
            
            <!-- Processing Steps -->
            <div class="mt-4">
                <span class="font-medium text-gray-700 dark:text-gray-300">Processing Steps:</span>
                <ul class="list-disc list-inside text-sm text-gray-600 dark:text-gray-400 mt-1">
                    ${inputProcessing.processing_steps.map(step => `<li>${step}</li>`).join('')}
                </ul>
            </div>
        </div>
        
        <!-- Enhanced Analysis -->
        <div class="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4 mb-6">
            <h5 class="font-semibold text-gray-900 dark:text-white mb-3">🧠 Enhanced Analysis</h5>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                <div>
                    <span class="font-medium text-gray-700 dark:text-gray-300">Final Recommendation:</span>
                    <span class="text-gray-600 dark:text-gray-400 ml-2">${enhancedAnalysis.final_recommendation}</span>
                </div>
                <div>
                    <span class="font-medium text-gray-700 dark:text-gray-300">Threat Detected:</span>
                    <span class="text-gray-600 dark:text-gray-400 ml-2">${enhancedAnalysis.threat_detected ? 'Yes' : 'No'}</span>
                </div>
            </div>
            
            ${enhancedAnalysis.llm_summary ? `
                <div class="mt-4 p-3 bg-white dark:bg-gray-700 rounded border">
                    <h6 class="font-medium text-gray-900 dark:text-white mb-2">🤖 LLM Analysis</h6>
                    <div class="text-sm space-y-1">
                        <div>
                            <span class="font-medium text-gray-700 dark:text-gray-300">Threat Level:</span>
                            <span class="text-gray-600 dark:text-gray-400 ml-2">${enhancedAnalysis.llm_summary.threat_level}</span>
                        </div>
                        <div>
                            <span class="font-medium text-gray-700 dark:text-gray-300">LLM Confidence:</span>
                            <span class="text-gray-600 dark:text-gray-400 ml-2">${enhancedAnalysis.llm_summary.confidence}%</span>
                        </div>
                        ${enhancedAnalysis.llm_summary.analysis_summary ? `
                            <div class="mt-2">
                                <span class="font-medium text-gray-700 dark:text-gray-300">Summary:</span>
                                <p class="text-gray-600 dark:text-gray-400 mt-1">${enhancedAnalysis.llm_summary.analysis_summary}</p>
                            </div>
                        ` : ''}
                    </div>
                </div>
            ` : ''}
        </div>
        
        <!-- Extracted Text Preview -->
        ${data.extracted_text_preview ? `
            <div class="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <h5 class="font-semibold text-gray-900 dark:text-white mb-3">📄 Extracted Text Preview</h5>
                <div class="bg-white dark:bg-gray-800 border rounded p-3 text-sm font-mono text-gray-700 dark:text-gray-300 max-h-40 overflow-y-auto">
                    ${escapeHtml(data.extracted_text_preview)}
                </div>
            </div>
        ` : ''}
    `;
    
    resultsContainer.classList.remove('hidden');
    
    // Animate result card
    setTimeout(() => {
        const resultCard = resultsContent.querySelector('.result-card');
        if (resultCard) {
            resultCard.classList.add('show');
        }
    }, 100);
}

/**
 * Create error result display
 */
function createErrorResult(error, processingInfo) {
    return `
        <div class="result-card bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-6">
            <div class="flex items-center justify-center mb-4">
                <span class="text-4xl mr-3">❌</span>
                <h4 class="text-2xl font-bold text-red-600 dark:text-red-400">Analysis Failed</h4>
            </div>
            <div class="text-center">
                <p class="text-gray-600 dark:text-gray-300 mb-4">${error}</p>
                ${processingInfo ? `
                    <div class="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm text-left">
                        <p><strong>Input Type:</strong> ${processingInfo.input_type || 'Unknown'}</p>
                        ${processingInfo.extracted_text_length !== undefined ? `
                            <p><strong>Text Length:</strong> ${processingInfo.extracted_text_length} characters</p>
                        ` : ''}
                        ${processingInfo.processing_steps ? `
                            <p><strong>Processing Steps:</strong></p>
                            <ul class="list-disc list-inside ml-4">
                                ${processingInfo.processing_steps.map(step => `<li>${step}</li>`).join('')}
                            </ul>
                        ` : ''}
                    </div>
                ` : ''}
            </div>
        </div>
    `;
}

/**
 * Show batch processing results
 */
function showBatchResults(data) {
    const resultsContainer = document.getElementById('resultsContainer');
    const resultsContent = document.getElementById('resultsContent');
    
    if (!data.success) {
        resultsContent.innerHTML = createErrorResult(data.error);
        resultsContainer.classList.remove('hidden');
        return;
    }
    
    const results = data.batch_results;
    const successful = data.successful_items;
    const failed = data.failed_items;
    
    resultsContent.innerHTML = `
        <div class="mb-6">
            <h4 class="text-xl font-bold text-gray-900 dark:text-white mb-4">Batch Processing Results</h4>
            <div class="grid grid-cols-3 gap-4 text-center">
                <div class="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                    <div class="text-2xl font-bold text-blue-600 dark:text-blue-400">${data.total_items}</div>
                    <div class="text-sm text-gray-600 dark:text-gray-400">Total Items</div>
                </div>
                <div class="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
                    <div class="text-2xl font-bold text-green-600 dark:text-green-400">${successful}</div>
                    <div class="text-sm text-gray-600 dark:text-gray-400">Successful</div>
                </div>
                <div class="bg-red-50 dark:bg-red-900/20 rounded-lg p-4">
                    <div class="text-2xl font-bold text-red-600 dark:text-red-400">${failed}</div>
                    <div class="text-sm text-gray-600 dark:text-gray-400">Failed</div>
                </div>
            </div>
        </div>
        
        <div class="space-y-3">
            ${results.map(result => `
                <div class="border border-gray-200 dark:border-gray-600 rounded-lg p-4 ${result.success ? 'bg-green-50 dark:bg-green-900/20' : 'bg-red-50 dark:bg-red-900/20'}">
                    <div class="flex items-center justify-between">
                        <div class="flex-1">
                            <h5 class="font-medium text-gray-900 dark:text-white">
                                ${result.filename || result.url || `Item ${result.index + 1}`}
                            </h5>
                            <p class="text-sm text-gray-600 dark:text-gray-400">
                                Type: ${result.input_type} • Prediction: ${result.prediction_type}
                                ${result.text_length ? ` • Text: ${result.text_length} chars` : ''}
                            </p>
                            ${result.preview ? `
                                <p class="text-xs text-gray-500 dark:text-gray-400 mt-1 truncate">
                                    Preview: ${result.preview.substring(0, 100)}...
                                </p>
                            ` : ''}
                        </div>
                        <div class="text-2xl">
                            ${result.success ? '✅' : '❌'}
                        </div>
                    </div>
                    ${result.error ? `
                        <div class="mt-2 text-sm text-red-600 dark:text-red-400">
                            Error: ${result.error}
                        </div>
                    ` : ''}
                </div>
            `).join('')}
        </div>
    `;
    
    resultsContainer.classList.remove('hidden');
}

/**
 * Load system capabilities
 */
async function loadCapabilities() {
    try {
        const response = await fetch('/capabilities/');
        const data = await response.json();
        capabilities = data;
        updateCapabilitiesStatus(data);
    } catch (error) {
        console.error('Failed to load capabilities:', error);
        updateCapabilitiesStatus(null);
    }
}

/**
 * Update capabilities status display
 */
function updateCapabilitiesStatus(data) {
    const statusElement = document.getElementById('capabilitiesStatus');
    
    if (!data) {
        statusElement.innerHTML = `
            <span class="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200">
                ❌ Capabilities unavailable
            </span>
        `;
        return;
    }
    
    const multimodalAvailable = data.multimodal_available;
    const enhancedAvailable = data.enhanced_service_available;
    
    if (multimodalAvailable && enhancedAvailable) {
        statusElement.innerHTML = `
            <span class="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
                🧠 Multi-Modal AI Active
            </span>
        `;
    } else if (enhancedAvailable) {
        statusElement.innerHTML = `
            <span class="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200">
                📊 Basic AI Active
            </span>
        `;
    } else {
        statusElement.innerHTML = `
            <span class="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200">
                ❌ AI Services Offline
            </span>
        `;
    }
}

/**
 * Show capabilities modal
 */
function showCapabilities() {
    const modal = document.getElementById('capabilitiesModal');
    const content = document.getElementById('capabilitiesContent');
    
    if (!capabilities || Object.keys(capabilities).length === 0) {
        content.innerHTML = '<p class="text-gray-600 dark:text-gray-400">Capabilities information not available.</p>';
    } else {
        content.innerHTML = createCapabilitiesDisplay(capabilities);
    }
    
    modal.classList.remove('hidden');
    modal.classList.add('flex');
}

/**
 * Hide capabilities modal
 */
function hideCapabilities() {
    const modal = document.getElementById('capabilitiesModal');
    modal.classList.add('hidden');
    modal.classList.remove('flex');
}

/**
 * Create capabilities display HTML
 */
function createCapabilitiesDisplay(data) {
    const caps = data.capabilities || {};
    const formats = data.supported_formats || {};
    
    return `
        <div class="space-y-6">
            <!-- Service Status -->
            <div>
                <h4 class="font-semibold text-gray-900 dark:text-white mb-3">Service Status</h4>
                <div class="grid grid-cols-2 gap-3">
                    <div class="flex items-center space-x-2">
                        <span class="${data.multimodal_available ? 'text-green-600' : 'text-red-600'}">${data.multimodal_available ? '✅' : '❌'}</span>
                        <span class="text-sm text-gray-700 dark:text-gray-300">Multi-Modal Processing</span>
                    </div>
                    <div class="flex items-center space-x-2">
                        <span class="${data.enhanced_service_available ? 'text-green-600' : 'text-red-600'}">${data.enhanced_service_available ? '✅' : '❌'}</span>
                        <span class="text-sm text-gray-700 dark:text-gray-300">Enhanced AI Service</span>
                    </div>
                </div>
            </div>
            
            <!-- Processing Capabilities -->
            <div>
                <h4 class="font-semibold text-gray-900 dark:text-white mb-3">Processing Capabilities</h4>
                <div class="grid grid-cols-2 gap-3">
                    ${Object.entries(caps).map(([key, value]) => `
                        <div class="flex items-center space-x-2">
                            <span class="${value ? 'text-green-600' : 'text-red-600'}">${value ? '✅' : '❌'}</span>
                            <span class="text-sm text-gray-700 dark:text-gray-300">${formatCapabilityName(key)}</span>
                        </div>
                    `).join('')}
                </div>
            </div>
            
            <!-- Supported Formats -->
            ${Object.keys(formats).length > 0 ? `
                <div>
                    <h4 class="font-semibold text-gray-900 dark:text-white mb-3">Supported Formats</h4>
                    ${Object.entries(formats).map(([type, formatList]) => `
                        <div class="mb-3">
                            <h5 class="text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">${type.charAt(0).toUpperCase() + type.slice(1)}:</h5>
                            <div class="flex flex-wrap gap-1">
                                ${formatList.map(format => `
                                    <span class="px-2 py-1 bg-gray-100 dark:bg-gray-700 text-xs rounded">${format}</span>
                                `).join('')}
                            </div>
                        </div>
                    `).join('')}
                </div>
            ` : ''}
        </div>
    `;
}

/**
 * Utility functions
 */
function formatCapabilityName(key) {
    return key.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
}

function getFileIcon(mimeType) {
    if (mimeType.startsWith('image/')) return '🖼️';
    if (mimeType.startsWith('audio/')) return '🎵';
    if (mimeType.includes('pdf')) return '📄';
    if (mimeType.includes('word')) return '📝';
    if (mimeType.includes('text')) return '📃';
    return '📁';
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `fixed top-4 right-4 px-6 py-3 rounded-lg shadow-lg z-50 ${
        type === 'error' ? 'bg-red-500 text-white' : 
        type === 'success' ? 'bg-green-500 text-white' : 
        'bg-blue-500 text-white'
    }`;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.remove();
    }, 5000);
}

// Make functions globally available
window.showCapabilities = showCapabilities;
window.hideCapabilities = hideCapabilities;
window.analyzeFile = analyzeFile;
window.clearFile = clearFile;
window.toggleRecording = toggleRecording;
window.analyzeAudio = analyzeAudio;
window.startBatchProcessing = startBatchProcessing;
