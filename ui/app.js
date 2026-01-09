/**
 * MangaColor AI - Frontend Logic
 * Handles API interactions and UI updates.
 */

const API_BASE = window.location.origin;
let currentJobId = null;
let statusPollingInterval = null;

// DOM Elements
const elements = {
    apiStatus: document.getElementById('api-status'),
    btnCreateJob: document.getElementById('btn-create-job'),
    jobInfo: document.getElementById('job-info'),
    currentJobId: document.getElementById('current-job-id'),
    step2: document.getElementById('step-2'),
    step3: document.getElementById('step-3'),
    inputPages: document.getElementById('input-pages'),
    inputRefs: document.getElementById('input-refs'),
    countPages: document.getElementById('count-pages'),
    countRefs: document.getElementById('count-refs'),
    btnUploadAll: document.getElementById('btn-upload-all'),
    footerUpload: document.getElementById('footer-upload'),
    btnRunJob: document.getElementById('btn-run-job'),
    statusDisplay: document.getElementById('status-display'),
    jobStatusText: document.getElementById('job-status-text'),
    jobProgressPercent: document.getElementById('job-progress-percent'),
    progressFill: document.getElementById('progress-fill'),
    pagesCompleted: document.getElementById('pages-completed'),
    pagesFailed: document.getElementById('pages-failed'),
    resultsSection: document.getElementById('results-section'),
    resultsGrid: document.getElementById('results-grid'),
    btnRefreshResults: document.getElementById('btn-refresh-results'),
    loader: document.getElementById('global-loader'),
    loaderText: document.getElementById('loader-text')
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    setupEventListeners();
});

function setupEventListeners() {
    elements.btnCreateJob.addEventListener('click', createJob);

    elements.inputPages.addEventListener('change', () => {
        elements.countPages.innerText = `${elements.inputPages.files.length} files`;
        updateUploadFooter();
    });

    elements.inputRefs.addEventListener('change', () => {
        elements.countRefs.innerText = `${elements.inputRefs.files.length} files`;
        updateUploadFooter();
    });

    elements.btnUploadAll.addEventListener('click', uploadAllAssets);
    elements.btnRunJob.addEventListener('click', runJob);
    elements.btnRefreshResults.addEventListener('click', fetchResults);
}

// --- API Calls ---

async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();
        const dot = elements.apiStatus.querySelector('.status-dot');
        const text = elements.apiStatus.querySelector('.status-text');

        if (data.status === 'healthy') {
            dot.classList.add('online');
            text.innerText = data.gpu_available ? 'API Online (GPU)' : 'API Online (CPU Fallback)';
        } else {
            dot.classList.add('offline');
            text.innerText = 'API Issues';
        }
    } catch (err) {
        elements.apiStatus.querySelector('.status-dot').classList.add('offline');
        elements.apiStatus.querySelector('.status-text').innerText = 'API Offline';
    }
}

async function createJob() {
    showLoader('Initializing Job...');
    try {
        const response = await fetch(`${API_BASE}/job/create`, { method: 'POST' });
        const data = await response.json();

        currentJobId = data.job_id;
        elements.currentJobId.innerText = currentJobId;
        elements.jobInfo.classList.remove('hidden');
        elements.btnCreateJob.disabled = true;
        elements.btnCreateJob.innerText = 'Job Initialized';

        // Unlock next step
        elements.step2.classList.remove('disabled');
        hideLoader();
    } catch (err) {
        alert('Failed to create job: ' + err.message);
        hideLoader();
    }
}

function updateUploadFooter() {
    if (elements.inputPages.files.length > 0 && elements.inputRefs.files.length > 0) {
        elements.footerUpload.classList.remove('hidden');
    }
}

async function uploadAllAssets() {
    if (!currentJobId) return;

    showLoader('Uploading assets...');
    try {
        // 1. Upload Pages
        const pageData = new FormData();
        for (let file of elements.inputPages.files) {
            pageData.append('files', file);
        }
        await fetch(`${API_BASE}/job/${currentJobId}/upload_pages`, {
            method: 'POST',
            body: pageData
        });

        // 2. Upload References
        const refData = new FormData();
        for (let file of elements.inputRefs.files) {
            refData.append('files', file);
        }
        await fetch(`${API_BASE}/job/${currentJobId}/upload_references`, {
            method: 'POST',
            body: refData
        });

        elements.btnUploadAll.disabled = true;
        elements.btnUploadAll.innerText = 'Assets Uploaded';
        elements.step3.classList.remove('disabled');
        hideLoader();
    } catch (err) {
        alert('Upload failed: ' + err.message);
        hideLoader();
    }
}

async function runJob() {
    if (!currentJobId) return;

    const options = {
        upscale: document.getElementById('opt-upscale').checked,
        lineart_method: document.getElementById('opt-method').value
    };

    showLoader('Starting processing...');
    try {
        await fetch(`${API_BASE}/job/${currentJobId}/run`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ options })
        });

        elements.btnRunJob.disabled = true;
        elements.statusDisplay.classList.remove('hidden');
        startPolling();
        hideLoader();
    } catch (err) {
        alert('Failed to start job: ' + err.message);
        hideLoader();
    }
}

function startPolling() {
    if (statusPollingInterval) clearInterval(statusPollingInterval);

    statusPollingInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE}/job/${currentJobId}/status`);
            const data = await response.json();

            updateProgressUI(data);

            if (data.status === 'completed' || data.status === 'partial' || data.status === 'failed') {
                clearInterval(statusPollingInterval);
                fetchResults();
            }
        } catch (err) {
            console.error('Polling error:', err);
        }
    }, 2000);
}

function updateProgressUI(data) {
    elements.jobStatusText.innerText = data.status.charAt(0).toUpperCase() + data.status.slice(1);
    elements.jobProgressPercent.innerText = `${data.progress_percent}%`;
    elements.progressFill.style.width = `${data.progress_percent}%`;
    elements.pagesCompleted.innerText = data.pages_completed;
    elements.pagesFailed.innerText = data.pages_failed;

    if (data.status === 'completed' || data.status === 'partial') {
        elements.progressFill.style.background = 'linear-gradient(90deg, #00c853, #00e676)';
    }
}

async function fetchResults() {
    try {
        const response = await fetch(`${API_BASE}/job/${currentJobId}/results`);
        const data = await response.json();

        displayResults(data);
    } catch (err) {
        console.error('Failed to fetch results:', err);
    }
}

function displayResults(data) {
    elements.resultsSection.classList.remove('hidden');
    elements.resultsGrid.innerHTML = '';

    data.pages.forEach(page => {
        const card = document.createElement('div');
        card.className = 'result-card glass';

        const imageUrl = page.output_file
            ? `${API_BASE}/job/${currentJobId}/download/${page.output_file}`
            : 'https://via.placeholder.com/250x350?text=Failed';

        card.innerHTML = `
            <div class="image-preview">
                <img src="${imageUrl}" alt="Page ${page.page_num}" loading="lazy">
            </div>
            <div class="result-info">
                <span>Page ${page.page_num}</span>
                ${page.status === 'success'
                ? `<a href="${imageUrl}" download class="btn btn-icon"><i class="fas fa-download"></i></a>`
                : `<i class="fas fa-exclamation-triangle" style="color:var(--error)"></i>`
            }
            </div>
        `;
        elements.resultsGrid.appendChild(card);
    });

    // Smooth scroll to results
    elements.resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// --- UI Helpers ---

function showLoader(text = 'Loading...') {
    elements.loaderText.innerText = text;
    elements.loader.classList.remove('hidden');
}

function hideLoader() {
    elements.loader.classList.add('hidden');
}
