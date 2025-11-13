// Enhanced AI NIDS Dashboard with improved UX/UI
// Supports toast notifications, theme toggle, animations, and accessibility

const API_BASE = 'http://localhost:5000';

// Enhanced element selectors with null safety
const els = {};

function initializeElements() {
  const elementIds = {
    // Core elements
    apiStatus: 'apiStatus',
    loadModelsBtn: 'loadModelsBtn',
    modelStatus: 'modelStatus',
    featureCount: 'featureCount',
    featuresForm: 'featuresForm',
    predictBtn: 'predictBtn',
    predictAllBtn: 'predictAllBtn',
    clearBtn: 'clearBtn',
    
    // Prediction display
    predClass: 'predClass',
    predThreat: 'predThreat',
    predConfidence: 'predConfidence',
    predAttack: 'predAttack',
    predModel: 'predModel',
    predError: 'predError',
    topProbs: 'topProbs',
    
    // UI controls
    labels: 'labels',
    feedbackLabel: 'feedbackLabel',
    sendFeedback: 'sendFeedback',
    feedbackStatus: 'feedbackStatus',
    presetThreats: 'presetThreats',
    
    // Analytics
    explainShap: 'explainShap',
    explainLime: 'explainLime',
    explanationOut: 'explanationOut',
    incUpdate: 'incUpdate',
    
    // New UI elements
    themeToggle: 'themeToggle',
    toastContainer: 'toastContainer',
    appHeader: 'appHeader',
    slideOverlay: 'slideOverlay',
    slidePanel: 'slidePanel',
    slidePanelClose: 'slidePanelClose',
    slidePanelTitle: 'slidePanelTitle',
    slidePanelContent: 'slidePanelContent',
    
    // Model comparison
    modelComparison: 'modelComparison',
    
  // Model & dataset selectors
  modelSelectSecondary: 'modelSelectSecondary',
  datasetSelect: 'datasetSelect'
  };

  // Safely get elements and handle missing ones
  Object.entries(elementIds).forEach(([key, id]) => {
    const element = document.getElementById(id);
    if (!element) {
      console.warn(`Element with ID '${id}' not found`);
    }
    els[key] = element;
  });
  
  return Object.values(els).some(el => el !== null); // Return true if any elements found
}

// Application state
let currentFeatures = [];
let lastPrediction = null;
let datasetMeta = {};
let currentTheme = localStorage.getItem('theme') || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
let animationStagger = 0;
// Models discovered from backend
let availableModels = [];
// Toast suppression flag per user request
const DISABLE_TOASTS = true;

// Debounce state for Compare All Models
let comparePending = false;
let compareTimer = null;

// Prediction Results placeholder control
function ensurePredictionPlaceholder() {
  // Create placeholder if it doesn't exist
  let ph = document.getElementById('predPlaceholder');
  if (!ph) {
    ph = document.createElement('div');
    ph.id = 'predPlaceholder';
    ph.className = 'text-sm text-muted';
    ph.style.marginTop = '8px';
    ph.textContent = 'Enter features or apply a preset,to predict the attack';
    const grid = document.querySelector('.prediction-grid');
    if (grid && grid.parentElement) {
      grid.parentElement.insertBefore(ph, grid);
    }
  }
  return ph;
}

function showPredictionPlaceholder() {
  const grid = document.querySelector('.prediction-grid');
  const probs = document.querySelector('.probabilities-section');
  const error = els.predError;
  const ph = ensurePredictionPlaceholder();
  if (grid) grid.style.display = 'none';
  if (probs) probs.style.display = 'none';
  if (error) error.style.display = 'none';
  if (ph) ph.style.display = '';
}

function showResultsSections() {
  const grid = document.querySelector('.prediction-grid');
  const probs = document.querySelector('.probabilities-section');
  const error = els.predError;
  const ph = document.getElementById('predPlaceholder');
  if (ph) ph.style.display = 'none';
  if (grid) grid.style.display = '';
  if (probs) probs.style.display = '';
  if (error) error.style.display = '';
}

function hidePredictionPlaceholder() {
  const ph = document.getElementById('predPlaceholder');
  if (ph) ph.style.display = 'none';
}

// Feature Inputs placeholder control
function ensureFeaturesPlaceholder() {
  let ph = document.getElementById('featuresPlaceholder');
  if (!ph) {
    ph = document.createElement('div');
    ph.id = 'featuresPlaceholder';
    ph.className = 'text-sm text-muted';
    ph.style.padding = '8px 0';
    ph.textContent = 'Load models to show the features';
    const wrapper = document.querySelector('.features-form-wrapper');
    if (wrapper) {
      wrapper.insertBefore(ph, wrapper.firstChild);
    }
  }
  return ph;
}

function showFeaturesPlaceholder() {
  const ph = ensureFeaturesPlaceholder();
  if (els.featuresForm) els.featuresForm.style.display = 'none';
  if (ph) ph.style.display = '';
}

function hideFeaturesPlaceholder() {
  const ph = document.getElementById('featuresPlaceholder');
  if (ph) ph.style.display = 'none';
  if (els.featuresForm) els.featuresForm.style.display = '';
}

// Toast notification system
class ToastManager {
  constructor() {
    this.toasts = [];
  }

  show(type, title, message, duration = 5000) {
    if (DISABLE_TOASTS) {
      // Quiet mode: log to console only
      console.log(`[toast:${type}] ${title} - ${message}`);
      if (els.toastContainer) els.toastContainer.style.display = 'none';
      return { dismiss: () => {} };
    }
    const toast = this.createElement(type, title, message, duration);
    els.toastContainer.appendChild(toast);
    this.toasts.push(toast);
    
    // Auto dismiss non-critical toasts
    if (type !== 'error' && duration > 0) {
      setTimeout(() => this.dismiss(toast), duration);
    }
    
    return toast;
  }

  createElement(type, title, message, duration) {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
      <div class="toast-icon" role="img" aria-label="${type} icon">
        ${this.getIcon(type)}
      </div>
      <div class="toast-content">
        <div class="toast-title">${title}</div>
        <div class="toast-message">${message}</div>
      </div>
      <button class="toast-close" aria-label="Close notification">✕</button>
    `;
    
    const closeBtn = toast.querySelector('.toast-close');
    closeBtn.addEventListener('click', () => this.dismiss(toast));
    
    return toast;
  }

  getIcon(type) {
    const icons = {
      success: '✅',
      error: '❌',
      warning: '⚠️',
      info: 'ℹ️'
    };
    return icons[type] || 'ℹ️';
  }

  dismiss(toast) {
    toast.style.animation = 'toast-slide-out 300ms cubic-bezier(0.16, 0.8, 0.24, 1) forwards';
    setTimeout(() => {
      if (toast.parentNode) {
        toast.parentNode.removeChild(toast);
      }
      this.toasts = this.toasts.filter(t => t !== toast);
    }, 300);
  }

  success(title, message) { return this.show('success', title, message); }
  error(title, message) { return this.show('error', title, message, 0); }
  warning(title, message) { return this.show('warning', title, message); }
  info(title, message) { return this.show('info', title, message); }
}

const toast = new ToastManager();

// Theme management
function initTheme() {
  // Get theme from localStorage or detect system preference
  let preferredTheme = localStorage.getItem('theme');
  if (!preferredTheme) {
    preferredTheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  
  currentTheme = preferredTheme;
  applyTheme(currentTheme);
  updateThemeToggle();
  
  // Listen for system theme changes
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
    if (!localStorage.getItem('theme')) {
      // Only follow system preference if user hasn't manually set a theme
      currentTheme = e.matches ? 'dark' : 'light';
      applyTheme(currentTheme);
      updateThemeToggle();
    }
  });
}

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  // Also add/remove class for additional CSS targeting if needed
  document.body.classList.toggle('dark-theme', theme === 'dark');
}

function toggleTheme() {
  currentTheme = currentTheme === 'light' ? 'dark' : 'light';
  applyTheme(currentTheme);
  localStorage.setItem('theme', currentTheme);
  updateThemeToggle();
  toast.info('Theme Changed', `Switched to ${currentTheme} mode`);
}

function updateThemeToggle() {
  const toggle = els.themeToggle;
  if (toggle) {
    toggle.setAttribute('aria-label', `Switch to ${currentTheme === 'light' ? 'dark' : 'light'} mode`);
    toggle.setAttribute('title', `Switch to ${currentTheme === 'light' ? 'dark' : 'light'} mode`);
    // Icon/knob is handled purely via CSS ::before
  }
}

// Enhanced animations and micro-interactions
function animateElement(element, animation = 'animate-in') {
  if (!element) return;
  
  // Respect reduced motion preference
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    return;
  }
  
  element.classList.add(animation);
  setTimeout(() => element.classList.remove(animation), 600);
}

function staggerAnimation(elements, delay = 30) {
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    return;
  }
  // Accept various collection types: HTMLCollection, NodeList, single Element, array
  if (!elements) return;
  let list;
  if (elements instanceof Element) {
    list = [elements];
  } else if (typeof elements.length === 'number' && !elements.forEach) {
    // HTMLCollection or array-like without forEach
    list = Array.from(elements);
  } else if (Array.isArray(elements)) {
    list = elements;
  } else if (typeof elements.forEach === 'function') {
    list = elements;
  } else {
    return; // Unsupported type
  }

  list.forEach((element, index) => {
    setTimeout(() => animateElement(element), index * delay);
  });
}

// Enhanced API status management with better error handling
async function apiCall(url, options = {}) {
  try {
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers
      }
    });
    
    // Handle non-JSON responses
    const contentType = response.headers.get('content-type');
    if (!contentType || !contentType.includes('application/json')) {
      throw new Error(`Server returned non-JSON response: ${response.status} ${response.statusText}`);
    }
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.error || `API Error: ${response.status} ${response.statusText}`);
    }
    
    return data;
  } catch (error) {
    if (error.name === 'TypeError' && error.message.includes('fetch')) {
      throw new Error('Network error: Unable to connect to API server');
    }
    throw error;
  }
}

async function pingApi() {
  try {
    const data = await apiCall(`${API_BASE}/status`);
    
    if (data.status === 'online') {
      updateApiStatus('online', `Online`);
  // Don't auto-load anymore - user will click button
  console.log('API ready - click "Load models" to begin');
    } else {
      updateApiStatus('warning', 'API Error');
    }
  } catch (e) {
    updateApiStatus('offline', 'Offline');
    console.error('API ping failed:', e);
  }
}

function updateApiStatus(status, text) {
  if (!els.apiStatus) return;
  
  els.apiStatus.className = `status-indicator ${status}`;
  const statusTextElement = els.apiStatus.querySelector('.status-text');
  if (statusTextElement) {
    statusTextElement.textContent = text;
  }
  
  // Show appropriate toast for status changes
  if (status === 'online' && !els.apiStatus.dataset.lastStatus) {
    toast.success('Connected', 'Successfully connected to AI NIDS API');
  } else if (status === 'offline') {
    toast.error('Connection Lost', 'Unable to connect to AI NIDS API');
  }
  
  if (els.apiStatus.dataset) {
    els.apiStatus.dataset.lastStatus = status;
  }
}

// Enhanced model management with auto-initialization
let currentDataset = 'unsw'; // Default dataset
let featuresLoaded = false;

async function initializeModels() {
  try {
    if (els.loadModelsBtn) {
      els.loadModelsBtn.disabled = true;
      els.loadModelsBtn.innerHTML = 'Loading...';
    }
    
    if (els.modelStatus) {
      els.modelStatus.textContent = 'Initializing...';
    }
    
  const data = await apiCall(`${API_BASE}/initialize`, { method: 'POST' });
    
    if (data.status === 'success') {
      currentDataset = data.default_dataset;
      availableModels = data.models || [];
      featuresLoaded = true;
      
      // Update UI
      if (els.modelStatus) {
        els.modelStatus.textContent = `Dataset: ${data.default_dataset.toUpperCase()} • Models: ${data.models.length}`;
      }
      
      if (els.featureCount) {
        els.featureCount.textContent = `Features: ${data.feature_count}`;
      }
      
      // Load features into form
      await loadFeatures(data.features);
      
      // Update other UI components
      renderLabels();
      renderPresets();
      loadMetrics();
  renderAllModels(availableModels);
      
      // Enable predict button and show action buttons
      if (els.predictBtn) {
        els.predictBtn.disabled = false;
      }
      
      // Show the action buttons in feature section
      const featureActions = document.getElementById('featureActions');
      if (featureActions) {
        featureActions.style.display = 'block';
      }
      
      toast.success('Models Loaded', `Successfully initialized ${data.models.length} models with ${data.default_dataset.toUpperCase()} dataset`);
      
    } else {
      throw new Error(data.message || 'Initialization failed');
    }
    
  } catch (e) {
    console.error('Failed to initialize models', e);
    if (els.modelStatus) {
      els.modelStatus.textContent = 'Failed to load models';
    }
    toast.error('Load Error', 'Failed to initialize models: ' + e.message);
  } finally {
    if (els.loadModelsBtn) {
      els.loadModelsBtn.disabled = false;
      els.loadModelsBtn.innerHTML = 'Load models';
    }
  }
}

async function loadDatasets() {
  // Legacy function - now replaced by initializeModels
  // Keep for compatibility but redirect to new function
  console.warn('loadDatasets() is deprecated, use initializeModels()');
  return initializeModels();
}

function renderLabels() {
  // Get labels for current dataset
  const labels = datasetMeta[currentDataset]?.labels || {
    0: "Normal", 1: "Analysis", 2: "Backdoor", 3: "DoS", 4: "Exploits",
    6: "Fuzzers", 7: "Generic", 8: "Reconnaissance", 9: "Worms"
  };
  
  if (!els.labels) return;
  
  els.labels.innerHTML = '';
  const keys = Object.keys(labels).filter(k => !isNaN(Number(k))).map(k => Number(k)).sort((a, b) => a - b);
  
  keys.forEach(k => {
    const chip = document.createElement('div');
    chip.className = 'label-chip';
    chip.innerHTML = `
      <span class="severity-indicator severity-${getSeverityClass(labels[k])}">
        <span class="severity-icon">${getSeverityIcon(labels[k])}</span>
      </span>
      <span>${k}: ${labels[k]}</span>
    `;
    
  const ds = els.datasetSelect ? els.datasetSelect.value : currentDataset;
    const meta = datasetMeta[ds];
    if (meta?.model_path) {
      chip.title = `Model: ${meta.model_path}`;
    }
    
    els.labels.appendChild(chip);
  });
  
  // Animate labels appearance
    staggerAnimation(els.labels?.children);
}

function getSeverityClass(label) {
  const lowerLabel = label.toLowerCase();
  if (lowerLabel.includes('normal') || lowerLabel.includes('benign')) return 'info';
  if (lowerLabel.includes('dos') || lowerLabel.includes('attack')) return 'critical';
  if (lowerLabel.includes('probe') || lowerLabel.includes('scan')) return 'medium';
  if (lowerLabel.includes('u2r') || lowerLabel.includes('r2l')) return 'high';
  return 'low';
}

function getSeverityIcon(label) {
  const severity = getSeverityClass(label);
  const icons = {
    info: '●',
    low: '▲',
    medium: '◆',
    high: '⬟',
    critical: '⬢'
  };
  return icons[severity] || '●';
}

async function loadFeatures(features = null) {
  // If features provided (from initialization), use them directly
  if (features && Array.isArray(features)) {
    currentFeatures = features;
    buildForm(currentFeatures);
    updateFormState();
    hideFeaturesPlaceholder();
    return;
  }
  
  // If models are not loaded yet, show placeholder and skip
  if (!featuresLoaded) {
    showFeaturesPlaceholder();
    return;
  }

  // Fallback to API call for current dataset
  if (!currentDataset) {
    toast.warning('No Dataset', 'Please initialize models first');
    return;
  }
  
  try {
    const res = await fetch(`${API_BASE}/features/${currentDataset}`);
    const j = await res.json();
    
    if (j.error) throw new Error(j.error);
    
    currentFeatures = j.features || [];
    buildForm(currentFeatures);
    updateFormState();
  hideFeaturesPlaceholder();
    
    toast.success('Features Loaded', `${currentFeatures.length} features ready for input`);
    
  } catch (e) {
    toast.error('Load Failed', e.message);
    console.error(e);
  }
}

function buildForm(features) {
  els.featuresForm.innerHTML = '';
  
  if (features.length === 0) {
    els.featuresForm.innerHTML = '<div class="text-center text-muted">No features available</div>';
    return;
  }
  
  features.forEach((name, index) => {
    const field = document.createElement('div');
    field.className = 'field';
    
    const label = document.createElement('label');
    label.className = 'field-label';
    label.textContent = name;
    label.htmlFor = `feature-${index}`;
    
    const input = document.createElement('input');
    input.type = 'number';
    input.step = 'any';
    input.placeholder = '0.000';
    input.className = 'field-input';
    input.id = `feature-${index}`;
    input.dataset.name = name;
    input.setAttribute('aria-describedby', `feature-${index}-desc`);
    
    // Add input validation and formatting
    input.addEventListener('input', (e) => {
      validateNumericInput(e.target);
    });
    
    input.addEventListener('blur', (e) => {
      formatNumericInput(e.target);
    });
    
    const desc = document.createElement('div');
    desc.id = `feature-${index}-desc`;
    desc.className = 'sr-only';
    desc.textContent = `Enter numeric value for ${name} feature`;
    
    field.appendChild(label);
    field.appendChild(input);
    field.appendChild(desc);
    els.featuresForm.appendChild(field);
  });
  
  // Animate form fields
  staggerAnimation(els.featuresForm?.children, 20);
}

function validateNumericInput(input) {
  const value = input.value;
  const isValid = value === '' || !isNaN(parseFloat(value));
  
  input.classList.toggle('invalid', !isValid);
  input.setAttribute('aria-invalid', !isValid);
}

function formatNumericInput(input) {
  if (input.value && !isNaN(parseFloat(input.value))) {
    const num = parseFloat(input.value);
    input.value = num.toFixed(6);
  }
}

function updateFormState() {
  const hasFeatures = currentFeatures.length > 0;
  const hasModels = availableModels.length > 0;
  
  els.predictBtn.disabled = !hasFeatures;
  els.predictAllBtn.disabled = !hasFeatures || !hasModels;
  els.clearBtn.disabled = !hasFeatures;
  // Form state validation complete
}

function collectValues() {
  const values = [];
  const inputs = els.featuresForm.querySelectorAll('input');
  
  inputs.forEach(inp => {
    const v = inp.value === '' ? 0 : Number(inp.value);
    values.push(v);
  });
  
  return values;
}

// Enhanced prediction with multi-model support
async function predict() {
  // Clear previous results
  clearPredictionDisplay();
  
  if (!featuresLoaded) {
    toast.warning('Models Not Loaded', 'Please load models first');
    return;
  }
  
  if (currentFeatures.length === 0) {
    toast.warning('No Features', 'Please load features first');
    return;
  }
  
  const features = collectValues();
  
  if (features.length === 0) {
    toast.warning('No Features', 'Please load features first');
    return;
  }
  
  // Get selected model if any
  const selectedModel = els.modelSelectSecondary?.value;
  
  // Show loading state
  if (els.predictBtn) {
    els.predictBtn.disabled = true;
  els.predictBtn.innerHTML = 'Predicting...';
  }
  
  try {
    const start = performance.now();
    
    // Build request payload with model selection support
    const payload = {
      features: features,
      dataset: currentDataset
    };
    
    // Add specific model if selected
    if (selectedModel && selectedModel !== '') {
      payload.model_key = selectedModel;
    }
    
    const data = await apiCall(`${API_BASE}/predict`, {
      method: 'POST',
      body: JSON.stringify(payload)
    });
    
    const latency = performance.now() - start;
    
  // Reveal results section and update with animations
  showResultsSections();
    updatePredictionDisplay(data, data.dataset || currentDataset);
    
    // Store for feedback
    lastPrediction = { 
      dataset: data.dataset || currentDataset, 
      features, 
      predicted_label: data.prediction,
      model_key: data.model_key
    };
    if (els.sendFeedback) {
      els.sendFeedback.disabled = false;
    }
    
    const modelInfo = selectedModel ? `${data.algorithm || selectedModel}` : `${data.algorithm || 'Default Model'}`;
    toast.success('Prediction Complete', `${modelInfo}: ${data.is_attack ? 'Attack' : 'Normal'}${data.confidence ? ' @ ' + (data.confidence*100).toFixed(1)+'%' : ''}`);
    
  } catch (e) {
    if (els.predError) {
      els.predError.textContent = e.message;
    }
    toast.error('Prediction Failed', e.message);
    console.error('Prediction error:', e);
  } finally {
    // Reset button state
    if (els.predictBtn) {
      els.predictBtn.disabled = false;
  els.predictBtn.innerHTML = 'Predict';
    }
  }
}

// Compare all available models
async function predictAllModels() {
  // Debounce fast repeated clicks; if running or just clicked, ignore
  if (comparePending) return;
  if (compareTimer) return;
  compareTimer = setTimeout(() => { compareTimer = null; }, 400);

  if (!featuresLoaded || availableModels.length === 0) {
    toast.warning('Models Not Ready', 'Please load models first');
    return;
  }
  
  const features = collectValues();
  if (features.length === 0) {
    toast.warning('No Features', 'Please load features first');
    return;
  }
  
  // Show loading state
  if (els.predictAllBtn) {
    els.predictAllBtn.disabled = true;
    els.predictAllBtn.innerHTML = 'Comparing...';
  }
  // Header spinner on
  const cmpSpin = document.getElementById('comparisonSpinner');
  if (cmpSpin) {
    cmpSpin.classList.add('active');
    cmpSpin.setAttribute('aria-busy', 'true');
    cmpSpin.setAttribute('title', 'Comparing models...');
  }

  comparePending = true;
  
  // Clear previous comparison results
  if (els.modelComparison) {
    els.modelComparison.innerHTML = '<div class="text-center text-muted">Running predictions on all models...</div>';
  }
  
  try {
    const startTime = performance.now();
    
    // Use batch prediction endpoint for better performance
    const data = await apiCall(`${API_BASE}/predict_batch`, {
      method: 'POST',
      body: JSON.stringify({
        features: features
      })
    });
    
    const totalTime = performance.now() - startTime;
    
    // Merge with available models data for metrics
    const predictions = data.results.map(result => {
      const modelInfo = availableModels.find(m => m.model_key === result.model_key) || {};
      return {
        ...result,
        metrics: modelInfo.metrics || {}
      };
    });
    
    // Display comparison results
    displayModelComparison(predictions, totalTime);
    
  const successCount = data.successful_predictions || predictions.filter(p => !p.error).length;
    toast.success('Comparison Complete', `Tested ${successCount}/${data.total_models} models in ${totalTime.toFixed(0)}ms`);
    
  } catch (e) {
    toast.error('Comparison Failed', e.message);
    console.error('Model comparison error:', e);
    if (els.modelComparison) {
      els.modelComparison.innerHTML = `<div class="text-center text-danger">Error: ${e.message}</div>`;
    }
  } finally {
    // Reset button state
    if (els.predictAllBtn) {
      els.predictAllBtn.disabled = false;
      els.predictAllBtn.innerHTML = 'Compare All Models';
    }
    // Header spinner off
    const cmpSpinOff = document.getElementById('comparisonSpinner');
    if (cmpSpinOff) {
      cmpSpinOff.classList.remove('active');
      cmpSpinOff.setAttribute('aria-busy', 'false');
      cmpSpinOff.setAttribute('title', 'Waiting to compare');
    }
    comparePending = false;
  }
}

// Display model comparison results
function displayModelComparison(predictions, totalTime) {
  if (!els.modelComparison) return;
  
  // Sort by confidence (descending) with errors at the end
  const sortedPredictions = predictions.sort((a, b) => {
    if (a.error && !b.error) return 1;
    if (!a.error && b.error) return -1;
    if (a.error && b.error) return 0;
    return (b.confidence || 0) - (a.confidence || 0);
  });
  
  // Count consensus
  const attackCount = sortedPredictions.filter(p => !p.error && p.is_attack).length;
  const normalCount = sortedPredictions.filter(p => !p.error && !p.is_attack).length;
  const errorCount = sortedPredictions.filter(p => p.error).length;
  
  const consensus = attackCount > normalCount ? 'Attack' : 'Normal';
  const consensusConfidence = Math.max(attackCount, normalCount) / (attackCount + normalCount);
  
  els.modelComparison.innerHTML = `
    <div class="comparison-summary">
      <h4>Consensus: <span class="consensus-${consensus.toLowerCase()}">${consensus}</span> (${(consensusConfidence * 100).toFixed(1)}% agreement)</h4>
      <div class="comparison-stats">
        <span>Attacks: ${attackCount}</span>
        <span>Normal: ${normalCount}</span>
        <span>Errors: ${errorCount}</span>
        <span>Total: ${totalTime.toFixed(0)}ms</span>
      </div>
    </div>
    <div class="comparison-results">
      ${sortedPredictions.map((pred, index) => `
        <div class="model-result ${pred.error ? 'error' : pred.is_attack ? 'attack' : 'normal'}">
          <div class="model-result-header">
            <strong>${pred.algorithm}</strong>
            <span class="model-confidence">${pred.error ? 'Error' : (pred.confidence ? (pred.confidence * 100).toFixed(1) + '%' : 'N/A')}</span>
          </div>
          <div class="model-result-details">
            ${pred.error ? 
              `<span class="error-text">${pred.error}</span>` :
              `<span class="prediction-class">Class: ${pred.prediction}</span>
               <span class="threat-type">${pred.threat}</span>
               <span class="attack-status ${pred.is_attack ? 'attack' : 'normal'}">${pred.is_attack ? 'Attack' : 'Normal'}</span>`
            }
          </div>
          ${pred.metrics && pred.metrics.accuracy ? 
            `<div class="model-accuracy">Accuracy: ${(pred.metrics.accuracy * 100).toFixed(1)}%</div>` : 
            ''
          }
        </div>
      `).join('')}
    </div>
  `;
  
  // Animate results
  setTimeout(() => {
    const results = els.modelComparison.querySelectorAll('.model-result');
    staggerAnimation(results, 100);
  }, 100);
}

function clearPredictionDisplay() {
  // Reset values but keep them hidden until a prediction occurs
  if (els.predError) els.predError.textContent = '';
  if (els.predClass) els.predClass.textContent = '';
  if (els.predThreat) els.predThreat.textContent = '';
  if (els.predConfidence) els.predConfidence.textContent = '';
  if (els.predAttack) els.predAttack.textContent = '';
  if (els.predModel) els.predModel.textContent = '';
  if (els.topProbs) els.topProbs.innerHTML = '';
  showPredictionPlaceholder();
}

function updatePredictionDisplay(prediction, dataset) {
  // Ensure sections are visible when updating
  showResultsSections();
  // Animate values appearing with enhanced model information
  const modelInfo = prediction.algorithm || prediction.model_type || 'Unknown';
  const modelKey = prediction.model_key || 'default';
  
  const updates = [
    { element: els.predClass, value: prediction.prediction },
    { element: els.predThreat, value: prediction.threat },
    { element: els.predConfidence, value: prediction.confidence ? `${(prediction.confidence * 100).toFixed(1)}%` : '—' },
    { element: els.predModel, value: `${modelInfo} (${modelKey})` }
  ];
  
  updates.forEach((update, index) => {
    setTimeout(() => {
      update.element.textContent = update.value;
      animateElement(update.element);
    }, index * 100);
  });
  
  // Special handling for attack status
  setTimeout(() => {
    const isAttack = prediction.is_attack;
    els.predAttack.textContent = isAttack ? 'YES' : 'NO';
    els.predAttack.className = `prediction-value ${isAttack ? 'attack-yes' : 'attack-no'}`;
    animateElement(els.predAttack);
  }, 300);
  
  // Display probabilities with animation
  if (prediction.probabilities) {
    setTimeout(() => {
      displayTopProbabilities(prediction.probabilities, datasetMeta[dataset]?.labels || {});
    }, 500);
  }
}

function displayTopProbabilities(probabilities, labelMap) {
  const probArray = Object.entries(probabilities)
    .map(([idx, prob]) => ({
      index: parseInt(idx),
      probability: prob,
      label: labelMap[idx] || `Class ${idx}`,
      severity: getSeverityClass(labelMap[idx] || '')
    }))
    .sort((a, b) => b.probability - a.probability)
    .slice(0, 5);
  
  els.topProbs.innerHTML = '';
  
  probArray.forEach((item, index) => {
    const probItem = document.createElement('div');
    probItem.className = 'probability-item';
    const percentage = (item.probability * 100).toFixed(1);
    
    probItem.innerHTML = `
      <span class="probability-label">
        <span class="severity-indicator severity-${item.severity}">
          <span class="severity-icon">${getSeverityIcon(item.label)}</span>
        </span>
        ${item.label}
      </span>
      <span class="probability-value">${percentage}%</span>
      <div class="probability-bar">
        <div class="probability-fill" style="width: 0%"></div>
      </div>
    `;
    
    els.topProbs.appendChild(probItem);
    
    // Animate probability bar
    setTimeout(() => {
      const fillElement = probItem.querySelector('.probability-fill');
      fillElement.style.width = `${percentage}%`;
      animateElement(probItem);
    }, index * 150);
  });
}

// Enhanced feedback system
async function sendFeedback() {
  if (!lastPrediction) {
    toast.warning('No Prediction', 'Make a prediction first before sending feedback');
    return;
  }
  
  const feedbackType = els.feedbackLabel.value;
  const payload = {
    features: lastPrediction.features,
    predicted_label: lastPrediction.predicted_label,
    feedback_label: feedbackType,
    timestamp: new Date().toISOString()
  };
  
  els.sendFeedback.disabled = true;
  els.sendFeedback.innerHTML = '<span role="img" aria-label="Sending">📤</span> Sending...';
  
  try {
    const res = await fetch(`${API_BASE}/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    const j = await res.json();
    
    if (!res.ok) {
      throw new Error(j.error || 'Feedback failed');
    }
    
    els.feedbackStatus.textContent = 'Feedback recorded successfully';
    els.feedbackStatus.style.color = 'var(--color-success)';
    
    toast.success('Feedback Sent', 'Thank you for improving the model');
    
    setTimeout(() => {
      els.feedbackStatus.textContent = '—';
      els.feedbackStatus.style.color = '';
    }, 3000);
    
  } catch (e) {
    els.feedbackStatus.textContent = `Error: ${e.message}`;
    els.feedbackStatus.style.color = 'var(--color-danger)';
    toast.error('Feedback Failed', e.message);
  } finally {
    els.sendFeedback.disabled = false;
    els.sendFeedback.innerHTML = '<span role="img" aria-label="Send icon">📤</span> Send Feedback';
  }
}

// Enhanced form management
function clearForm() {
  els.featuresForm.querySelectorAll('input').forEach(input => {
    input.value = '';
    input.classList.remove('invalid');
    input.setAttribute('aria-invalid', 'false');
  });
  
  clearPredictionDisplay();
  els.sendFeedback.disabled = true;
  
  toast.info('Form Cleared', 'All input fields have been reset');
}


// Enhanced presets with better UX
function renderPresets() {
  const ds = els.datasetSelect ? els.datasetSelect.value : currentDataset;
  const labels = (datasetMeta[ds]?.labels) || {};
  
  els.presetThreats.innerHTML = '';
  const entries = Object.entries(labels);
  if (entries.length === 0) {
    els.presetThreats.innerHTML = '<div class="text-sm text-muted">No label presets available.</div>';
    return;
  }
  // Special grouping for cicids
  if (ds === 'cicids') {
    // Group labels into categories
    const groups = {
      'DoS / DDoS': entries.filter(([_, n]) => /dos|ddos/i.test(n)),
      'Web Attacks': entries.filter(([_, n]) => /web attack/i.test(n)),
      'Brute/Patator': entries.filter(([_, n]) => /patator/i.test(n)),
      'Other': entries.filter(([_, n]) => !(/dos|ddos|web attack|patator/i.test(n)))
    };
    Object.entries(groups).forEach(([groupName, groupEntries]) => {
      if (groupEntries.length === 0) return;
      const groupDiv = document.createElement('div');
      groupDiv.className = 'preset-group';
      const heading = document.createElement('div');
      heading.className = 'preset-group-title';
      heading.textContent = groupName;
      groupDiv.appendChild(heading);
      const wrap = document.createElement('div');
      wrap.className = 'chip-group-inner';
      groupEntries.forEach(([id, name]) => {
        const chip = document.createElement('button');
        chip.className = 'chip';
        const cleanName = name.replace('\ufffd', '-');
        chip.innerHTML = `
          <span class="severity-indicator severity-${getSeverityClass(cleanName)}">
            <span class="severity-icon">${getSeverityIcon(cleanName)}</span>
          </span>
          ${cleanName}
        `;
        chip.title = `Load sample data for ${cleanName} (Class ${id})`;
        chip.setAttribute('aria-label', `Load ${cleanName} sample data`);
        chip.addEventListener('click', (event) => applyThreatPreset(Number(id), event, cleanName));
        wrap.appendChild(chip);
      });
      groupDiv.appendChild(wrap);
      els.presetThreats.appendChild(groupDiv);
    });
  } else {
    // Default behavior for other datasets
    entries.forEach(([id, name]) => {
      const chip = document.createElement('button');
      chip.className = 'chip';
      chip.innerHTML = `
        <span class="severity-indicator severity-${getSeverityClass(name)}">
          <span class="severity-icon">${getSeverityIcon(name)}</span>
        </span>
        ${name}
      `;
      chip.title = `Load sample data for ${name} (Class ${id})`;
      chip.setAttribute('aria-label', `Load ${name} sample data`);
      chip.addEventListener('click', (event) => applyThreatPreset(Number(id), event));
      els.presetThreats.appendChild(chip);
    });
  }
  
  // Animate preset chips
  staggerAnimation(els.presetThreats?.children, 50);
}

async function applyThreatPreset(threatId, event = null, labelName = null) {
  const selectedModel = els.modelSelectSecondary?.value;
  const dataset = els.datasetSelect ? els.datasetSelect.value : currentDataset;
  
  // Prioritize model selection over dataset selection
  if (!selectedModel && !dataset) {
    toast.warning('No Model Selected', 'Please select a model first');
    return;
  }
  
  if (currentFeatures.length === 0) {
    await loadFeatures();
  }
  
  // Show loading state
  if (event?.target) {
    const originalText = event.target.innerHTML;
    const originalClasses = event.target.className;
    event.target.innerHTML = '<span role="img" aria-label="Loading">🔄</span> Loading...';
    event.target.disabled = true;
    event.target.classList.add('loading');
    
    // Store original state for restoration
    event.target._originalText = originalText;
    event.target._originalClasses = originalClasses;
  }
  
  try {
    // Use model-aware approach if a specific model is selected
    if (selectedModel) {
      // Find the model's dataset
      const modelInfo = availableModels.find(m => m.model_key === selectedModel);
      const modelDataset = modelInfo?.dataset_guess || dataset;
      
      const body = { dataset: modelDataset, label: threatId };
      if (modelDataset === 'cicids' && labelName) body.label_name = labelName;
      
      const res = await fetch(`${API_BASE}/preset/sample`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      
      const j = await res.json();
      if (!res.ok) throw new Error(j.error || 'Preset fetch failed');
      
      console.log('Got preset response for model:', { 
        selectedModel,
        dataset: j.dataset, 
        label: j.label, 
        threat: j.threat, 
        featuresCount: j.features?.length, 
        featureOrderCount: j.feature_order?.length 
      });
      
      // Fill form with sample data
      fillFormWithData(j.feature_order || currentFeatures, j.features || []);
      
      const labelName = datasetMeta[modelDataset]?.labels?.[threatId] || `Class ${threatId}`;
      toast.success('Sample Loaded', `Applied ${labelName} sample for ${selectedModel}`);
      
    } else {
      // Fallback to old dataset-only approach
      const body = { dataset, label: threatId };
      if (dataset === 'cicids' && labelName) body.label_name = labelName;
      
      const res = await fetch(`${API_BASE}/preset/sample`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      
      const j = await res.json();
      if (!res.ok) throw new Error(j.error || 'Preset fetch failed');
      
      console.log('Got preset response:', { 
        dataset: j.dataset, 
        label: j.label, 
        threat: j.threat, 
        featuresCount: j.features?.length, 
        featureOrderCount: j.feature_order?.length 
      });
      
      // Fill form with sample data
      fillFormWithData(j.feature_order || currentFeatures, j.features || []);
      
      const labelName = datasetMeta[dataset]?.labels?.[threatId] || `Class ${threatId}`;
      toast.success('Sample Loaded', `Applied ${labelName} sample data`);
    }
    
    // Visual feedback
    if (event?.target) {
      // Restore original state
      event.target.innerHTML = event.target._originalText || event.target.innerHTML.replace('<span role="img" aria-label="Loading">🔄</span> Loading...', '');
      event.target.className = event.target._originalClasses || event.target.className.replace(' loading', '');
      event.target.disabled = false;
      
      // Success animation
      event.target.classList.add('active');
      setTimeout(() => {
        event.target.classList.remove('active');
      }, 2000);
    }
    
  } catch (e) {
    console.error(e);
    els.predError.textContent = e.message;
    toast.error('Preset Failed', e.message);
    
    // Restore button state on error
    if (event?.target) {
      event.target.innerHTML = event.target._originalText || event.target.innerHTML.replace('<span role="img" aria-label="Loading">🔄</span> Loading...', '');
      event.target.className = event.target._originalClasses || event.target.className.replace(' loading', '');
      event.target.disabled = false;
    }
  } finally {
    // Reset button state
    if (event?.target) {
      const selectedModel = els.modelSelectSecondary?.value;
      const modelInfo = availableModels.find(m => m.model_key === selectedModel);
      const targetDataset = modelInfo?.dataset_guess || els.datasetSelect.value;
      const labelName = datasetMeta[targetDataset]?.labels?.[threatId] || `Class ${threatId}`;
      event.target.innerHTML = `
        <span class="severity-indicator severity-${getSeverityClass(labelName)}">
          <span class="severity-icon">${getSeverityIcon(labelName)}</span>
        </span>
        ${labelName}
      `;
      event.target.disabled = false;
    }
  }
}

function fillFormWithData(featureOrder, values) {
  console.log('fillFormWithData called with:', { featureOrder: featureOrder?.slice(0, 3), values: values?.slice(0, 3) });
  
  const featureMap = {};
  featureOrder.forEach((feature, index) => {
    if (index < values.length) {
      featureMap[feature] = values[index];
    }
  });
  
  console.log('Created feature map with', Object.keys(featureMap).length, 'entries');
  
  const inputs = els.featuresForm.querySelectorAll('input');
  console.log('Found', inputs.length, 'input fields in form');
  
  let filled = 0;
  inputs.forEach(input => {
    const name = input.dataset.name;
    if (name in featureMap) {
      input.value = parseFloat(featureMap[name]).toFixed(6);
      animateElement(input.parentElement);
      filled++;
    }
  });
  
  console.log('Filled', filled, 'input fields');
}

// Enhanced metrics and KPI management
async function loadMetrics() {
  try {
    const [metaRes, modelsRes] = await Promise.all([
      fetch(`${API_BASE}/model_meta`),
      fetch(`${API_BASE}/models`)
    ]);
    
    if (!metaRes.ok || !modelsRes.ok) return;
    
    const metaData = await metaRes.json();
    const modelsData = await modelsRes.json();
    
    const datasets = metaData.datasets || [];
    const allModels = modelsData.models || [];
    
    renderAllModels(allModels);
    
  } catch (e) {
    console.error('Failed to load metrics:', e);
    toast.error('Metrics Error', 'Failed to load performance metrics');
  }
}

function renderAllModels(models) {
  const allModelsGrid = document.getElementById('allModels');
  if (!allModelsGrid) return;

  allModelsGrid.innerHTML = '';

  if (!Array.isArray(models)) {
    allModelsGrid.innerHTML = '<div class="text-sm text-muted">No model data loaded yet.</div>';
    return;
  }

  if (models.length === 0) {
    allModelsGrid.innerHTML = '<div class="text-sm text-muted">No models found.</div>';
    return;
  }
  
  models.forEach(model => {
    const modelCard = document.createElement('div');
    modelCard.className = 'model-card';

    // Prefer backend-provided algorithm_guess, fallback to extraction
    let algorithmName = model.algorithm_guess || extractAlgorithmName(model.model_key);
    if (model.balanced && !/balanced/i.test(algorithmName)) {
      algorithmName += ' (Balanced)';
    }

    const m = model.metrics || {};
    
    // Handle accuracy - values are already in decimal format (0.0-1.0)
    const accuracy = (typeof m.accuracy === 'number') ? m.accuracy : (typeof m.acc === 'number' ? m.acc : undefined);
    const f1Score = (typeof m.f1_weighted === 'number') ? m.f1_weighted : (typeof m.f1 === 'number' ? m.f1 : (typeof m.f1_score === 'number' ? m.f1_score : undefined));
    const precision = (typeof m.precision_weighted === 'number') ? m.precision_weighted : undefined;
    const recall = (typeof m.recall_weighted === 'number') ? m.recall_weighted : undefined;

    // Convert to percentage (multiply by 100) for display
    const accuracyText = accuracy != null ? `${(accuracy * 100).toFixed(2)}%` : '—';
    const f1Text = f1Score != null ? `${(f1Score * 100).toFixed(2)}%` : '—';
    const precisionText = precision != null ? `${(precision * 100).toFixed(2)}%` : '—';
    const recallText = recall != null ? `${(recall * 100).toFixed(2)}%` : '—';

    const sampleCount = model.samples != null ? model.samples.toLocaleString() : '—';
    const datasetBadge = model.dataset_guess ? `<span class="dataset-badge">${model.dataset_guess.toUpperCase()}</span>` : '';

    function severityColor(v) {
      if (v == null) return 'muted';
      if (v >= 0.9) return 'success';
      if (v >= 0.75) return 'warning';
      return 'danger';
    }

    modelCard.innerHTML = `
      <div class="model-header">
        <span class="model-name">${algorithmName}</span>
        ${datasetBadge}
      </div>
      <div class="model-details">
        <div class="model-row"><span>Algorithm:</span><strong>${algorithmName}</strong></div>
        <div class="model-row"><span>Accuracy:</span><strong style="color: var(--color-${severityColor(accuracy)});">${accuracyText}</strong></div>
        <div class="model-row"><span>F1 Score:</span><strong style="color: var(--color-${severityColor(f1Score)});">${f1Text}</strong></div>
        <div class="model-row"><span>Precision:</span><strong>${precisionText}</strong></div>
        <div class="model-row"><span>Recall:</span><strong>${recallText}</strong></div>
        <div class="model-row"><span>Samples:</span><strong>${sampleCount}</strong></div>
        <div class="model-row"><span>Size:</span><strong>${model.size_mb ? model.size_mb + ' MB' : '—'}</strong></div>
      </div>
    `;

    allModelsGrid.appendChild(modelCard);
  });
  
  staggerAnimation(allModelsGrid.children);
}

function extractAlgorithmName(modelKey) {
  // Extract algorithm name from model key
  const key = modelKey.toLowerCase();
  const isBalanced = key.includes('balanced');
  
  if (key.includes('rf') || key.includes('randomforest')) {
    if (key.includes('deep')) return 'Random Forest (Deep)';
    return isBalanced ? 'Random Forest (Balanced)' : 'Random Forest';
  }
  if (key.includes('gan') && key.includes('cnn')) return 'CNN-GAN';
  if (key.includes('extratrees')) return 'Extra Trees';
  if (key.includes('gradientboosting') || key.includes('gb')) return 'Gradient Boosting';
  if (key.includes('svm')) {
    if (key.includes('rbf')) return 'SVM (RBF)';
    return 'SVM';
  }
  if (key.includes('logistic')) return 'Logistic Regression';
  if (key.includes('mlp') || key.includes('neural')) return 'Neural Network (MLP)';
  if (key.includes('decisiontree') || key.includes('tree')) return 'Decision Tree';
  if (key.includes('naivebayes') || key.includes('nb')) return 'Naive Bayes';
  if (key.includes('adversarial') || key.includes('cnn')) return 'CNN (Adversarial)';
  
  // Fallback: clean up the model key
  return modelKey.replace('saved_model_', '').replace(/_/g, ' ')
    .replace(/\b\w/g, l => l.toUpperCase());
}

function formatMetric(value) {
  if (typeof value === 'number') {
    return `${(value * 100).toFixed(1)}%`;
  }
  return value || '—';
}

// Enhanced explainability features
async function explainShap() {
  if (!lastPrediction) {
    toast.warning('No Prediction', 'Make a prediction first to get explanations');
    return;
  }
  
  els.explanationOut.textContent = 'Computing SHAP analysis...';
  els.explainShap.disabled = true;
  
  try {
    const res = await fetch(`${API_BASE}/explain/shap`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dataset: lastPrediction.dataset,
        features: lastPrediction.features
      })
    });
    
    const j = await res.json();
    if (!res.ok) throw new Error(j.error || 'SHAP analysis failed');
    
    const shapText = j.shap_values
      .map(r => `${r.feature}: ${r.importance.toFixed(4)}`)
      .slice(0, 10)
      .join('\n');
    
    els.explanationOut.textContent = shapText;
    toast.success('SHAP Complete', 'Feature importance analysis ready');
    
  } catch (e) {
    els.explanationOut.textContent = `Error: ${e.message}`;
    toast.error('SHAP Failed', e.message);
  } finally {
    els.explainShap.disabled = false;
  }
}

async function explainLime() {
  if (!lastPrediction) {
    toast.warning('No Prediction', 'Make a prediction first to get explanations');
    return;
  }
  
  els.explanationOut.textContent = 'Computing LIME analysis...';
  els.explainLime.disabled = true;
  
  try {
    const res = await fetch(`${API_BASE}/explain/lime`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dataset: lastPrediction.dataset,
        features: lastPrediction.features
      })
    });
    
    const j = await res.json();
    if (!res.ok) throw new Error(j.error || 'LIME analysis failed');
    
    const limeText = j.lime_explanation
      .map(r => `${r.feature}: ${r.weight.toFixed(3)}`)
      .slice(0, 10)
      .join('\n');
    
    els.explanationOut.textContent = limeText;
    toast.success('LIME Complete', 'Local explanation analysis ready');
    
  } catch (e) {
    els.explanationOut.textContent = `Error: ${e.message}`;
    toast.error('LIME Failed', e.message);
  } finally {
    els.explainLime.disabled = false;
  }
}

async function incrementalUpdate() {
  els.incUpdate.disabled = true;
  els.incUpdate.textContent = 'Updating...';
  
  try {
    const res = await fetch(`${API_BASE}/incremental/update`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset: 'unsw' })
    });
    
    const j = await res.json();
    if (!res.ok) throw new Error(j.error || 'Update failed');
    
    loadMetrics();
    toast.success('Model Updated', `Processed ${j.samples_used} new samples`);
    
  } catch (e) {
    toast.error('Update Failed', e.message);
  } finally {
    els.incUpdate.disabled = false;
    els.incUpdate.innerHTML = '<span aria-hidden="true"><svg class="icon" width="16" height="16"><use href="#icon-update"/></svg></span> Update Model';
  }
}

// Enhanced scroll effects
function initScrollEffects() {
  if (!els.appHeader) return;
  
  let lastScrollY = window.scrollY;
  
  function updateHeader() {
    const currentScrollY = window.scrollY;
    
    if (currentScrollY > 10) {
      els.appHeader.classList.add('scrolled');
    } else {
      els.appHeader.classList.remove('scrolled');
    }
    
    lastScrollY = currentScrollY;
  }
  
  window.addEventListener('scroll', updateHeader, { passive: true });
  updateHeader(); // Initial call
}

// Slide panel management
function openSlidePanel(title, content) {
  if (!els.slidePanel || !els.slideOverlay) return;
  
  els.slidePanelTitle.textContent = title;
  els.slidePanelContent.innerHTML = content;
  
  els.slideOverlay.classList.add('active');
  els.slidePanel.classList.add('active');
  
  // Focus management for accessibility
  els.slidePanelClose.focus();
  
  // Prevent body scroll
  document.body.style.overflow = 'hidden';
}

function closeSlidePanel() {
  if (!els.slidePanel || !els.slideOverlay) return;
  
  els.slideOverlay.classList.remove('active');
  els.slidePanel.classList.remove('active');
  
  // Restore body scroll
  document.body.style.overflow = '';
  
  // Return focus to trigger element if available
  const triggerElement = document.querySelector('[data-slide-trigger]');
  if (triggerElement) {
    triggerElement.focus();
  }
}

// Auto-refresh with exponential backoff
let refreshAttempts = 0;
const MAX_REFRESH_ATTEMPTS = 5;
const BASE_REFRESH_DELAY = 30000; // 30 seconds

async function autoRefreshStatus() {
  try {
    const res = await fetch(`${API_BASE}/status`);
    const j = await res.json();
    
    if (j.status === 'online') {
      updateApiStatus('online', `Online • ${j.models.available}/${j.models.total} models • ${j.models.cached} cached • ${j.feedback_samples} samples`);
      refreshAttempts = 0; // Reset on success
    } else {
      updateApiStatus('warning', 'API Error');
    }
  } catch (e) {
    updateApiStatus('offline', 'Connection Lost');
    refreshAttempts++;
    
    // Exponential backoff on repeated failures
    if (refreshAttempts >= MAX_REFRESH_ATTEMPTS) {
      clearInterval(refreshInterval);
      toast.error('Connection Failed', 'Auto-refresh disabled after multiple failures. Please refresh manually.');
    }
  }
}

let refreshInterval;

function startAutoRefresh() {
  refreshInterval = setInterval(autoRefreshStatus, BASE_REFRESH_DELAY);
}

function stopAutoRefresh() {
  if (refreshInterval) {
    clearInterval(refreshInterval);
    refreshInterval = null;
  }
}

// Enhanced keyboard shortcuts - removed per user request
function initKeyboardShortcuts() {
  // Keyboard shortcuts have been removed per user request
  // Only keeping essential accessibility shortcuts
  document.addEventListener('keydown', (e) => {
    // Escape: Close slide panel
    if (e.key === 'Escape') {
      if (els.slidePanel?.classList.contains('active')) {
        closeSlidePanel();
      }
    }
  });
}

// Event Listeners Setup
function initEventListeners() {
  // Core functionality - with null checks
  els.loadModelsBtn?.addEventListener('click', initializeModels);
  
  els.predictBtn?.addEventListener('click', predict);
  els.predictAllBtn?.addEventListener('click', predictAllModels);
  els.clearBtn?.addEventListener('click', clearForm);
  els.sendFeedback?.addEventListener('click', sendFeedback);
  
  // bulkFill removed
  
  if (els.themeToggle) {
    els.themeToggle.addEventListener('click', toggleTheme);
    els.themeToggle.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        toggleTheme();
      }
    });
  }
  
  // Explainability
  els.explainShap?.addEventListener('click', explainShap);
  els.explainLime?.addEventListener('click', explainLime);
  els.incUpdate?.addEventListener('click', incrementalUpdate);
  
  // Slide panel
  els.slidePanelClose?.addEventListener('click', closeSlidePanel);
  
  if (els.slideOverlay) {
    els.slideOverlay.addEventListener('click', (e) => {
      if (e.target === els.slideOverlay) {
        closeSlidePanel();
      }
    });
  }
  
  // Slide panel tabs
  document.querySelectorAll('.slide-panel-tab').forEach(tab => {
    tab.addEventListener('click', (e) => {
      document.querySelectorAll('.slide-panel-tab').forEach(t => t.classList.remove('active'));
      e.target.classList.add('active');
      
      const tabType = e.target.dataset.tab;
      loadSlideContent(tabType);
    });
  });
  
  // Unified model selector change handler (header and sidebar)
  function handleModelChange(sourceSelect) {
    const selectedModel = sourceSelect.value;
    if (selectedModel && availableModels.length > 0) {
      const model = availableModels.find(m => m.model_key === selectedModel);
      if (model && model.dataset_guess && els.datasetSelect) {
        els.datasetSelect.value = model.dataset_guess;
        els.datasetSelect.dispatchEvent(new Event('change'));
        toast.info('Model Selected', `${selectedModel} → ${model.dataset_guess} dataset`);
      } else {
        toast.info('Model Selected', selectedModel);
      }
    } else {
      toast.info('Model Selected', 'Auto');
    }
    // No need to sync since we only have one model selector now
  }
  
  // Only the secondary model selector now
  els.modelSelectSecondary?.addEventListener('change', () => handleModelChange(els.modelSelectSecondary));
  // Dataset change listener
  els.datasetSelect?.addEventListener('change', async () => {
    currentDataset = els.datasetSelect.value;
    await loadFeatures();
    renderPresets();
  });
  
  // Window events
  window.addEventListener('beforeunload', () => {
    stopAutoRefresh();
  });
  
  // Visibility change handling
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      stopAutoRefresh();
    } else {
      startAutoRefresh();
    }
  });
}

function loadSlideContent(tabType) {
  const content = els.slidePanelContent;
  if (!content) return;
  
  switch (tabType) {
    case 'overview':
      content.innerHTML = '<div class="text-center text-muted">Overview content would go here</div>';
      break;
    case 'features':
      content.innerHTML = '<div class="text-center text-muted">Feature analysis would go here</div>';
      break;
    case 'correlation':
      content.innerHTML = '<div class="text-center text-muted">Correlation analysis would go here</div>';
      break;
    case 'raw':
      content.innerHTML = '<div class="text-center text-muted">Raw data would go here</div>';
      break;
    default:
      content.innerHTML = '<div class="text-center text-muted">Content not available</div>';
  }
}

// Performance monitoring
function initPerformanceMonitoring() {
  // Track page load performance
  window.addEventListener('load', () => {
    try {
      let loadTime = null;
      if (performance.getEntriesByType) {
        const navEntries = performance.getEntriesByType('navigation');
        if (navEntries && navEntries.length > 0) {
          loadTime = navEntries[0].loadEventEnd;
        }
      }
      if (loadTime === null && performance.timing) {
        // Fallback (deprecated API); guard against 0 values producing negatives
        const t = performance.timing;
        if (t.loadEventEnd && t.navigationStart) {
          loadTime = Math.max(0, t.loadEventEnd - t.navigationStart);
        }
      }
      if (loadTime !== null) {
        console.log(`Page load time: ${loadTime.toFixed(0)}ms`);
        if (loadTime > 3000) {
          console.warn('Slow page load detected');
        }
      }
    } catch (e) {
      console.warn('Performance timing not available', e);
    }
  });
  
  // Track API response times
  const originalFetch = window.fetch;
  window.fetch = function(...args) {
    const startTime = performance.now();
    return originalFetch.apply(this, args).then(response => {
      const endTime = performance.now();
      const duration = endTime - startTime;
      
      if (args[0].includes(API_BASE)) {
        console.log(`API call to ${args[0]} took ${duration.toFixed(2)}ms`);
        
        if (duration > 5000) {
          toast.warning('Slow Response', 'API response time is slower than expected');
        }
      }
      
      return response;
    });
  };
}

// Error boundary and global error handling
function initErrorHandling() {
  window.addEventListener('error', (e) => {
    console.error('Global error:', e.error);
    toast.error('Application Error', 'An unexpected error occurred. Please refresh the page.');
  });
  
  window.addEventListener('unhandledrejection', (e) => {
    console.error('Unhandled promise rejection:', e.reason);
    toast.error('Network Error', 'A network error occurred. Please check your connection.');
    e.preventDefault(); // Prevent default browser error handling
  });
}

// Initialization
async function initialize() {
  try {
    // Check if DOM is ready
    if (document.readyState === 'loading') {
      await new Promise(resolve => {
        document.addEventListener('DOMContentLoaded', resolve);
      });
    }
    
    // Initialize DOM elements first
    const hasElements = initializeElements();
    if (!hasElements) {
      console.error('Critical DOM elements not found');
      return;
    }

    // Remove header API status indicator (right-side online/models text)
    // Keeps the theme toggle but removes the status text block entirely.
    (function removeHeaderStatusIndicator(){
      const headerIndicator = document.querySelector('.app-header .status-indicator');
      if (headerIndicator) {
        headerIndicator.remove();
      }
      // Ensure code paths that try to update it no-op
      if (els && 'apiStatus' in els) {
        els.apiStatus = null;
      }
    })();
    
    // Initialize core systems
    initTheme();
    initErrorHandling();
    initPerformanceMonitoring();
    initScrollEffects();
    initKeyboardShortcuts();
    initEventListeners();
  // Ensure prediction results show only the placeholder before first prediction
  showPredictionPlaceholder();
  // Ensure features placeholder is visible until models are loaded
  showFeaturesPlaceholder();
    
    // Load initial data
  await pingApi();
  await fetchDatasets();
  await fetchModels();
    
    // Start auto-refresh
    startAutoRefresh();
    
    // Show welcome message
    setTimeout(() => {
      toast.info('Welcome', 'AI NIDS Dashboard v4.0 loaded successfully');
    }, 1000);
    
    console.log('AI NIDS Dashboard initialized successfully');
    
  } catch (error) {
    console.error('Initialization failed:', error);
    toast.error('Initialization Failed', 'Failed to initialize dashboard. Please refresh the page.');
  }
}

// Fetch list of available model artifacts for manual selection
async function fetchModels() {
  try {
    const data = await apiCall(`${API_BASE}/models`);
    availableModels = data.models || [];
    populateModelSelect(els.modelSelectSecondary);
    syncModelDropdowns();
  } catch (e) {
    console.error('Failed to fetch models:', e);
    toast.error('Models Error', e.message);
  }
}

function populateModelSelect(select) {
  if (!select) return;
  Array.from(select.options).slice(1).forEach(o => o.remove());
  availableModels.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m.model_key;
    const ds = m.dataset_guess ? `[${m.dataset_guess}] ` : '';
    const algorithmName = m.algorithm_guess || extractAlgorithmName(m.model_key);
    const size = m.size_mb ? ` (${m.size_mb}MB)` : '';
    opt.textContent = `${ds}${algorithmName}${size}`;
    select.appendChild(opt);
  });
  select.setAttribute('aria-live', 'polite');
}

function syncModelDropdowns() {
  // No synchronization needed since we only have one model selector
}

async function fetchDatasets() {
  try {
    const data = await apiCall(`${API_BASE}/datasets`);
    const select = els.datasetSelect;
    if (!select) return;
    select.innerHTML = '';
    data.datasets.forEach(d => {
      const opt = document.createElement('option');
      opt.value = d.key;
      opt.textContent = d.key.toUpperCase();
      select.appendChild(opt);
      datasetMeta[d.key] = { labels: d.labels, model_path: d.model_path };
    });
    if (!currentDataset && data.datasets.length) {
      currentDataset = data.datasets[0].key;
    }
    if (currentDataset) select.value = currentDataset;
  } catch (e) {
    console.error('Failed to fetch datasets:', e);
  }
}

// Start the application
initialize();
