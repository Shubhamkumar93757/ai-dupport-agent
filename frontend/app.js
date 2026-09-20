/**
 * app.js — Frontend logic for the AI Support Agent Dashboard
 *
 * Handles:
 *  - Tab switching (Pipeline ↔ Evaluations)
 *  - POST /api/analyze → render pipeline results
 *  - POST /api/evaluate → render batch evaluation results
 *  - Editable draft reply with approve/escalate actions
 *  - Toast notifications
 *  - Loading states with pipeline step indicators
 */

// ── Configuration ─────────────────────────────────────────────
const API_BASE = window.location.origin;

// ── State ─────────────────────────────────────────────────────
let currentResult = null;

// ── Tab Switching ─────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    // Deactivate all
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

    // Activate clicked
    btn.classList.add('active');
    const tabId = btn.dataset.tab + 'Tab';
    document.getElementById(tabId).classList.add('active');
  });
});

// ── Health Check on Load ──────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    const data = await res.json();

    const dot = document.getElementById('statusDot');
    const text = document.getElementById('statusText');

    if (data.status === 'ok' && data.gemini_key_set && data.chroma_db_exists) {
      dot.style.background = 'var(--accent-green)';
      text.textContent = 'All systems operational';
    } else {
      dot.style.background = 'var(--accent-orange)';
      const issues = [];
      if (!data.gemini_key_set) issues.push('No Gemini key');
      if (!data.chroma_db_exists) issues.push('No ChromaDB');
      if (!data.groq_configured) issues.push('No Groq key');
      text.textContent = issues.join(' · ') || 'Partial';
    }
  } catch (e) {
    document.getElementById('statusDot').style.background = 'var(--accent-red)';
    document.getElementById('statusText').textContent = 'API unreachable';
  }
}

// ── Analyze Message ───────────────────────────────────────────
async function analyzeMessage() {
  const input = document.getElementById('messageInput');
  const message = input.value.trim();

  if (!message) {
    showToast('Please enter a customer message.', 'error');
    return;
  }

  // Show loading, hide results
  setLoading('pipeline', true);
  document.getElementById('emptyState').style.display = 'none';
  document.getElementById('pipelineResults').classList.remove('active');
  document.getElementById('pipelineSteps').style.display = 'flex';

  // Animate pipeline steps
  animateSteps();

  try {
    const res = await fetch(`${API_BASE}/api/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    currentResult = data;

    // Mark all steps done
    document.querySelectorAll('.pipeline-step').forEach(s => {
      s.classList.remove('active');
      s.classList.add('done');
    });

    renderResults(data);
    showToast('Pipeline completed successfully!', 'success');

  } catch (e) {
    showToast(`Error: ${e.message}`, 'error');
    document.getElementById('emptyState').style.display = 'block';
  } finally {
    setLoading('pipeline', false);
  }
}

// ── Render Pipeline Results ───────────────────────────────────
function renderResults(data) {
  const container = document.getElementById('pipelineResults');

  // Intent breakdown
  renderIntents(data.intent_breakdown || []);

  // Retrieved cases
  renderCases(data.retrieved_cases || []);

  // Draft reply
  const draftEl = document.getElementById('draftReply');
  draftEl.value = data.draft_reply || '';

  // Escalation banner
  renderEscalation(data.escalation || {});

  // Enable buttons
  document.getElementById('approveBtn').disabled = false;
  document.getElementById('escalateBtn').disabled = false;
  document.getElementById('regenerateBtn').disabled = false;

  container.classList.add('active');
}

function renderIntents(intents) {
  const list = document.getElementById('intentList');
  const ranks = ['Primary', 'Secondary', 'Tertiary'];
  const barClasses = ['primary', 'secondary', 'tertiary'];
  const colors = ['var(--accent-blue)', 'var(--accent-orange)', 'var(--accent-purple)'];

  list.innerHTML = intents.map((item, i) => {
    const pct = Math.round(item.confidence * 100);
    return `
      <div class="intent-item">
        <div class="intent-item__header">
          <div>
            <span class="intent-item__rank">${ranks[i] || `#${i+1}`}</span>
            <span class="intent-item__name">${item.intent}</span>
          </div>
          <span class="intent-item__score" style="color: ${colors[i]}">${pct}%</span>
        </div>
        <div class="intent-bar">
          <div class="intent-bar__fill intent-bar__fill--${barClasses[i] || 'tertiary'}"
               data-width="${pct}%"
               style="width: 0"></div>
        </div>
      </div>
    `;
  }).join('');

  // Animate bars after render
  requestAnimationFrame(() => {
    setTimeout(() => {
      list.querySelectorAll('.intent-bar__fill').forEach(bar => {
        bar.style.width = bar.dataset.width;
      });
    }, 100);
  });
}

function renderCases(cases) {
  const list = document.getElementById('caseList');
  document.getElementById('retrievalCount').textContent = cases.length;

  if (cases.length === 0) {
    list.innerHTML = '<div class="text-sm text-muted" style="padding: 16px;">No similar cases found.</div>';
    return;
  }

  list.innerHTML = cases.map((c, i) => {
    const simPct = Math.round(c.similarity * 100);
    return `
      <div class="case-item" onclick="this.classList.toggle('expanded')">
        <div class="case-item__header">
          <span class="case-item__rank">CASE ${i + 1}</span>
          <span class="case-item__similarity">${simPct}% match</span>
        </div>
        <div class="case-item__label">Customer</div>
        <div class="case-item__text">${truncate(c.customer_problem, 120)}</div>
        <div class="case-item__details">
          <div class="divider"></div>
          <div class="case-item__label">Full Customer Message</div>
          <div class="case-item__text">${c.customer_problem}</div>
          <div class="case-item__label" style="margin-top: 8px;">Brand Resolution</div>
          <div class="case-item__text case-item__text--resolution">${c.brand_resolution}</div>
        </div>
      </div>
    `;
  }).join('');
}

function renderEscalation(esc) {
  const banner = document.getElementById('escalationBanner');
  const icon = document.getElementById('escalationIcon');
  const label = document.getElementById('escalationLabel');
  const reason = document.getElementById('escalationReason');

  if (!esc.decision) {
    banner.style.display = 'none';
    return;
  }

  banner.style.display = 'flex';

  if (esc.decision === 'auto') {
    banner.className = 'escalation-banner escalation-banner--auto';
    icon.textContent = '✅';
    label.textContent = 'AUTO-SEND RECOMMENDED';
  } else {
    banner.className = 'escalation-banner escalation-banner--escalate';
    icon.textContent = '🚨';
    label.textContent = 'ESCALATION RECOMMENDED';
  }

  reason.textContent = esc.reason || '';
}

// ── Approve / Escalate Handlers ───────────────────────────────
function handleApprove() {
  const draftEl = document.getElementById('draftReply');
  const reply = draftEl.value.trim();

  if (!reply) {
    showToast('Draft reply is empty.', 'error');
    return;
  }

  document.getElementById('approveBtn').disabled = true;
  document.getElementById('escalateBtn').disabled = true;
  document.getElementById('draftBadge').textContent = 'APPROVED';
  document.getElementById('draftBadge').className = 'card__badge badge-completed';

  showToast('Reply approved and ready to send!', 'success');
}

function handleEscalate() {
  document.getElementById('approveBtn').disabled = true;
  document.getElementById('escalateBtn').disabled = true;
  document.getElementById('draftBadge').textContent = 'ESCALATED';
  document.getElementById('draftBadge').className = 'card__badge badge-error';

  showToast('Case escalated to human agent.', 'info');
}

// ── Run Evaluation ────────────────────────────────────────────
async function runEvaluation() {
  const numSamples = parseInt(document.getElementById('sampleCount').value);

  setLoading('eval', true);
  document.getElementById('evalEmptyState').style.display = 'none';
  document.getElementById('evalSummary').style.display = 'none';

  // Clear previous results (keep the empty state div)
  const resultsContainer = document.getElementById('evalResults');
  resultsContainer.querySelectorAll('.eval-case').forEach(el => el.remove());

  try {
    const res = await fetch(`${API_BASE}/api/evaluate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ num_samples: numSamples }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    renderEvalResults(data);
    showToast(`Evaluation complete! Avg score: ${data.avg_score}/100`, 'success');

  } catch (e) {
    showToast(`Evaluation failed: ${e.message}`, 'error');
    document.getElementById('evalEmptyState').style.display = 'block';
  } finally {
    setLoading('eval', false);
  }
}

function renderEvalResults(data) {
  // Summary stats
  document.getElementById('evalSummary').style.display = 'flex';
  document.getElementById('avgScore').textContent = `${data.avg_score}/100`;
  document.getElementById('totalCases').textContent = data.total;
  document.getElementById('successCases').textContent = data.successful;

  // Color the avg score
  const scoreEl = document.getElementById('avgScore');
  if (data.avg_score >= 75) scoreEl.style.color = 'var(--accent-green)';
  else if (data.avg_score >= 50) scoreEl.style.color = 'var(--accent-orange)';
  else scoreEl.style.color = 'var(--accent-red)';

  // Individual results
  const container = document.getElementById('evalResults');
  document.getElementById('evalEmptyState').style.display = 'none';

  data.results.forEach((r, i) => {
    const scoreClass = r.score >= 75 ? 'high' : r.score >= 50 ? 'mid' : 'low';

    const caseEl = document.createElement('div');
    caseEl.className = 'eval-case';
    caseEl.innerHTML = `
      <div class="eval-case__header">
        <span class="eval-case__number">CASE ${i + 1}${r.intent ? ` · ${r.intent}` : ''}</span>
        ${r.error
          ? '<span class="card__badge badge-error">ERROR</span>'
          : `<span class="eval-case__score eval-case__score--${scoreClass}">${r.score}/100</span>`
        }
      </div>

      <div class="eval-case__section">
        <div class="eval-case__section-label">Customer Message</div>
        <div class="eval-case__section-text">${truncate(r.customer_message, 200)}</div>
      </div>

      ${r.error ? `
        <div class="eval-case__section">
          <div class="eval-case__section-label">Error</div>
          <div class="eval-case__section-text" style="color: var(--accent-red);">${r.error}</div>
        </div>
      ` : `
        <div class="eval-case__section">
          <div class="eval-case__section-label">AI Draft</div>
          <div class="eval-case__section-text">${r.draft_reply}</div>
        </div>

        <div class="eval-case__section">
          <div class="eval-case__section-label">Actual Brand Reply</div>
          <div class="eval-case__section-text" style="color: var(--accent-green);">${r.actual_reply}</div>
        </div>

        <div class="eval-case__reasoning">${r.reasoning}</div>
      `}
    `;

    container.appendChild(caseEl);
  });
}

// ── Pipeline Step Animation ───────────────────────────────────
function animateSteps() {
  const steps = document.querySelectorAll('.pipeline-step');
  const loadingStep = document.getElementById('loadingStep');
  const labels = ['Classifying intent...', 'Retrieving cases...', 'Generating reply...', 'Deciding escalation...'];

  steps.forEach(s => { s.classList.remove('active', 'done'); });

  let current = 0;
  const interval = setInterval(() => {
    if (current > 0) steps[current - 1].classList.replace('active', 'done');
    if (current < steps.length) {
      steps[current].classList.add('active');
      loadingStep.textContent = labels[current];
      current++;
    } else {
      clearInterval(interval);
    }
  }, 1500);

  // Store interval so we can clear it if results arrive early
  window._stepInterval = interval;
}

// ── Loading State ─────────────────────────────────────────────
function setLoading(section, isLoading) {
  if (section === 'pipeline') {
    const overlay = document.getElementById('pipelineLoading');
    const btn = document.getElementById('analyzeBtn');

    if (isLoading) {
      overlay.classList.add('active');
      btn.disabled = true;
      btn.innerHTML = '<div class="spinner" style="width:16px;height:16px;border-width:2px;"></div> Analyzing...';
    } else {
      overlay.classList.remove('active');
      btn.disabled = false;
      btn.innerHTML = '⚡ Analyze';
      if (window._stepInterval) clearInterval(window._stepInterval);
    }
  } else if (section === 'eval') {
    const overlay = document.getElementById('evalLoading');
    const btn = document.getElementById('evalBtn');

    if (isLoading) {
      overlay.classList.add('active');
      btn.disabled = true;
      btn.innerHTML = '<div class="spinner" style="width:16px;height:16px;border-width:2px;"></div> Evaluating...';
    } else {
      overlay.classList.remove('active');
      btn.disabled = false;
      btn.innerHTML = '▶ Run Evaluation';
    }
  }
}

// ── Toast Notifications ───────────────────────────────────────
function showToast(message, type = 'info') {
  const container = document.getElementById('toastContainer');

  const icons = { success: '✅', error: '❌', info: 'ℹ️' };

  const toast = document.createElement('div');
  toast.className = `toast toast--${type}`;
  toast.innerHTML = `<span>${icons[type] || ''}</span> <span>${message}</span>`;

  container.appendChild(toast);

  // Auto-remove after 4s
  setTimeout(() => {
    toast.classList.add('toast-exit');
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

// ── Utilities ─────────────────────────────────────────────────
function truncate(str, maxLen) {
  if (!str) return '';
  return str.length > maxLen ? str.substring(0, maxLen) + '...' : str;
}

// Allow Enter key to submit (Ctrl+Enter or Cmd+Enter)
document.getElementById('messageInput').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    analyzeMessage();
  }
});

// ── Init ──────────────────────────────────────────────────────
checkHealth();
