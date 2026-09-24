/* ── main.js — EssayAI Frontend Logic ── */
'use strict';

// ─── Navbar scroll effect ────────────────────────────────────────────────────
window.addEventListener('scroll', () => {
  const nav = document.getElementById('navbar');
  nav.style.background = window.scrollY > 60
    ? 'rgba(7,11,20,0.97)' : 'rgba(7,11,20,0.8)';
});

// ─── Textarea counters ───────────────────────────────────────────────────────
const textarea    = document.getElementById('essayInput');
const wordCounter = document.getElementById('wordCounter');
const charCount   = document.getElementById('charCount');

textarea.addEventListener('input', updateCounters);

function updateCounters() {
  const text  = textarea.value;
  const words = text.trim() ? text.trim().split(/\s+/).length : 0;
  const chars = text.length;
  wordCounter.textContent = `${words} word${words !== 1 ? 's' : ''}`;
  charCount.textContent   = `${chars.toLocaleString()} / 10,000 chars`;
  charCount.style.color   = chars > 9500 ? '#f43f5e' : '';
}

// ─── Clear button ────────────────────────────────────────────────────────────
document.getElementById('clearBtn').addEventListener('click', () => {
  textarea.value = '';
  updateCounters();
  showPlaceholder();
});

// ─── Demo essay button ───────────────────────────────────────────────────────
document.getElementById('demoBtn').addEventListener('click', async () => {
  const btn = document.getElementById('demoBtn');
  btn.textContent = 'Loading…';
  try {
    const res = await fetch('/api/demo');
    if (res.ok) {
      const data = await res.json();
      textarea.value = data.essay;
      updateCounters();
    }
  } catch {
    // Fallback inline essay if server not running
    textarea.value = `Social media has fundamentally transformed the way teenagers communicate and learn. While platforms like Instagram offer unprecedented opportunities for self-expression, they also introduce significant challenges to adolescent mental health.

On one hand, social media democratizes information access. Students can now follow scientists and thought leaders directly, exposing themselves to diverse perspectives. A teenager in rural India can learn from MIT professors, fostering cultural empathy in ways previous generations could not imagine.

However, the algorithm-driven nature of these platforms creates echo chambers. Research by the American Psychological Association found that teenagers who spend more than three hours daily on social media report significantly higher rates of anxiety and depression.

In conclusion, social media's impact on teenagers is neither wholly positive nor negative but depends critically on how it is used. Educational institutions and parents must work together to establish healthy digital habits and teach media literacy.`;
    updateCounters();
  }
  btn.innerHTML = `<svg viewBox="0 0 20 20" fill="currentColor" style="width:13px;height:13px"><path fill-rule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 6a1 1 0 011-1h6a1 1 0 110 2H7a1 1 0 01-1-1zm1 3a1 1 0 100 2h6a1 1 0 100-2H7z" clip-rule="evenodd"/></svg> Demo Essay`;
});

// ─── Intersection observer for feature cards ─────────────────────────────────
const cards = document.querySelectorAll('.feature-card');
const observer = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      const delay = e.target.dataset.delay || 0;
      e.target.style.animationDelay = `${delay}ms`;
      e.target.style.animationPlayState = 'running';
    }
  });
}, { threshold: 0.1 });
cards.forEach(c => { c.style.animationPlayState = 'paused'; observer.observe(c); });

// ─── State helpers ───────────────────────────────────────────────────────────
function showPlaceholder() {
  document.getElementById('resultsPlaceholder').style.display = 'flex';
  document.getElementById('resultsLoading').style.display     = 'none';
  document.getElementById('resultsContent').style.display     = 'none';
}
function showLoading() {
  document.getElementById('resultsPlaceholder').style.display = 'none';
  document.getElementById('resultsLoading').style.display     = 'flex';
  document.getElementById('resultsContent').style.display     = 'none';
}
function showResults() {
  document.getElementById('resultsPlaceholder').style.display = 'none';
  document.getElementById('resultsLoading').style.display     = 'none';
  document.getElementById('resultsContent').style.display     = 'flex';
}

// ─── Loading animation steps ─────────────────────────────────────────────────
function animateLoadingSteps() {
  const steps = ['lStep1','lStep2','lStep3','lStep4'];
  const bar   = document.getElementById('loadingBar');
  let current = 0;

  steps.forEach(id => {
    const el = document.getElementById(id);
    el.className = 'loading-step';
  });
  bar.style.width = '0%';

  const tick = setInterval(() => {
    if (current > 0) {
      const prev = document.getElementById(steps[current - 1]);
      prev.className = 'loading-step done';
      prev.textContent = '✓ ' + prev.textContent.replace('✓ ','');
    }
    if (current < steps.length) {
      document.getElementById(steps[current]).className = 'loading-step active';
      bar.style.width = `${((current + 1) / steps.length) * 100}%`;
      current++;
    } else {
      clearInterval(tick);
    }
  }, 700);

  return () => clearInterval(tick);
}

// ─── Score ring animation ────────────────────────────────────────────────────
function animateScoreRing(score) {
  const arc       = document.getElementById('scoreArc');
  const textMain  = document.getElementById('scoreTextMain');
  const scoreLabel= document.getElementById('scoreLabel');
  const CIRCUMFERENCE = 2 * Math.PI * 85; // r=85

  const pct     = (score - 1) / 5;        // normalise 1-6 → 0-1
  const offset  = CIRCUMFERENCE * (1 - pct);

  // Reset first
  arc.style.strokeDashoffset = CIRCUMFERENCE;
  setTimeout(() => { arc.style.strokeDashoffset = offset; }, 50);

  // Animate number counting up
  let current = 0;
  const step  = score / 40;
  const counter = setInterval(() => {
    current = Math.min(current + step, score);
    textMain.textContent = current.toFixed(1);
    if (current >= score) {
      textMain.textContent = score.toFixed(1);
      clearInterval(counter);
    }
  }, 40);

  // Color the arc based on score
  const grad = document.getElementById('scoreGrad');
  if (score >= 5) {
    grad.children[0].setAttribute('stop-color', '#10b981');
    grad.children[1].setAttribute('stop-color', '#00d4ff');
    scoreLabel.textContent = 'Excellent';
    scoreLabel.setAttribute('fill', '#10b981');
  } else if (score >= 4) {
    grad.children[0].setAttribute('stop-color', '#00d4ff');
    grad.children[1].setAttribute('stop-color', '#fbbf24');
    scoreLabel.textContent = 'Good';
    scoreLabel.setAttribute('fill', '#fbbf24');
  } else if (score >= 3) {
    grad.children[0].setAttribute('stop-color', '#fbbf24');
    grad.children[1].setAttribute('stop-color', '#f59e0b');
    scoreLabel.textContent = 'Average';
    scoreLabel.setAttribute('fill', '#f59e0b');
  } else {
    grad.children[0].setAttribute('stop-color', '#f43f5e');
    grad.children[1].setAttribute('stop-color', '#f97316');
    scoreLabel.textContent = 'Below Average';
    scoreLabel.setAttribute('fill', '#f43f5e');
  }
}

// ─── Score bar breakdown ─────────────────────────────────────────────────────
function renderScoreBars(score) {
  const group = document.getElementById('scoreBarGroup');
  const pct = Math.max(0, Math.min((score - 1) / 5, 1));
  group.innerHTML = `
    <div class="score-bar-row">
      <span class="score-bar-label">Score range</span>
      <div class="score-bar-track">
        <div class="score-bar-fill" style="width:0%;background:#00d4ff"
             data-target="${Math.round(pct * 100)}"></div>
      </div>
      <span style="font-size:0.7rem;color:#9ca3af;width:30px;text-align:right">${score.toFixed(1)}</span>
    </div>
  `;

  // Animate bars in
  setTimeout(() => {
    group.querySelectorAll('.score-bar-fill').forEach(bar => {
      bar.style.width = bar.dataset.target + '%';
    });
  }, 200);
}

// ─── Feedback rendering ───────────────────────────────────────────────────────
function getFeedbackClass(text) {
  if (text.startsWith('✅'))  return 'positive';
  if (text.startsWith('⚠'))   return 'critical';
  return 'warning';
}

function renderFeedback(feedbackList) {
  const container = document.getElementById('feedbackList');
  const count     = document.getElementById('feedbackCount');

  container.replaceChildren(...feedbackList.map((fb, i) => {
    const item = document.createElement('div');
    item.className = `feedback-item ${getFeedbackClass(fb)}`;
    item.style.animationDelay = `${i * 120}ms`;
    item.textContent = fb;
    return item;
  }));

  count.textContent = `${feedbackList.length} suggestion${feedbackList.length !== 1 ? 's' : ''}`;
}

// ─── Main scoring function ────────────────────────────────────────────────────
async function scoreEssay() {
  const essay = textarea.value.trim();
  const btn   = document.getElementById('scoreBtn');
  const btnTxt= document.getElementById('scoreBtnText');

  if (!essay || essay.length < 50) {
    showValidationError('Essay must be at least 50 characters long.');
    return;
  }

  // UI → Loading
  btn.disabled = true;
  btnTxt.textContent = 'Analyzing…';
  showLoading();
  const stopLoading = animateLoadingSteps();

  try {
    const res  = await fetch('/api/score', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ essay })
    });

    const data = await res.json();
    stopLoading();

    if (!res.ok || data.error) {
      showError(data.error || 'Unknown error from server');
      return;
    }

    // Ensure loading shows for at least 2.8s for effect
    await new Promise(r => setTimeout(r, 300));

    // Render results
    showResults();
    const score = Number(data.score);
    if (!Number.isFinite(score) || score < 1 || score > 6) {
      showError('The scoring service returned an invalid score.');
      return;
    }
    animateScoreRing(score);
    renderScoreBars(score);
    renderFeedback(data.feedback || []);

    document.getElementById('confidenceValue').textContent = data.confidence || '—';
    document.getElementById('processingTime').textContent  = data.time_ms
      ? `${data.time_ms}ms` : '—';

    document.querySelectorAll('.demo-note').forEach(note => note.remove());
    if (data.demo_mode) {
      const demoNote = document.createElement('div');
      demoNote.className = 'demo-note';
      demoNote.style.cssText = 'font-size:0.72rem;color:#f59e0b;padding:8px 14px;background:rgba(251,191,36,0.08);border-radius:6px;border:1px solid rgba(251,191,36,0.2);';
      demoNote.textContent = `⚠ Demo mode: ${data.demo_reason || 'no trained model is available'}. This score is heuristic, not a model prediction.`;
      document.getElementById('resultsContent').prepend(demoNote);
    }

  } catch (err) {
    stopLoading();
    showError(`Connection failed: ${err.message}. Make sure the Flask server is running (python app.py).`);
  } finally {
    btn.disabled = false;
    btnTxt.textContent = 'Score My Essay';
  }
}

function showValidationError(msg) {
  textarea.style.border = '1px solid #f43f5e';
  setTimeout(() => { textarea.style.border = ''; }, 2000);

  const existing = document.getElementById('validationError');
  if (existing) existing.remove();
  const el = document.createElement('div');
  el.id = 'validationError';
  el.style.cssText = 'color:#f43f5e;font-size:0.8rem;padding:8px 20px;';
  el.textContent = '⚠ ' + msg;
  document.querySelector('.panel-card').insertBefore(el, document.getElementById('scoreBtn'));
  setTimeout(() => el.remove(), 3000);
}

function showError(msg) {
  showResults();
  const content = document.getElementById('resultsContent');
  const panel = document.createElement('div');
  panel.style.cssText = 'padding:40px;text-align:center;';
  const icon = document.createElement('div');
  icon.style.cssText = 'font-size:2.5rem;margin-bottom:16px;';
  icon.textContent = '⚠️';
  const heading = document.createElement('h3');
  heading.style.cssText = 'color:#f43f5e;margin-bottom:8px;';
  heading.textContent = 'Scoring Failed';
  const message = document.createElement('p');
  message.style.cssText = 'color:#9ca3af;font-size:0.85rem;line-height:1.6;';
  message.textContent = msg;
  panel.append(icon, heading, message);
  content.replaceChildren(panel);
}

// ─── Smooth scroll for CTA ────────────────────────────────────────────────────
document.getElementById('hero-cta-btn').addEventListener('click', (e) => {
  e.preventDefault();
  document.getElementById('scorer').scrollIntoView({ behavior: 'smooth' });
});
