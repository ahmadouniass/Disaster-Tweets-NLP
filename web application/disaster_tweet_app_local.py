"""
Disaster Tweet Detection System — LOCAL VERSION
Design: Premium Custom HTML Injector
Connection: Local FastAPI (http://localhost:8000)
"""

import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(
    page_title="Disaster Tweet Detection",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Cache tout le chrome Streamlit pour une immersion totale
st.markdown("""
<style>
#MainMenu, footer, header, .stDeployButton,
[data-testid="stToolbar"], [data-testid="stDecoration"],
.viewerBadge_container__r5tak { display: none !important; }
body, .stApp { background: #07090f !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stMain"] > div { padding: 0 !important; }
iframe { display: block; border: none; }
</style>
""", unsafe_allow_html=True)

# ── App HTML complète adaptée pour le LOCAL ──────────────────────────────────
HTML_APP = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:opsz,wght@9..40,300;0,9..40,400;0,9..40,500&display=swap" rel="stylesheet"/>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#07090f;--bg2:#0c1018;--bg3:#111926;--bg4:#162030;
  --border:#1c2a3e;--border2:#253a55;
  --blue:#3b82f6;--blue2:#60a5fa;--cyan:#06b6d4;
  --red:#ef4444;--red2:#fca5a5;--green:#22c55e;--green2:#86efac;
  --orange:#f97316;
  --txt:#eef2f8;--txt2:#8ba3c1;--txt3:#3f5570;
  --ff:'Syne',sans-serif;--fb:'DM Sans',sans-serif;
}
html{scroll-behavior:smooth}
body{
  background:var(--bg);color:var(--txt);
  font-family:var(--fb);font-size:15px;line-height:1.6;
  overflow-x:hidden;
}
::-webkit-scrollbar{width:4px}
::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:var(--border2);border-radius:4px}

/* ─── HERO ──────────────────────────────────── */
.hero{
  position:relative;min-height:370px;
  display:flex;flex-direction:column;align-items:center;
  justify-content:center;text-align:center;
  padding:88px 24px 68px;overflow:hidden;
}
.hero-bg{
  position:absolute;inset:0;z-index:0;
  background:
    radial-gradient(ellipse 90% 70% at 50% -5%,rgba(59,130,246,.18) 0%,transparent 65%),
    radial-gradient(ellipse 50% 40% at 85% 40%,rgba(6,182,212,.07) 0%,transparent 60%);
}
.hero-grid{
  position:absolute;inset:0;z-index:0;
  background-image:
    linear-gradient(rgba(59,130,246,.035) 1px,transparent 1px),
    linear-gradient(90deg,rgba(59,130,246,.035) 1px,transparent 1px);
  background-size:52px 52px;
  animation:gridScroll 24s linear infinite;
}
@keyframes gridScroll{to{background-position:52px 52px}}
.hero-scan{
  position:absolute;left:0;right:0;height:2px;
  background:linear-gradient(90deg,transparent,var(--cyan),transparent);
  opacity:0;animation:scanDown 6s ease-in-out infinite;z-index:1;
}
@keyframes scanDown{
  0%{top:0;opacity:0}8%{opacity:.5}90%{opacity:.4}100%{top:100%;opacity:0}
}
.hero-inner{position:relative;z-index:2}
.badge{
  display:inline-flex;align-items:center;gap:8px;
  background:rgba(59,130,246,.1);border:1px solid rgba(59,130,246,.28);
  border-radius:100px;padding:5px 18px 5px 10px;
  font-family:var(--ff);font-size:11px;font-weight:700;
  letter-spacing:.14em;text-transform:uppercase;color:var(--cyan);
  margin-bottom:24px;user-select:none;
}
.badge-dot{
  width:7px;height:7px;border-radius:50%;background:var(--cyan);
  animation:blink 2.2s ease-in-out infinite;
}
@keyframes blink{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.3;transform:scale(.6)}}
h1{
  font-family:var(--ff);font-size:clamp(34px,5.5vw,62px);
  font-weight:800;line-height:1.05;letter-spacing:-.025em;margin-bottom:18px;
}
h1 .g{
  background:linear-gradient(120deg,var(--blue),var(--cyan));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}
.hero-sub{
  font-size:clamp(14px,1.8vw,17px);font-weight:300;color:var(--txt2);
  max-width:580px;margin:0 auto 34px;line-height:1.75;
}
.pills{display:flex;flex-wrap:wrap;gap:10px;justify-content:center}
.pill{
  display:flex;align-items:center;gap:7px;
  background:rgba(255,255,255,.04);border:1px solid var(--border);
  border-radius:100px;padding:6px 16px;font-size:12px;font-weight:500;color:var(--txt2);
}

/* ─── LAYOUT ────────────────────────────────── */
.container{max-width:840px;margin:0 auto;padding:0 24px 100px}
.slabel{
  font-family:var(--ff);font-size:11px;font-weight:700;
  letter-spacing:.15em;text-transform:uppercase;color:var(--blue2);
  display:flex;align-items:center;gap:9px;margin-bottom:12px;
}
.slabel::before{content:'';display:block;width:22px;height:2px;background:var(--blue);border-radius:2px}

/* ─── CARD ──────────────────────────────────── */
.card{
  background:var(--bg3);border:1px solid var(--border);
  border-radius:24px;padding:40px;
  position:relative;overflow:hidden;transition:border-color .3s;
}
.card::before{
  content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,var(--blue),var(--cyan),transparent);opacity:.45;
}
.card:hover{border-color:var(--border2)}
.card-title{font-family:var(--ff);font-size:20px;font-weight:700;margin-bottom:6px}
.card-desc{font-size:13px;color:var(--txt2);margin-bottom:28px;line-height:1.65}

/* ─── TEXTAREA ──────────────────────────────── */
textarea{
  width:100%;background:var(--bg2);border:1.5px solid var(--border2);
  border-radius:14px;color:var(--txt);font-family:var(--fb);
  font-size:15px;line-height:1.7;padding:18px 20px;resize:vertical;
  min-height:130px;transition:border-color .2s,box-shadow .2s;outline:none;
}
textarea::placeholder{color:var(--txt3)}
textarea:focus{border-color:var(--blue);box-shadow:0 0 0 4px rgba(59,130,246,.11)}
.char-counter{text-align:right;font-size:12px;color:var(--txt3);margin-top:6px;transition:color .2s}
.char-counter.warn{color:var(--orange)}

/* ─── BUTTON ────────────────────────────────── */
.btn{
  width:100%;
  background:linear-gradient(135deg,var(--blue) 0%,#1d4ed8 100%);
  color:#fff;border:none;border-radius:14px;padding:17px 32px;
  font-family:var(--ff);font-size:15px;font-weight:700;letter-spacing:.04em;
  cursor:pointer;margin-top:18px;
  transition:transform .22s,box-shadow .22s;
  box-shadow:0 4px 28px rgba(59,130,246,.28);
  display:flex;align-items:center;justify-content:center;gap:10px;
}
.btn:hover{transform:translateY(-3px);box-shadow:0 10px 36px rgba(59,130,246,.42)}
.btn:active{transform:translateY(-1px)}
.btn:disabled{opacity:.5;cursor:not-allowed;transform:none;box-shadow:none}
.spin{
  width:18px;height:18px;border:2.5px solid rgba(255,255,255,.3);
  border-top-color:#fff;border-radius:50%;animation:spin .7s linear infinite;
}
@keyframes spin{to{transform:rotate(360deg)}}

/* ─── EXAMPLES ──────────────────────────────── */
.ex-section{margin:34px 0}
.ex-label{
  font-family:var(--ff);font-size:12px;font-weight:700;
  color:var(--txt3);text-transform:uppercase;letter-spacing:.1em;
  margin-bottom:14px;display:flex;align-items:center;gap:8px;
}
.ex-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.ex-chip{
  background:rgba(255,255,255,.025);border:1px solid var(--border);
  border-radius:14px;padding:13px 16px;font-size:12.5px;color:var(--txt2);
  line-height:1.55;cursor:pointer;transition:all .2s;
  display:flex;align-items:flex-start;gap:10px;
  user-select:none;text-align:left;width:100%;
}
.ex-chip:hover{
  background:rgba(59,130,246,.07);border-color:rgba(59,130,246,.3);
  color:var(--txt);transform:translateY(-2px);
}
.ex-chip span{font-size:16px;flex-shrink:0;margin-top:1px}

/* ─── RESULT ────────────────────────────────── */
#result-zone{margin-top:28px}
.result-card{
  border-radius:24px;padding:38px;
  position:relative;overflow:hidden;
  animation:revealUp .5s cubic-bezier(.16,1,.3,1) forwards;
}
@keyframes revealUp{
  from{opacity:0;transform:translateY(24px) scale(.97)}
  to{opacity:1;transform:translateY(0) scale(1)}
}
.result-card.disaster{
  background:linear-gradient(135deg,rgba(239,68,68,.1),rgba(239,68,68,.04));
  border:1px solid rgba(239,68,68,.32);
}
.result-card.safe{
  background:linear-gradient(135deg,rgba(34,197,94,.09),rgba(34,197,94,.03));
  border:1px solid rgba(34,197,94,.32);
}
.result-card::after{
  content:'';position:absolute;top:-50px;right:-50px;
  width:180px;height:180px;border-radius:50%;pointer-events:none;
}
.result-card.disaster::after{background:radial-gradient(circle,rgba(239,68,68,.13),transparent 70%)}
.result-card.safe::after{background:radial-gradient(circle,rgba(34,197,94,.1),transparent 70%)}
.result-header{display:flex;align-items:center;gap:18px;margin-bottom:24px}
.result-emoji{
  width:60px;height:60px;border-radius:16px;
  display:flex;align-items:center;justify-content:center;font-size:30px;flex-shrink:0;
}
.disaster .result-emoji{background:rgba(239,68,68,.14)}
.safe .result-emoji{background:rgba(34,197,94,.12)}
.result-title{font-family:var(--ff);font-size:23px;font-weight:800;line-height:1.1;margin-bottom:5px}
.disaster .result-title{color:var(--red2)}
.safe .result-title{color:var(--green2)}
.result-desc{font-size:13px;color:var(--txt2);line-height:1.6}
.conf-wrap{margin-top:4px}
.conf-row{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}
.conf-lbl{font-size:11px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--txt2)}
.conf-val{font-family:var(--ff);font-size:22px;font-weight:800}
.disaster .conf-val{color:var(--red)}
.safe .conf-val{color:var(--green)}
.track{height:11px;background:rgba(255,255,255,.05);border-radius:100px;overflow:hidden;border:1px solid rgba(255,255,255,.04)}
.fill{height:100%;border-radius:100px;transition:width 1.1s cubic-bezier(.16,1,.3,1)}
.disaster .fill{background:linear-gradient(90deg,#991b1b,var(--red),#f87171)}
.safe .fill{background:linear-gradient(90deg,#166534,var(--green),#4ade80)}
.ticks{display:flex;justify-content:space-between;margin-top:7px;font-size:10.5px;color:var(--txt3);font-weight:500}

/* ─── DIVIDER ───────────────────────────────── */
.divider{height:1px;background:linear-gradient(90deg,transparent,var(--border),transparent);margin:52px 0}

/* ─── ABOUT ─────────────────────────────────── */
.about{
  background:var(--bg3);border:1px solid var(--border);
  border-radius:24px;padding:44px 40px;position:relative;overflow:hidden;
}
.about::before{
  content:'';position:absolute;bottom:-30px;right:-30px;
  width:240px;height:240px;
  background:radial-gradient(circle,rgba(59,130,246,.06),transparent 70%);
  pointer-events:none;
}
.about-title{font-family:var(--ff);font-size:23px;font-weight:800;margin-bottom:10px}
.about-desc{font-size:14px;color:var(--txt2);line-height:1.75;margin-bottom:34px;max-width:680px}
.feat-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}
.feat{
  background:var(--bg2);border:1px solid var(--border);
  border-radius:16px;padding:20px 22px;transition:border-color .2s,transform .2s;
}
.feat:hover{border-color:var(--border2);transform:translateY(-3px)}
.feat-ico{font-size:24px;margin-bottom:12px;display:block}
.feat-name{font-family:var(--ff);font-size:14.5px;font-weight:700;color:var(--txt);margin-bottom:5px}
.feat-desc{font-size:12.5px;color:var(--txt3);line-height:1.55}
.tech-row{display:flex;flex-wrap:wrap;gap:9px;margin-top:30px}
.tech{background:rgba(255,255,255,.035);border:1px solid var(--border2);border-radius:100px;padding:5px 15px;font-size:12px;font-weight:600;color:var(--txt2)}
.tech.hi{background:rgba(59,130,246,.1);border-color:rgba(59,130,246,.38);color:#93c5fd}

/* ─── TOAST ─────────────────────────────────── */
#toast{
  position:fixed;bottom:28px;left:50%;transform:translateX(-50%) translateY(80px);
  background:var(--bg4);border:1px solid var(--border2);border-radius:14px;
  padding:14px 24px;font-size:14px;font-weight:500;
  box-shadow:0 8px 32px rgba(0,0,0,.5);opacity:0;
  transition:all .35s cubic-bezier(.16,1,.3,1);pointer-events:none;z-index:999;
}
#toast.show{opacity:1;transform:translateX(-50%) translateY(0)}

/* ─── FOOTER ────────────────────────────────── */
footer{border-top:1px solid var(--border);background:var(--bg2);padding:32px 24px;text-align:center}
.foot-title{font-family:var(--ff);font-size:13px;font-weight:700;color:var(--txt2);letter-spacing:.06em;margin-bottom:6px}
.foot-sub{font-size:11.5px;color:var(--txt3);letter-spacing:.04em}
.foot-dot{color:var(--blue);margin:0 7px}

@media(max-width:600px){
  .feat-grid,.ex-grid{grid-template-columns:1fr}
  .card,.about{padding:24px 20px}
  .hero{padding:56px 18px 48px}
  .container{padding:0 16px 70px}
}
</style>
</head>
<body>

<header class="hero">
  <div class="hero-bg"></div>
  <div class="hero-grid"></div>
  <div class="hero-scan"></div>
  <div class="hero-inner">
    <div class="badge"><span class="badge-dot"></span>NLP · Real-Time Classification</div>
    <h1>Disaster Tweet<br><span class="g">Detection System</span></h1>
    <p class="hero-sub">
      A fine-tuned BERT / DistilBERT model that instantly classifies whether a tweet
      describes a real-world emergency — powered by Transformers &amp; deployed on Render.
    </p>
    <div class="pills">
      <div class="pill">🤗 HuggingFace Transformers</div>
      <div class="pill">⚡ Real-Time Inference</div>
      <div class="pill">☁️ Render API</div>
      <div class="pill">🎯 Binary NLP Classification</div>
    </div>
  </div>
</header>

<main class="container">
  <div class="card">
    <div class="slabel">Analysis</div>
    <div class="card-title">Paste or type a tweet below</div>
    <div class="card-desc">
      The model will analyse the text and determine whether it describes a genuine
      disaster event, returning a prediction label and a confidence score.
    </div>
    <textarea id="tweetInput" maxlength="280"
      placeholder="e.g. 'Wildfires destroying thousands of acres, emergency evacuations underway…'"></textarea>
    <div class="char-counter" id="charCount">0 / 280</div>
    <button class="btn" id="analyzeBtn" onclick="analyze()">
      <span id="btnIcon">🔍</span>
      <span id="btnText">Analyze Tweet</span>
    </button>
  </div>

  <div class="ex-section">
    <div class="ex-label">💡 &nbsp;Try an example</div>
    <div class="ex-grid" id="exGrid"></div>
  </div>

  <div id="result-zone"></div>
  <div class="divider"></div>

  <div class="about">
    <div class="slabel">About the Model</div>
    <div class="about-title">How does it work?</div>
    <p class="about-desc">
      This system leverages a fine-tuned <strong>DistilBERT</strong> (or BERT) model from HuggingFace
      Transformers, trained on the <em>Kaggle NLP Getting Started</em> dataset (~10 000 labelled tweets).
      The model learns contextual patterns that distinguish genuine disaster reports from metaphorical
      or everyday language, then serves predictions via a REST API deployed on Render.
    </p>
    <div class="feat-grid">
      <div class="feat"><span class="feat-ico">🧠</span>
        <div class="feat-name">Transformer Architecture</div>
        <div class="feat-desc">DistilBERT / BERT pre-trained on 3.3 B words, fine-tuned for binary disaster classification.</div>
      </div>
      <div class="feat"><span class="feat-ico">📊</span>
        <div class="feat-name">NLP Binary Classification</div>
        <div class="feat-desc">Outputs <em>disaster</em> / <em>not disaster</em> with a softmax confidence score between 0 and 1.</div>
      </div>
      <div class="feat"><span class="feat-ico">☁️</span>
        <div class="feat-name">Deployed on Render</div>
        <div class="feat-desc">FastAPI / Flask REST endpoint containerised and hosted on Render for serverless inference.</div>
      </div>
      <div class="feat"><span class="feat-ico">⚡</span>
        <div class="feat-name">Real-Time Inference</div>
        <div class="feat-desc">Average response latency under 2 s per tweet, including tokenisation and forward pass.</div>
      </div>
    </div>
    <div class="tech-row">
      <span class="tech hi">DistilBERT</span>
      <span class="tech hi">HuggingFace 🤗</span>
      <span class="tech hi">PyTorch</span>
      <span class="tech">FastAPI</span>
      <span class="tech">Python 3.11</span>
      <span class="tech">Docker</span>
      <span class="tech">Render</span>
      <span class="tech">Streamlit</span>
    </div>
  </div>
</main>

<footer>
  <div class="foot-title">
    Machine Learning Project
    <span class="foot-dot">·</span>NLP Disaster Classification
    <span class="foot-dot">·</span>BERT + Render
  </div>
  <div class="foot-sub">Built with 🤗 HuggingFace Transformers &nbsp;|&nbsp; Kaggle NLP Getting Started dataset</div>
</footer>

<div id="toast"></div>

<script>
const API_URL = 'https://ahmedtrip-disaster-tweet-api.hf.space/predict'; // ← LOCAL API

const examples = [
  {icon:'🔥', text:'Wildfire spreading rapidly through neighborhoods, thousands evacuated'},
  {icon:'🌊', text:'Massive flooding in downtown streets, rescue teams deployed urgently'},
  {icon:'🌍', text:'7.2 magnitude earthquake hits capital, buildings collapsed downtown'},
  {icon:'🌀', text:'Category 4 hurricane making landfall, emergency declaration issued'},
  {icon:'☕', text:'My coffee is literally burning my tongue this morning, total disaster lol'},
  {icon:'📚', text:'That exam was an absolute disaster but at least summer break is close'},
  {icon:'🎵', text:'This new album is fire, been listening on repeat all week long'},
  {icon:'⚽', text:'Just watched an incredible sunset at the beach, feeling so blessed today'},
];

const grid = document.getElementById('exGrid');
examples.forEach(e => {
  const b = document.createElement('button');
  b.className = 'ex-chip';
  b.innerHTML = `<span>${e.icon}</span>${e.text}`;
  b.onclick = () => {
    document.getElementById('tweetInput').value = e.text;
    updateCount();
    b.style.borderColor = 'rgba(59,130,246,.5)';
    setTimeout(() => b.style.borderColor = '', 600);
  };
  grid.appendChild(b);
});

const input = document.getElementById('tweetInput');
const cc    = document.getElementById('charCount');
function updateCount(){
  const n = input.value.length;
  cc.textContent = n + ' / 280';
  cc.className = 'char-counter' + (n > 240 ? ' warn' : '');
}
input.addEventListener('input', updateCount);

function toast(msg, ms=3500){
  const t = document.getElementById('toast');
  t.textContent = msg; t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), ms);
}

async function analyze(){
  const text = input.value.trim();
  if(!text){ toast('⚠️  Please enter a tweet first.'); return; }
  const btn = document.getElementById('analyzeBtn');
  const ico = document.getElementById('btnIcon');
  const lbl = document.getElementById('btnText');
  btn.disabled = true;
  ico.innerHTML = '<div class="spin"></div>';
  lbl.textContent = 'Analysing…';
  try {
    const res = await fetch(API_URL, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text})
    });
    if(!res.ok) throw new Error('HTTP ' + res.status);
    const d = await res.json();
    renderResult(d.prediction, parseFloat(d.confidence));
  } catch(err) {
    console.error('API Error:', err);
    toast('❌ API Connection Error: ' + err.message);
  } finally {
    btn.disabled = false;
    ico.textContent = '🔍';
    lbl.textContent = 'Analyze Tweet';
  }
}

function simulateAPI(text){
  const kw = ['fire','flood','earthquake','hurricane','disaster','tsunami',
    'explosion','crash','emergency','evacuat','dead','killed','wildfire',
    'tornado','storm','collapse','blaze','rescue'];
  const t   = text.toLowerCase();
  const hit = kw.filter(k => t.includes(k)).length;
  const isD = hit >= 1 && !t.includes('lol') && !t.includes('exam') && !t.includes('coffee');
  const raw = 0.55 + hit * 0.09 + Math.random() * 0.07;
  return {
    prediction: isD ? 'disaster' : 'not disaster',
    confidence: Math.min(.98, Math.max(.51, isD ? raw : 1 - raw + .05))
  };
}

function renderResult(pred, conf){
  const zone = document.getElementById('result-zone');
  const isD  = pred.toLowerCase() === 'disaster';
  const cls  = isD ? 'disaster' : 'safe';
  const pct  = Math.round(conf * 100);
  zone.innerHTML = `
    <div class="result-card ${cls}">
      <div class="result-header">
        <div class="result-emoji">${isD ? '🚨' : '✅'}</div>
        <div>
          <div class="result-title">${isD ? '⚠️ DISASTER DETECTED' : '✅ NO DISASTER'}</div>
          <div class="result-desc">${isD
            ? 'This tweet likely describes a real emergency or catastrophic event.'
            : 'This tweet does not appear to describe a real disaster event.'}</div>
        </div>
      </div>
      <div class="conf-wrap">
        <div class="conf-row">
          <span class="conf-lbl">Confidence Score</span>
          <span class="conf-val" id="confVal">0%</span>
        </div>
        <div class="track"><div class="fill" id="fillBar" style="width:0%"></div></div>
        <div class="ticks"><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div>
      </div>
    </div>`;
  requestAnimationFrame(() => {
    document.getElementById('fillBar').style.width = pct + '%';
    animCounter(document.getElementById('confVal'), 0, pct, 1000);
    zone.scrollIntoView({behavior: 'smooth', block: 'nearest'});
  });
}

function animCounter(el, from, to, dur){
  const s = performance.now();
  (function step(now){
    const p = Math.min((now - s) / dur, 1);
    const e = 1 - Math.pow(1 - p, 4);
    el.textContent = Math.round(from + (to - from) * e) + '%';
    if(p < 1) requestAnimationFrame(step);
  })(performance.now());
}

input.addEventListener('keydown', e => {
  if(e.key === 'Enter' && (e.ctrlKey || e.metaKey)) analyze();
});
</script>
</body>
</html>"""

# Injecte l'HTML — height suffisamment grand pour contenir toute la page
components.html(HTML_APP, height=2600, scrolling=True)
