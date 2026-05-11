"""
Disaster Tweet Detection System — PREMIUM VERSION V8
Design: Fixed Navigation + Internal Scrolling + Theme Support
"""

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Disaster Tweet Detection | BERT",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Cache tout le chrome Streamlit
st.markdown("""
<style>
#MainMenu, footer, header, .stDeployButton,
[data-testid="stToolbar"], [data-testid="stDecoration"],
.viewerBadge_container__r5tak { display: none !important; }
body, .stApp { background: #07090f !important; overflow: hidden !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
iframe { display: block; border: none; height: 100vh !important; }
</style>
""", unsafe_allow_html=True)

HTML_APP = """<!DOCTYPE html>
<html lang="fr" data-theme="dark">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet"/>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root[data-theme="dark"]{
  --bg:#07090f;--bg-card:rgba(17, 25, 40, 0.75);
  --blue:#3b82f6;--cyan:#06b6d4;--gold:#f59e0b;
  --red:#ef4444;--green:#22c55e;
  --txt:#f8fafc;--txt-muted:#94a3b8;
  --glass: rgba(255, 255, 255, 0.03);
  --glass-border: rgba(255, 255, 255, 0.1);
  --nav-bg: rgba(7,9,15,0.85);
}
:root[data-theme="light"]{
  --bg:#f8fafc;--bg-card:rgba(255, 255, 255, 0.9);
  --blue:#2563eb;--cyan:#0891b2;--gold:#d97706;
  --red:#dc2626;--green:#16a34a;
  --txt:#0f172a;--txt-muted:#64748b;
  --glass: rgba(0, 0, 0, 0.02);
  --glass-border: rgba(0, 0, 0, 0.08);
  --nav-bg: rgba(248,250,252,0.9);
}

html{scroll-behavior:smooth; height: 100%;}
body{
  background:var(--bg);color:var(--txt);
  font-family:var(--ff-b);font-size:15px;line-height:1.6;
  transition: background 0.4s ease, color 0.4s ease;
  height: 100%; overflow-y: auto; overflow-x: hidden;
}

/* ─── NAVIGATION ────────────────────────────── */
nav{
  position:fixed;top:0;left:0;right:0;z-index:1000;
  background:var(--nav-bg);backdrop-filter:blur(16px);
  border-bottom:1px solid var(--glass-border);
  padding:15px 40px;display:flex;justify-content:space-between;align-items:center;
}
.nav-logo{font-family:'Outfit',sans-serif;font-weight:800;font-size:18px;letter-spacing:-0.5px}
.nav-logo span{color:var(--blue)}
.nav-right{display:flex;align-items:center;gap:30px}
.nav-links{display:flex;gap:20px}
.nav-links a{
  text-decoration:none;color:var(--txt-muted);font-size:12px;font-weight:700;
  text-transform:uppercase;letter-spacing:1px;transition:color 0.2s;cursor:pointer;
}
.nav-links a:hover{color:var(--blue)}

.theme-toggle{
  background:var(--glass);border:1px solid var(--glass-border);
  width:40px;height:40px;border-radius:12px;cursor:pointer;
  display:flex;align-items:center;justify-content:center;font-size:18px;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.theme-toggle:hover{transform:rotate(15deg) scale(1.1);border-color:var(--blue)}

/* ─── HERO ──────────────────────────────────── */
.hero{
  position:relative;min-height:550px;
  display:flex;flex-direction:column;align-items:center;
  justify-content:center;text-align:center;
  padding:120px 24px 60px;overflow:hidden;
  background: radial-gradient(circle at 50% -20%, rgba(59,130,246,0.15), transparent 60%);
}
.hero-grid{
  position:absolute;inset:0;z-index:0;opacity:0.2;
  background-image: linear-gradient(var(--glass-border) 1px, transparent 1px), linear-gradient(90deg, var(--glass-border) 1px, transparent 1px);
  background-size: 40px 40px;mask-image: radial-gradient(circle at center, black, transparent 80%);
}
.hero-inner{position:relative;z-index:2;max-width:900px}
.badges{display:flex;gap:12px;justify-content:center;margin-bottom:24px;flex-wrap:wrap}
.badge{
  background:var(--glass);border:1px solid var(--glass-border);
  padding:6px 16px;border-radius:100px;font-size:11px;font-weight:700;
  text-transform:uppercase;letter-spacing:1px;color:var(--txt-muted);
}
.badge.champion{color:var(--gold);border-color:rgba(217,119,6,0.3);background:rgba(217,119,6,0.05)}

h1{font-family:'Outfit',sans-serif;font-size:clamp(40px,7vw,72px);font-weight:800;line-height:1;letter-spacing:-0.03em;margin-bottom:20px}
.g-txt{background:linear-gradient(to right, #3b82f6, #06b6d4);-webkit-background-clip:text;-webkit-text-fill-color:transparent}

.btn-git{
  display:inline-flex;align-items:center;gap:10px;
  background:var(--glass);border:1px solid var(--glass-border);
  padding:12px 24px;border-radius:12px;color:var(--txt);
  text-decoration:none;font-weight:600;font-size:14px;transition:all 0.3s;
}
.btn-git:hover{background:var(--txt);color:var(--bg)}

/* ─── SECTIONS ──────────────────────────────── */
.container{max-width:950px;margin:0 auto;padding:0 24px 60px}
section{padding-top:110px;margin-top:0px}

.mission-grid{display:grid;grid-template-columns:1fr 1fr;gap:30px;margin-bottom:60px}
.m-card{background:var(--glass);border:1px solid var(--glass-border);padding:35px;border-radius:28px}
.m-title{font-family:'Outfit',sans-serif;font-size:19px;font-weight:700;margin-bottom:12px;display:flex;align-items:center;gap:12px}
.m-text{font-size:14.5px;color:var(--txt-muted);line-height:1.75}

.main-card{
  background:var(--bg-card);backdrop-filter:blur(16px);
  border:1px solid var(--glass-border);border-radius:35px;
  padding:50px;box-shadow: 0 30px 60px -12px rgba(0,0,0,0.15);
}
[data-theme="dark"] .main-card{box-shadow: 0 30px 60px -12px rgba(0,0,0,0.6)}

textarea{
  width:100%;background:rgba(0,0,0,0.05);border:1px solid var(--glass-border);
  border-radius:20px;color:var(--txt);font-family:inherit;padding:24px;
  font-size:16px;min-height:140px;outline:none;transition:all 0.3s;resize:none;
}
[data-theme="dark"] textarea{background:rgba(0,0,0,0.25)}
textarea:focus{border-color:var(--blue);box-shadow:0 0 0 4px rgba(59,130,246,0.1)}

.analyze-btn{
  width:100%;background:var(--blue);color:white;border:none;
  padding:19px;border-radius:20px;font-weight:700;font-size:16px;
  margin-top:22px;cursor:pointer;transition:all 0.3s;
  box-shadow:0 12px 24px -6px rgba(59,130,246,0.3);
}

.ex-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:34px}
.ex-chip{
  background:var(--glass);border:1px solid var(--glass-border);
  padding:15px;border-radius:15px;font-size:13px;color:var(--txt-muted);
  text-align:left;cursor:pointer;transition:all 0.2s;display:flex;gap:10px;
}
.ex-chip:hover{background:var(--blue);color:white;border-color:var(--blue)}

/* ─── RESULTS ───────────────────────────────── */
#result-zone{margin-top:40px}
.result-card{border-radius:28px;padding:40px;animation:revealUp .5s ease-out forwards}
@keyframes revealUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
.result-card.disaster{background:rgba(239,68,68,0.08);border:1px solid var(--red)}
.result-card.safe{background:rgba(34,197,94,0.08);border:1px solid var(--green)}
.result-header{display:flex;align-items:center;gap:20px;margin-bottom:26px}
.result-emoji{width:64px;height:64px;border-radius:18px;display:flex;align-items:center;justify-content:center;font-size:32px;background:rgba(255,255,255,0.1)}
.result-title{font-family:'Outfit',sans-serif;font-size:25px;font-weight:800;line-height:1.1}
.disaster .result-title{color:var(--red)}.safe .result-title{color:var(--green)}
.conf-val{font-family:'Outfit',sans-serif;font-size:24px;font-weight:800}
.track{height:12px;background:rgba(0,0,0,0.1);border-radius:100px;overflow:hidden;margin-top:10px}
.fill{height:100%;border-radius:100px;transition:width 1s ease-out}
.disaster .fill{background:var(--red)}.safe .fill{background:var(--green)}

/* ─── TEAM ──────────────────────────────────── */
.team-grid{display:grid;grid-template-columns:repeat(4, 1fr);gap:18px;margin-top:50px}
.team-member{
  background:var(--glass);border:1px solid var(--glass-border);
  padding:30px 15px;border-radius:24px;text-align:center;
  text-decoration:none;color:inherit;transition:all 0.3s;
  display:flex;flex-direction:column;align-items:center;gap:15px;
}
.team-member:hover{transform:translateY(-10px);border-color:var(--blue);background:var(--glass)}
.mem-avatar{width:80px;height:80px;border-radius:50%;border:2px solid var(--glass-border);padding:4px}
.mem-avatar img{width:100%;height:100%;border-radius:50%;object-fit:cover}
.mem-name{font-weight:700;font-size:13px}

/* ─── FOOTER ────────────────────────────────── */
.footer{background:var(--glass);padding:80px 40px 60px;border-top:1px solid var(--glass-border);margin-top:40px}
.footer-inner{max-width:1100px;margin:0 auto;display:grid;grid-template-columns:2fr 1fr 1fr;gap:60px}
.f-about-t{font-family:'Outfit',sans-serif;font-size:20px;font-weight:800;margin-bottom:15px}
.f-about-p{color:var(--txt-muted);font-size:14px;line-height:1.7}
.f-title{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;color:var(--blue);margin-bottom:20px}
.f-links{display:flex;flex-direction:column;gap:12px}
.f-links a{text-decoration:none;color:var(--txt-muted);font-size:14px;transition:color 0.2s;cursor:pointer}
.f-links a:hover{color:var(--txt)}
.f-bottom{margin-top:60px;padding-top:30px;border-top:1px solid var(--glass-border);display:flex;justify-content:space-between;align-items:center;font-size:12px;color:var(--txt-muted)}

@media(max-width:900px){
  .footer-inner{grid-template-columns:1fr;gap:40px}
  .team-grid{grid-template-columns:repeat(2, 1fr)}
  .mission-grid{grid-template-columns:1fr}
}
</style>
</head>
<body>

<nav>
  <div class="nav-logo">DISASTER<span>DETECT</span></div>
  <div class="nav-right">
    <div class="nav-links">
      <a onclick="scrollSection('home')">Home</a>
      <a onclick="scrollSection('mission')">Mission</a>
      <a onclick="scrollSection('app')">Analyze</a>
      <a onclick="scrollSection('team')">Team</a>
    </div>
    <button class="theme-toggle" id="themeToggle" onclick="toggleTheme()">☀️</button>
  </div>
</nav>

<div id="home"></div>
<header class="hero">
  <div class="hero-grid"></div>
  <div class="hero-inner">
    <div class="badges">
      <div class="badge champion">Champion Model: BERT-base</div>
      <div class="badge">Hugging Face 🤗</div>
    </div>
    <h1>Detect Disaster<br><span class="g-txt">In Every Tweet</span></h1>
    <div class="hero-actions">
      <a href="https://github.com/ahmadouniass/Disaster-Tweets-NLP" class="btn-git" target="_blank">
        <svg height="20" viewBox="0 0 16 16" width="20" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>
        View Source Code
      </a>
    </div>
  </div>
</header>

<div class="container">
  <section id="mission">
    <div class="mission-grid">
      <div class="m-card">
        <div class="m-title"><span>🎯</span> Project Mission</div>
        <div class="m-text">Separating real emergency reports from social media noise using high-precision NLP architectures.</div>
      </div>
      <div class="m-card">
        <div class="m-title"><span>🧠</span> Technology</div>
        <div class="m-text">Fine-tuned BERT transformers capable of understanding global semantic context and intent.</div>
      </div>
    </div>
  </section>

  <section id="app">
    <div class="main-card">
      <textarea id="tweetInput" placeholder="Analyze a tweet here..."></textarea>
      <button class="analyze-btn" id="analyzeBtn" onclick="analyze()">Analyze Tweet</button>
      <div class="ex-grid" id="exGrid"></div>
      <div id="result-zone"></div>
    </div>
  </section>

  <section id="team">
    <h2 style="text-align:center; margin-bottom: 40px; font-family:'Outfit',sans-serif; font-size:28px">Our Team</h2>
    <div class="team-grid">
      <a href="https://github.com/ahmadouniass" target="_blank" class="team-member">
        <div class="mem-avatar"><img src="https://github.com/ahmadouniass.png"></div>
        <div class="mem-name">Ahmadou Niass</div>
      </a>
      <a href="https://github.com/Khadidiatou1010" target="_blank" class="team-member">
        <div class="mem-avatar"><img src="https://github.com/Khadidiatou1010.png"></div>
        <div class="mem-name">Khadidiatou Coulibaly</div>
      </a>
      <a href="https://github.com/dior204" target="_blank" class="team-member">
        <div class="mem-avatar"><img src="https://github.com/dior204.png"></div>
        <div class="mem-name">Dior Mbengue</div>
      </a>
      <a href="https://github.com/Kerencia2" target="_blank" class="team-member">
        <div class="mem-avatar"><img src="https://github.com/Kerencia2.png"></div>
        <div class="mem-name">Pahane S. K. D.</div>
      </a>
    </div>
  </section>
</div>

<footer class="footer">
  <div class="footer-inner">
    <div>
      <div class="f-about-t">DisasterDetect<span>.</span></div>
      <p class="f-about-p">Developed for the TP Machine Learning curriculum at ENSAE Dakar. Focused on real-time NLP crisisinformatics.</p>
    </div>
    <div>
      <div class="f-title">Quick Links</div>
      <div class="f-links">
        <a onclick="scrollSection('home')">Home</a>
        <a onclick="scrollSection('mission')">Mission</a>
        <a onclick="scrollSection('app')">Analysis App</a>
        <a onclick="scrollSection('team')">Team</a>
      </div>
    </div>
    <div>
      <div class="f-title">Resources</div>
      <div class="f-links">
        <a href="https://github.com/ahmadouniass/Disaster-Tweets-NLP">GitHub Repo</a>
        <a href="https://ahmedtrip-disaster-tweet-api.hf.space/docs">API Docs</a>
      </div>
    </div>
  </div>
  <div class="f-bottom">
    <div>&copy; 2026 Disaster NLP Project Team</div>
    <div>Built with 💙 using PyTorch & Streamlit</div>
  </div>
</footer>

<script>
const API_URL = 'https://ahmedtrip-disaster-tweet-api.hf.space/predict';

function scrollSection(id){
  const el = document.getElementById(id);
  if(el) el.scrollIntoView({ behavior: 'smooth' });
}

function toggleTheme(){
  const html = document.documentElement;
  const btn = document.getElementById('themeToggle');
  const current = html.getAttribute('data-theme');
  const next = current === 'dark' ? 'light' : 'dark';
  html.setAttribute('data-theme', next);
  btn.innerHTML = next === 'dark' ? '☀️' : '🌙';
}

const examples = [
  {icon:'🚨', text:'Confirmed: Wildfire approaching the city, evacuations ordered!'},
  {icon:'🌊', text:'Serious flooding reported downtown after last night flash rain.'},
  {icon:'🎵', text:'This new album is absolute fire, listening on repeat!'},
  {icon:'🍿', text:'The movie was a disaster, definitely not worth the price.'}
];

const grid = document.getElementById('exGrid');
examples.forEach(e => {
  const chip = document.createElement('div');
  chip.className = 'ex-chip';
  chip.innerHTML = `<span>${e.icon}</span> ${e.text}`;
  chip.onclick = () => {
    document.getElementById('tweetInput').value = e.text;
    analyze();
  };
  grid.appendChild(chip);
});

async function analyze(){
  const text = document.getElementById('tweetInput').value.trim();
  if(!text) return;
  const btn = document.getElementById('analyzeBtn');
  const zone = document.getElementById('result-zone');
  btn.disabled = true;
  btn.innerHTML = '⚡ Analysing...';
  
  try {
    const res = await fetch(API_URL, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text})
    });
    const data = await res.json();
    const isDisaster = data.prediction === "Disaster";
    const conf = (data.confidence * 100).toFixed(1);
    
    zone.innerHTML = `
      <div class="result-card ${isDisaster ? 'disaster' : 'safe'}" style="margin-top:30px">
        <div class="result-header">
          <div class="result-emoji">${isDisaster ? '🚨' : '✅'}</div>
          <div>
            <div class="result-title">${isDisaster ? 'Disaster Detected' : 'No Danger Detected'}</div>
            <div style="font-size:13px; opacity:0.7">Analysis complete.</div>
          </div>
        </div>
        <div>
          <div style="display:flex; justify-content:space-between; margin-bottom:10px">
            <span style="font-size:12px; font-weight:700">Confidence Level</span>
            <span class="conf-val">${conf}%</span>
          </div>
          <div class="track"><div class="fill" style="width:${conf}%"></div></div>
        </div>
      </div>
    `;
  } catch (e) {
    zone.innerHTML = '<div style="color:var(--red); padding:20px; text-align:center">Error connecting to API.</div>';
  } finally {
    btn.disabled = false;
    btn.innerHTML = 'Analyze Tweet';
  }
}
</script>
</body>
</html>
"""

# Ici, on force l'iframe à prendre TOUTE la hauteur disponible de la page Streamlit
# et on désactive le défilement extérieur.
components.html(HTML_APP, height=1200, scrolling=True)
