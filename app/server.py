"""
server.py — 離線伺服器，讓手機透過瀏覽器使用
==========================================================
架構「方案 A」：筆電 = 大腦（負責推論），手機 = 眼睛＋喇叭
透過同一個 Wi-Fi／熱點連線 — **不需要網際網路 = 真正離線**（資服版另支援 Tailscale 從外面連回）

只用 Python 內建模組（http.server + ssl）— 不需要安裝 Flask／FastAPI
憑證：用 openssl 產生；沒有 openssl 時改用 cryptography 套件；Tailscale 開啟 HTTPS 時自動用正式憑證

使用方式（建議加 --https，手機的相機才能開啟）：
    conda activate vlm_dml
    python -m app.server --device igpu --https
  → 手機打開 https://<筆電 IP>:8443
    （自簽憑證只需第一次按「顯示詳細資訊／繼續前往」略過警告）

為視障者設計的操作：點螢幕任何地方 = 描述、震動確認、
連續模式（自動重複）、唸出狀態與結果、偵測到危險時強烈震動
"""

import sys
import io
import os
import ssl
import json
import time
import base64
import socket
import argparse
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import threading

import numpy as np
from PIL import Image

from app import config
from app.assistant import build_backend

BACKEND = None  # 啟動時載入一次
SAVE_DIR = None  # 有設定時 = 把畫面和輸出存下來除錯用
_SEQ = 0
GPU_LOCK = threading.Lock()   # [資服版] 描述和語音提問不要同時用顯示卡
_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
_STATIC = {"/manifest.json": ("manifest.json", "application/manifest+json"),
           "/icon-180.png": ("icon-180.png", "image/png"),
           "/icon-512.png": ("icon-512.png", "image/png")}
VOICE_READY = False

PAGE = """<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no, viewport-fit=cover">
<meta name="theme-color" content="#000000">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="視覺助理">
<link rel="manifest" href="/manifest.json">
<link rel="apple-touch-icon" href="/icon-180.png">
<link rel="icon" href="/icon-180.png">
<title>視覺助理</title>
<style>
  :root { --glass:rgba(16,17,22,.74); --line:rgba(255,255,255,.14); --muted:rgba(255,255,255,.66);
          --accent:#5ac8fa; --danger:#ff453a; --ok:#32d74b; }
  * { box-sizing:border-box; -webkit-user-select:none; user-select:none; -webkit-tap-highlight-color:transparent; }
  html,body { margin:0; height:100%; background:#000; color:#fff; overflow:hidden;
              font-family:-apple-system,"PingFang TC","Noto Sans TC","Microsoft JhengHei",system-ui,sans-serif; }
  #video { position:fixed; inset:0; width:100%; height:100%; object-fit:cover; }
  /* 上下漸層，讓文字在亮的畫面上也看得清楚 */
  #shade { position:fixed; inset:0; z-index:1; pointer-events:none;
           background:linear-gradient(to bottom, rgba(0,0,0,.55) 0, rgba(0,0,0,0) 22%,
                                      rgba(0,0,0,0) 55%, rgba(0,0,0,.6) 100%); }
  /* 點螢幕任何地方 = 描述（視障者不必對準按鈕） */
  #tap { position:fixed; inset:0; z-index:2; }

  #bar { position:fixed; top:0; left:0; right:0; z-index:5; pointer-events:none;
         padding:calc(env(safe-area-inset-top) + 12px) 14px 0; }
  #modeBtn, #cont, #menu, #gear, #settings { pointer-events:auto; }   /* 空白處的點擊照樣傳給 #tap */
  #head { display:flex; align-items:center; justify-content:space-between; gap:10px; }
  .glass { background:var(--glass); border:1px solid var(--line);
           -webkit-backdrop-filter:blur(18px) saturate(1.4); backdrop-filter:blur(18px) saturate(1.4); }
  #dot { flex:none; width:10px; height:10px; border-radius:50%; background:var(--ok); box-shadow:0 0 10px var(--ok); }
  #dot.off { background:#8e8e93; box-shadow:none; }
  #dot.bad { background:var(--danger); box-shadow:0 0 10px var(--danger); }
  #offline { display:none; margin-top:10px; padding:10px 16px; border-radius:16px; width:fit-content;
             background:var(--danger); color:#fff; font-size:17px; font-weight:700; pointer-events:none; }
  #offline.show { display:block; }

  /* 模式：點了才展開的下拉選單 */
  #modeBtn { display:flex; align-items:center; gap:10px; min-height:46px; padding:0 16px 0 14px;
             border-radius:999px; font:inherit; font-size:18px; font-weight:700; color:#fff; }
  #modeBtn .chev { width:9px; height:9px; margin-left:2px; border-right:2.5px solid currentColor;
                   border-bottom:2.5px solid currentColor; transform:translateY(-3px) rotate(45deg); transition:transform .2s; }
  #modeBtn[aria-expanded="true"] .chev { transform:translateY(2px) rotate(-135deg); }
  #menu { display:none; margin-top:10px; width:min(320px, 100%); padding:6px; border-radius:22px;
          background:rgba(22,23,28,.9); transform-origin:top left; }
  #menu.open { display:block; animation:pop .16s ease-out; }
  @keyframes pop { from { opacity:0; transform:scale(.96) translateY(-4px); } to { opacity:1; transform:none; } }
  .opt { display:flex; align-items:center; justify-content:space-between; width:100%; min-height:58px;
         padding:8px 16px; border:0; border-radius:16px; background:transparent; color:#fff; text-align:left;
         font:inherit; font-size:19px; font-weight:700; }
  .opt small { display:block; margin-top:2px; font-size:14px; font-weight:500; color:var(--muted); }
  .opt[aria-checked="true"] { background:rgba(255,255,255,.12); }
  .opt[aria-checked="true"]::after { content:"✓"; font-size:20px; font-weight:800; color:var(--accent); }
  #scrim { position:fixed; inset:0; z-index:4; display:none; background:rgba(0,0,0,.35); }
  #scrim.open { display:block; }

  /* 連續：開關 */
  #cont { display:flex; align-items:center; gap:10px; font-size:17px; font-weight:600; min-height:46px;
          padding:0 8px 0 14px; border-radius:999px; }
  .sw { flex:none; -webkit-appearance:none; appearance:none; margin:0; width:48px; height:28px; border-radius:999px;
        background:rgba(255,255,255,.25); position:relative; transition:background .2s; }
  .sw::after { content:""; position:absolute; top:3px; left:3px; width:22px; height:22px; border-radius:50%;
               background:#fff; transition:transform .2s; box-shadow:0 1px 3px rgba(0,0,0,.4); }
  .sw:checked { background:var(--ok); }
  .sw:checked::after { transform:translateX(20px); }

  /* 設定：齒輪按鈕 + 面板 */
  #head .right { display:flex; align-items:center; gap:8px; }
  #gear { pointer-events:auto; width:46px; height:46px; border-radius:50%; display:flex; align-items:center;
          justify-content:center; color:#fff; padding:0; }
  #gear svg { width:22px; height:22px; }
  #settings { display:none; pointer-events:auto; margin-top:10px; padding:8px 16px 14px; border-radius:22px;
              background:rgba(22,23,28,.92); max-width:440px; }
  #settings.open { display:block; animation:pop .16s ease-out; }
  .srow { display:flex; align-items:center; justify-content:space-between; gap:12px; min-height:60px;
          border-bottom:1px solid rgba(255,255,255,.08); font-size:18px; font-weight:700; }
  .srow small { display:block; font-size:13px; font-weight:500; color:var(--muted); }
  .seg { display:flex; gap:4px; padding:4px; border-radius:14px; background:rgba(255,255,255,.1); }
  .seg button { min-width:46px; min-height:40px; padding:0 10px; border:0; border-radius:10px; background:transparent;
                color:#fff; font:inherit; font-size:16px; font-weight:600; }
  .seg button[aria-checked="true"] { background:#fff; color:#000; }
  .sbtns { display:flex; gap:10px; margin-top:14px; }
  .sbtns button { flex:1; min-height:50px; border:0; border-radius:14px; font:inherit; font-size:17px; font-weight:700; }
  #resetSet { background:rgba(255,255,255,.12); color:#fff; }
  #closeSet { background:#fff; color:#000; }

  /* 下方：結果卡片 + 語音提問按鈕 */
  #bottom { position:fixed; left:12px; right:12px; z-index:3; display:flex; flex-direction:column; gap:10px;
            bottom:calc(env(safe-area-inset-bottom) + 12px); }
  #card { position:relative; overflow:hidden; padding:16px 18px 20px; border-radius:24px;
          transition:background .25s, border-color .25s; }
  #heard { display:none; margin-bottom:6px; font-size:16px; color:var(--muted); }
  #heard.show { display:block; }
  #mic { display:flex; align-items:center; justify-content:center; gap:10px; min-height:64px; border:0;
         border-radius:999px; background:#fff; color:#000; font:inherit; font-size:20px; font-weight:700;
         transition:background .2s, color .2s; }
  #mic svg { width:24px; height:24px; }
  #mic.listen { background:var(--danger); color:#fff; animation:pulse 1.2s ease-out infinite; }
  #mic.think { background:rgba(255,255,255,.28); color:#fff; }
  @keyframes pulse { from { box-shadow:0 0 0 0 rgba(255,69,58,.65); } to { box-shadow:0 0 0 18px rgba(255,69,58,0); } }
  #card.danger { background:rgba(110,14,10,.86); border-color:var(--danger); }
  #meta { display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;
          font-size:15px; color:var(--muted); }
  #tag { display:inline-flex; align-items:center; gap:6px; font-weight:700; color:var(--accent); }
  #tag::before { content:""; width:8px; height:8px; border-radius:50%; background:currentColor; }
  #card.danger #tag { color:#fff; }
  #time { font-variant-numeric:tabular-nums; }
  #out { font-size:var(--fs, 25px); line-height:1.5; font-weight:500; min-height:1.5em; }
  #out.hint { color:var(--muted); }
  /* 處理中：卡片頂端跑動的光條 */
  #prog { position:absolute; top:0; left:0; right:0; height:3px; opacity:0; transition:opacity .2s;
          background:linear-gradient(90deg, transparent, var(--accent), transparent); background-size:50% 100%;
          background-repeat:no-repeat; animation:slide 1.1s linear infinite; }
  #card.busy #prog { opacity:1; }
  @keyframes slide { from { background-position:-50% 0; } to { background-position:150% 0; } }
  @media (prefers-reduced-motion: reduce) { #prog, #mic.listen, #menu.open { animation:none; }
                                            #prog { background-size:100% 100%; } }
</style>
</head>
<body>
  <video id="video" autoplay playsinline muted></video>
  <div id="shade"></div>
  <div id="tap" aria-label="點一下畫面開始描述"></div>
  <div id="scrim"></div>
  <div id="bar">
    <div id="head">
      <button id="modeBtn" class="glass" aria-haspopup="true" aria-expanded="false" aria-controls="menu">
        <span id="dot" class="off"></span><span id="modeName">自動</span><span class="chev"></span>
      </button>
      <div class="right">
        <label id="cont" class="glass">連續<input type="checkbox" id="contChk" class="sw" role="switch" aria-label="連續模式"></label>
        <button id="gear" class="glass" aria-label="設定" aria-expanded="false" aria-controls="settings">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <circle cx="12" cy="12" r="3"></circle>
            <path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"></path>
          </svg>
        </button>
      </div>
    </div>
    <div id="settings" class="glass" role="dialog" aria-label="設定">
      <div class="srow"><span>語速</span>
        <div class="seg" role="radiogroup" aria-label="語速" data-key="rate">
          <button role="radio">慢</button><button role="radio">標準</button><button role="radio">快</button><button role="radio">很快</button>
        </div></div>
      <div class="srow"><span>字體大小</span>
        <div class="seg" role="radiogroup" aria-label="字體大小" data-key="font">
          <button role="radio">標準</button><button role="radio">大</button><button role="radio">特大</button>
        </div></div>
      <label class="srow" id="vibRow"><span>震動提示</span><input type="checkbox" class="sw" role="switch" data-key="vib" aria-label="震動提示"></label>
      <label class="srow" id="shakeRow"><span>搖一搖提問<small>不用找按鈕，搖兩下手機就開始聽</small></span><input type="checkbox" class="sw" role="switch" data-key="shake" aria-label="搖一搖提問"></label>
      <div class="sbtns"><button id="resetSet">恢復預設</button><button id="closeSet">完成</button></div>
    </div>
    <div id="offline" role="alert">連不到電腦，正在重新連線…</div>
    <div id="menu" class="glass" role="menu" aria-label="選擇模式">
      <button class="opt" role="menuitemradio" data-mode="auto" data-name="自動"><span>自動<small>依畫面自動判斷</small></span></button>
      <button class="opt" role="menuitemradio" data-mode="street" data-name="過馬路"><span>過馬路<small>行人號誌與車輛</small></span></button>
      <button class="opt" role="menuitemradio" data-mode="surrounding" data-name="周遭環境"><span>周遭環境<small>描述前方景物</small></span></button>
      <button class="opt" role="menuitemradio" data-mode="indoor" data-name="室內"><span>室內<small>格局與障礙物</small></span></button>
      <button class="opt" role="menuitemradio" data-mode="read" data-name="讀字"><span>讀字<small>唸出看到的文字</small></span></button>
      <button class="opt" role="menuitemradio" data-mode="object" data-name="辨識物品"><span>辨識物品<small>手上拿的東西</small></span></button>
    </div>
  </div>
  <div id="bottom">
    <div id="card" class="glass">
      <div id="prog"></div>
      <div id="meta"><span id="tag">自動</span><span id="time"></span></div>
      <div id="heard"></div>
      <div id="out" class="hint" aria-live="assertive">點一下畫面描述，或按下方按鈕用說的</div>
    </div>
    <button id="mic" aria-label="語音提問">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <rect x="9" y="3" width="6" height="11" rx="3"></rect><path d="M5 11a7 7 0 0 0 14 0M12 18v3"></path></svg>
      <span id="micText">語音提問</span>
    </button>
  </div>
<script>
const video=document.getElementById('video'), out=document.getElementById('out'),
      tap=document.getElementById('tap'), card=document.getElementById('card'),
      tag=document.getElementById('tag'), timeEl=document.getElementById('time'),
      dot=document.getElementById('dot'), contChk=document.getElementById('contChk'),
      modeBtn=document.getElementById('modeBtn'), modeNameEl=document.getElementById('modeName'),
      menu=document.getElementById('menu'), scrim=document.getElementById('scrim'),
      mic=document.getElementById('mic'), micText=document.getElementById('micText'),
      heardEl=document.getElementById('heard'), offlineEl=document.getElementById('offline');
let busy=false, started=false, wakeLock=null, lastText='';

// [資服版] 語音提問的對話歷史：追問才沿用（由伺服器判斷），點畫面描述、換模式、
//   超過 2 分鐘沒問就清掉（人可能已經走到別的地方）
let convo=[], convoAt=0;
const CONVO_TTL=120000;
function clearConvo(){ convo=[]; convoAt=0; }

// 模式下拉選單，記住上次選的模式
const opts=[...document.querySelectorAll('.opt')];
let MODE='auto';
try{ MODE=localStorage.getItem('mode')||'auto'; }catch(e){}
function modeName(m){ const o=opts.find(o=>o.dataset.mode===m); return o?o.dataset.name:''; }
function setMode(m, announce){
  if(!opts.some(o=>o.dataset.mode===m)) m='auto';
  MODE=m;
  clearConvo();                                   // 換模式 = 換話題
  opts.forEach(o=>o.setAttribute('aria-checked', o.dataset.mode===m?'true':'false'));
  modeNameEl.textContent=modeName(m);
  modeBtn.setAttribute('aria-label', '模式：'+modeName(m));
  tag.textContent=modeName(m);
  try{ localStorage.setItem('mode', m); }catch(e){}
  if(announce) speak(modeName(m)+'模式');
}
function openMenu(open){
  if(open) openSettings(false);
  menu.classList.toggle('open', open);
  scrim.classList.toggle('open', open);
  modeBtn.setAttribute('aria-expanded', open?'true':'false');
  if(open){ const cur=opts.find(o=>o.getAttribute('aria-checked')==='true'); if(cur) cur.focus(); }
}
modeBtn.addEventListener('click', ()=>{ vibrate(20); openMenu(!menu.classList.contains('open')); });
scrim.addEventListener('click', ()=>{ openMenu(false); openSettings(false); });
opts.forEach(o=>o.addEventListener('click', ()=>{ vibrate(30); setMode(o.dataset.mode, true); openMenu(false); }));
setMode(MODE, false);

// ---------------------------------------------------------------------------
// [資服版] 個人化設定：語速、字體大小、震動；存在手機上，也可以用語音調整
// ---------------------------------------------------------------------------
const RATES=[0.75, 0.95, 1.2, 1.45], RATE_NAMES=['慢','標準','快','很快'];
const FONTS=[25, 31, 38], FONT_NAMES=['標準','大','特大'];
const CAN_VIB=('vibrate' in navigator);           // iPhone 的 Safari 不支援網頁震動
const CAN_SHAKE=('DeviceMotionEvent' in window);
const DEFAULTS={rate:1, font:0, vib:true, shake:true};
let SET=Object.assign({}, DEFAULTS);
try{ Object.assign(SET, JSON.parse(localStorage.getItem('settings')||'{}')); }catch(e){}
const settingsEl=document.getElementById('settings'), gear=document.getElementById('gear');
const segs={}; document.querySelectorAll('.seg').forEach(s=>segs[s.dataset.key]=s);
const vibSw=document.querySelector('.sw[data-key="vib"]'), shakeSw=document.querySelector('.sw[data-key="shake"]');
if(!CAN_VIB) document.getElementById('vibRow').style.display='none';
if(!CAN_SHAKE) document.getElementById('shakeRow').style.display='none';

function applySettings(){
  document.documentElement.style.setProperty('--fs', FONTS[SET.font]+'px');
  for(const k in segs) [...segs[k].children].forEach((b,i)=>b.setAttribute('aria-checked', i===SET[k]?'true':'false'));
  vibSw.checked=!!SET.vib;
  shakeSw.checked=!!SET.shake;
  try{ localStorage.setItem('settings', JSON.stringify(SET)); }catch(e){}
}
// 調整設定，回傳要說的話（設定面板和語音指令共用）
function changeSetting(key, value){
  if(key==='reset'){ SET=Object.assign({}, DEFAULTS); applySettings(); return '已恢復預設設定。'; }
  if(key==='vib'){
    if(!CAN_VIB) return '這支手機的瀏覽器不支援震動。';
    SET.vib=!!value; applySettings(); if(SET.vib) vibrate(120);
    return SET.vib?'震動已開啟。':'震動已關閉。';
  }
  if(key==='shake'){
    if(!CAN_SHAKE) return '這支手機的瀏覽器不支援搖一搖。';
    SET.shake=!!value; applySettings(); if(SET.shake) enableMotion();
    return SET.shake?'搖一搖提問已開啟，搖兩下手機就會開始聽。':'搖一搖提問已關閉。';
  }
  const max=(key==='rate'?RATES:FONTS).length-1, names=key==='rate'?RATE_NAMES:FONT_NAMES;
  const label=key==='rate'?'語速':'字體';
  let v=value==='up'?SET[key]+1:value==='down'?SET[key]-1:value;
  if(v>max) return key==='rate'?'已經是最快的語速了。':'字已經是最大了。';
  if(v<0) return key==='rate'?'已經是最慢的語速了。':'字已經是最小了。';
  SET[key]=v; applySettings();
  return label+'：'+names[v]+'。';
}
function openSettings(open){
  settingsEl.classList.toggle('open', open);
  gear.setAttribute('aria-expanded', open?'true':'false');
  if(open){ menu.classList.remove('open'); modeBtn.setAttribute('aria-expanded','false'); }
  scrim.classList.toggle('open', open || menu.classList.contains('open'));
}
gear.addEventListener('click', ()=>{ vibrate(20); openSettings(!settingsEl.classList.contains('open')); });
document.getElementById('closeSet').addEventListener('click', ()=>openSettings(false));
document.getElementById('resetSet').addEventListener('click', ()=>speak(changeSetting('reset')));
for(const k in segs) [...segs[k].children].forEach((b,i)=>b.addEventListener('click', ()=>speak(changeSetting(k, i))));
vibSw.addEventListener('change', ()=>speak(changeSetting('vib', vibSw.checked)));
shakeSw.addEventListener('change', ()=>speak(changeSetting('shake', shakeSw.checked)));
applySettings();

// 相機和麥克風一起要，iPhone 只會跳一次權限視窗（主畫面 App 每次開啟都會重問，無法記住）
//   麥克風拿到權限後馬上關掉：一直開著的話 iPhone 會切到通話模式，喇叭聲音變小
const CAM={facingMode:{ideal:'environment'}, width:{ideal:1920}, height:{ideal:1080}};
function startCamera(s){
  s.getAudioTracks().forEach(t=>t.stop());
  video.srcObject=new MediaStream(s.getVideoTracks());
  dot.classList.remove('off');
}
navigator.mediaDevices.getUserMedia({video:CAM, audio:true})
  .then(startCamera)
  .catch(()=>navigator.mediaDevices.getUserMedia({video:CAM}).then(startCamera))   // 不給麥克風也能用相機
  .catch(e=>{ show('相機錯誤：'+e.message); speak('相機錯誤'); });

// 更新結果卡片
function show(text, opts){
  opts=opts||{};
  out.textContent=text;
  out.classList.toggle('hint', !!opts.hint);
  card.classList.toggle('danger', !!opts.danger);
  timeEl.textContent=opts.time||'';
}

// ---------------------------------------------------------------------------
// [資服版] 連線狀態：請求加上逾時，失敗時分清楚是「連不到電腦」還是「電腦太慢／出錯」，
//   並每 10 秒確認一次連線；斷線、恢復都會用語音告知
// ---------------------------------------------------------------------------
let online=true;
const ERR_TEXT={offline:'連不到電腦，請確認網路，或電腦是否開著。',
                slow:'電腦回應太慢，請再試一次。',
                server:'電腦處理時發生錯誤，請再試一次。'};
function netErr(kind, detail){ const e=new Error(detail||kind); e.kind=kind; return e; }
async function api(path, body, ms){
  const ctl=new AbortController(), tm=setTimeout(()=>ctl.abort(), ms);
  let r;
  try{
    r=await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'},
                         body:JSON.stringify(body), signal:ctl.signal});
  }catch(e){
    clearTimeout(tm);
    if(e.name==='AbortError') throw netErr('slow');
    setOnline(false, true); throw netErr('offline');
  }
  clearTimeout(tm); setOnline(true, true);
  let j=null; try{ j=await r.json(); }catch(e){}
  if(!r.ok || !j || j.error){ console.log('server error', j&&j.error); throw netErr('server', j&&j.error); }
  return j;
}
function fail(e){
  const msg=ERR_TEXT[e.kind]||ERR_TEXT.server;
  show(msg, {danger:true}); speak(msg); vibrate([400]);
}
function setOnline(v, quiet){
  if(v===online) return;
  online=v;
  offlineEl.classList.toggle('show', !v);
  dot.classList.toggle('bad', !v);
  if(v){
    if(!quiet) speak('已重新連上電腦。');
    if(contChk.checked && !busy && !asking) whenQuiet(describe);     // 斷線時暫停的連續描述接著跑
  }else if(!quiet){ speak('和電腦的連線中斷了。'); vibrate([400]); }
}
async function ping(){
  if(document.visibilityState!=='visible') return;
  const ctl=new AbortController(), tm=setTimeout(()=>ctl.abort(), 5000);
  try{ const r=await fetch('/ping', {cache:'no-store', signal:ctl.signal}); setOnline(r.ok); }
  catch(e){ setOnline(false); }
  clearTimeout(tm);
}
setInterval(ping, 10000);

async function keepAwake(){ try{ wakeLock=await navigator.wakeLock.request('screen'); }catch(e){} }
function vibrate(p){ if(!SET.vib) return; try{ navigator.vibrate&&navigator.vibrate(p); }catch(e){} }

// 依語言挑語音 — 清楚的英文＋中文（OCR 讀得到中文招牌，所以也要能唸中文，不能走音）
let VOICE=null, VOICE_ZH=null;
function pickVoice(){
  const all=speechSynthesis.getVoices();
  const vs=all.filter(v=>v.lang&&v.lang.toLowerCase().startsWith('en'));
  const pref=['Samantha','Aria','Jenny','Google US English','Microsoft Zira','Daniel','Karen','en-US'];
  for(const p of pref){ const m=vs.find(v=>v.name&&v.name.includes(p)); if(m){VOICE=m;break;} }
  if(!VOICE) VOICE=vs.find(v=>v.lang&&v.lang.toLowerCase()==='en-us')||vs[0]||null;
  // 中文語音：優先臺灣（zh-TW）→ 其他中文
  const zh=all.filter(v=>v.lang&&v.lang.toLowerCase().startsWith('zh'));
  VOICE_ZH=zh.find(v=>/tw|hant/i.test(v.lang))||zh.find(v=>!/cn|hans/i.test(v.lang))||zh[0]||null;
}
if('speechSynthesis' in window){ speechSynthesis.onvoiceschanged=pickVoice; pickVoice(); }
const CJK=/[\\u3400-\\u9fff\\uf900-\\ufaff]/;
const HAS_DIGIT_OR_CJK=/[\\u3400-\\u9fff\\uf900-\\ufaff0-9]/;
// 英文片段 = 必須是真正的「拉丁字詞」— 符號 /（）: 和數字跟著中文一起唸
// （否則中文句子裡的日期／斜線會被切開，英文口音來回切換 — 實測回饋）
const LATIN_RUN=/[A-Za-z][A-Za-z' -]*[A-Za-z]|[A-Za-z]/g;
function utter(t, zh){ const u=new SpeechSynthesisUtterance(t);
  if(zh){ u.lang='zh-TW'; if(VOICE_ZH){ u.voice=VOICE_ZH; u.lang=VOICE_ZH.lang; } }
  else { u.lang='en-US'; if(VOICE)u.voice=VOICE; }
  u.rate=RATES[SET.rate]||0.95; u.pitch=1.0; speechSynthesis.speak(u); }
function speak(t){ try{
    speechSynthesis.cancel();
    const s=String(t);
    if(!CJK.test(s)){ if(s.trim()) utter(s.trim(), false); return; }
    // 句子裡有中文：只有真正的拉丁字詞才切到英文，其餘（含數字／符號）= 中文
    let last=0, m;
    LATIN_RUN.lastIndex=0;
    while((m=LATIN_RUN.exec(s))!==null){
      const before=s.slice(last, m.index).trim();
      if(before && HAS_DIGIT_OR_CJK.test(before)) utter(before, true);
      const word=m[0].trim(); if(word) utter(word, false);
      last=m.index+m[0].length;
    }
    const tail=s.slice(last).trim();
    if(tail && HAS_DIGIT_OR_CJK.test(tail)) utter(tail, true);
  }catch(e){} }

// 表示危險的字詞 → 強烈震動（還沒聽完語音前就先提醒）
const DANGER=/not safe|unsafe|danger|moving|wait|caution|careful|stop\\b|do not|注意|請等待|請不要|危險|紅燈|有人在你前方/i;

function grabFrame(){
  if(!video.videoWidth) return null;
  const c=document.createElement('canvas');
  c.width=video.videoWidth; c.height=video.videoHeight;
  c.getContext('2d').drawImage(video,0,0,c.width,c.height);
  return c.toDataURL('image/jpeg',0.85);
}
function setHeard(h, followup){
  heardEl.textContent=h?'你說：「'+h+'」'+(followup?'　· 接續上一題':''):'';
  heardEl.classList.toggle('show', !!h);
}

async function describe(){
  if(busy||asking||rec) return;
  const img=grabFrame();
  if(!img){ show('相機還沒準備好，請稍等', {hint:true});
    speak('相機還沒準備好'); vibrate([400]); return; }  // 避免送出全黑的畫面
  busy=true;
  vibrate(60);                      // 確認有點到
  card.classList.add('busy');
  tag.textContent=modeName(MODE);
  setHeard('');
  clearConvo();                                   // 點畫面描述 = 開始新話題
  show('辨識中…', {hint:true});
  try{
    const j=await api('/describe', {image:img, mode:MODE}, 30000);
    const danger=DANGER.test(j.text);
    // 自動模式回來的「室內模式。」前綴移到標籤上顯示（語音照樣唸完整句子）
    let shown=j.text;
    const m=shown.match(/^(\\S{1,6}模式)。\\s*/);
    if(m){ tag.textContent=m[1]; shown=shown.slice(m[0].length); }
    show(shown, {danger:danger, time:j.seconds+' 秒'});
    vibrate(danger?[300,120,300,120,300]:[90]);  // 危險 = 連續震動
    lastText=j.text;
    speak(j.text);
  }catch(e){ fail(e); }
  card.classList.remove('busy');
  busy=false;
  // 斷線時不重試（避免一直唸錯誤），等連線恢復再接著跑
  // 連續模式：等這一句唸完、停頓 0.8 秒再拍下一張，後面的結果不會蓋掉還沒唸完的句子
  if(contChk.checked && online) whenQuiet(()=>{ if(contChk.checked) describe(); }, 800);
}

// ---------------------------------------------------------------------------
// [資服版] 語音提問：按按鈕 → 嗶一聲開始錄音 → 停頓 1.3 秒自動結束（或再按一次）
//   錄音轉成 16kHz PCM 送到筆電，由 Whisper 辨識（聲音不經過第三方雲端）
//   可以說指令（切換到過馬路模式、開始連續、再說一次）或直接問問題
// ---------------------------------------------------------------------------
let actx=null, micStream=null, rec=null, asking=false, resumeCont=false;
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
function beep(freq, dur){ try{
  const o=actx.createOscillator(), g=actx.createGain(), t=actx.currentTime;
  o.frequency.value=freq;
  g.gain.setValueAtTime(0.0001,t); g.gain.exponentialRampToValueAtTime(0.3,t+0.02);
  g.gain.exponentialRampToValueAtTime(0.0001,t+dur);
  o.connect(g); g.connect(actx.destination); o.start(t); o.stop(t+dur+0.02);
}catch(e){} }
function setMic(state){
  mic.className=state||'';
  micText.textContent=state==='listen'?'聆聽中…再按一次結束':state==='think'?'思考中…':'語音提問';
  mic.setAttribute('aria-label', micText.textContent);
}
// 等語音唸完（含排隊中的句子）再停頓 gap 毫秒才執行；最多等 30 秒，避免 iPhone 偶爾卡在「播放中」
function whenQuiet(fn, gap){
  const t0=Date.now();
  const tick=()=>{
    const talking=speechSynthesis.speaking||speechSynthesis.pending;
    if(talking && Date.now()-t0<30000) setTimeout(tick,250); else setTimeout(fn, gap||0);
  };
  setTimeout(tick,300);
}
function restoreCont(){
  if(resumeCont){ contChk.checked=true; whenQuiet(()=>{ if(contChk.checked) describe(); }); }
  resumeCont=false;
}

async function startListen(){
  if(asking||rec) return;
  speechSynthesis.cancel();
  resumeCont=contChk.checked; contChk.checked=false;      // 錄音時先暫停連續描述
  try{
    actx=actx||new (window.AudioContext||window.webkitAudioContext)();
    const resumed=actx.resume();                          // 要在點擊當下呼叫（iOS 規定）
    micStream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true, noiseSuppression:true, autoGainControl:true}});
    await resumed;
  }catch(e){
    show('無法使用麥克風：'+e.message, {danger:true}); speak('無法使用麥克風'); restoreCont(); return;
  }
  vibrate(40); beep(880, 0.12);
  setMic('listen'); setHeard(''); tag.textContent='語音提問';
  show('請說話…', {hint:true});
  const src=actx.createMediaStreamSource(micStream), node=actx.createScriptProcessor(4096,1,1);
  const t0=performance.now();
  rec={src:src, node:node, chunks:[], t0:t0, last:t0, heard:false, floor:0, n:0, peak:0};
  node.onaudioprocess=e=>{
    if(!rec) return;
    const now=performance.now();
    if(now-rec.t0<220) return;                            // 跳過提示音
    const d=e.inputBuffer.getChannelData(0);
    rec.chunks.push(new Float32Array(d));
    let s=0; for(let i=0;i<d.length;i++) s+=d[i]*d[i];
    const rms=Math.sqrt(s/d.length);
    rec.peak=Math.max(rec.peak, rms);
    if(rec.n<3){ rec.floor=Math.max(rec.floor, Math.min(rms,0.03)); rec.n++; return; }   // 估計背景噪音
    if(rms>Math.max(0.015, rec.floor*2.5)){ rec.heard=true; rec.last=now; }
    if((rec.heard && now-rec.last>1300) || (!rec.heard && now-rec.t0>6000) || now-rec.t0>12000) stopListen();
  };
  src.connect(node); node.connect(actx.destination);
}

function stopListen(){
  if(!rec) return;
  const r=rec; rec=null;
  try{ r.node.disconnect(); r.src.disconnect(); }catch(e){}
  try{ micStream.getTracks().forEach(t=>t.stop()); }catch(e){}   // 關掉麥克風，iPhone 喇叭音量才會恢復
  micStream=null;
  vibrate(30); beep(520, 0.12);
  if(!r.chunks.length || r.peak<0.01){
    setMic(''); show('沒有聽到聲音，請再試一次。', {hint:true}); speak('沒有聽到聲音，請再試一次'); restoreCont(); return;
  }
  sendAsk(toPCM16(r.chunks, actx.sampleRate));
}

function toPCM16(chunks, rate){                           // 降到 16kHz、轉 16-bit、base64
  let len=0; chunks.forEach(c=>len+=c.length);
  const all=new Float32Array(len); let o=0; chunks.forEach(c=>{ all.set(c,o); o+=c.length; });
  const ratio=rate/16000, n=Math.floor(len/ratio), pcm=new Int16Array(n);
  for(let i=0;i<n;i++){
    const a=Math.floor(i*ratio), b=Math.min(len, Math.floor((i+1)*ratio));
    let s=0; for(let k=a;k<b;k++) s+=all[k];
    const v=Math.max(-1, Math.min(1, b>a?s/(b-a):0));
    pcm[i]=v<0?v*32768:v*32767;
  }
  const u8=new Uint8Array(pcm.buffer); let bin='';
  for(let i=0;i<u8.length;i+=0x8000) bin+=String.fromCharCode.apply(null, u8.subarray(i,i+0x8000));
  return btoa(bin);
}

async function sendAsk(audio){
  asking=true; setMic('think'); card.classList.add('busy');
  show('思考中…', {hint:true});
  if(convoAt && Date.now()-convoAt>CONVO_TTL) clearConvo();
  try{
    const j=await api('/ask', {audio:audio, image:grabFrame(), mode:MODE, history:convo}, 40000);
    setHeard(j.heard, j.followup);
    if(j.action==='answer' && !j.gated){            // 追問就接在後面，否則從這題重新開始
      if(!j.followup) convo=[];
      convo.push({q:j.heard, a:j.text, t:j.used});
      convo=convo.slice(-3); convoAt=Date.now();
    }
    if(j.action==='switch') setMode(j.mode, false);
    if(j.action==='continuous'){ resumeCont=!!j.on; }
    if(j.action==='setting'){
      const msg=changeSetting(j.key, j.value);
      show(msg); speak(msg);                        // 用新的語速說，馬上聽得出差別
    }else if(j.action==='repeat'){
      show(lastText||'還沒有可以重複的內容。'); speak(lastText||'還沒有可以重複的內容');
    }else{
      const danger=j.action==='answer' && DANGER.test(j.text);
      if(j.action==='answer'){ tag.textContent=modeName(j.used)||'語音提問'; lastText=j.text; }
      show(j.text, {danger:danger, hint:j.action==='none', time:j.seconds?j.seconds+' 秒':''});
      vibrate(danger?[300,120,300,120,300]:[90]);
      speak(j.text);
    }
  }catch(e){ fail(e); }
  card.classList.remove('busy'); setMic(''); asking=false;
  restoreCont();
}

mic.addEventListener('click', ()=>{ firstStart(true); if(rec) stopListen(); else startListen(); });

// ---------------------------------------------------------------------------
// [資服版] 搖一搖提問：短時間內用力搖兩下 → 開始錄音（一般走路的晃動不會觸發）
//   iPhone 要先在點擊時取得「動作與方向」權限，所以在第一次點畫面時一併請求
// ---------------------------------------------------------------------------
let motionOn=false, jolts=[], lastAcc=null, shakeCooldown=0;
function enableMotion(){
  if(motionOn || !CAN_SHAKE) return;
  const listen=()=>{ if(motionOn) return; motionOn=true; window.addEventListener('devicemotion', onMotion); };
  if(typeof DeviceMotionEvent.requestPermission==='function'){
    DeviceMotionEvent.requestPermission().then(s=>{ if(s==='granted') listen(); }).catch(()=>{});
  }else listen();
}
function onMotion(e){
  const a=e.accelerationIncludingGravity; if(!a || a.x==null) return;
  if(lastAcc){
    const d=Math.abs(a.x-lastAcc.x)+Math.abs(a.y-lastAcc.y)+Math.abs(a.z-lastAcc.z);
    const now=Date.now();
    if(d>35){ jolts=jolts.filter(t=>now-t<1000); jolts.push(now); }
    if(jolts.length>=3 && now>shakeCooldown){
      jolts=[]; shakeCooldown=now+2500;
      if(SET.shake && started && !rec && !asking) startListen();
    }
  }
  lastAcc={x:a.x, y:a.y, z:a.z};
}

function firstStart(quiet){
  if(started) return; started=true;
  keepAwake();
  // 在點擊當下解鎖音效與動作感測，之後搖一搖觸發錄音時才能出聲、收音
  try{ actx=actx||new (window.AudioContext||window.webkitAudioContext)(); actx.resume(); }catch(e){}
  if(SET.shake) enableMotion();
  if(!quiet) speak('已就緒，點一下畫面描述，或按下方按鈕、搖一搖手機用說的。');
}
// 錄音中點畫面任何地方 = 結束錄音
tap.addEventListener('click', ()=>{ firstStart(); if(rec){ stopListen(); return; } describe(); });
contChk.addEventListener('change', ()=>{
  firstStart();
  speak(contChk.checked?'連續模式已開啟':'連續模式已關閉');
  if(contChk.checked) describe();
});
// 螢幕關閉再回來時，重新取得螢幕常亮鎖
document.addEventListener('visibilitychange', ()=>{ if(document.visibilityState==='visible') keepAwake(); });
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif self.path == "/ping":            # [資服版] 手機用來確認連線
            self._send(200, json.dumps({"ok": True}))
        elif self.path in _STATIC:            # [資服版] 加入主畫面用的 App 設定與圖示
            name, ctype = _STATIC[self.path]
            with open(os.path.join(_STATIC_DIR, name), "rb") as f:
                self._send(200, f.read(), ctype)
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path == "/ask":
            return self._ask()
        if self.path != "/describe":
            self._send(404, json.dumps({"error": "not found"}))
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n).decode("utf-8"))
            image = _decode_image(payload["image"])
            mode = payload.get("mode", config.DEFAULT_MODE)

            reason = _quality_gate(image, mode)
            if reason:
                self._send(200, json.dumps({"text": reason, "seconds": 0, "gated": True}))
                return

            global _SEQ
            t0 = time.perf_counter()
            from app import pipeline          # 所有模式的邏輯集中在這裡（OCR／過馬路混合流程／自動）
            with GPU_LOCK:
                text, _mode_used = pipeline.describe(BACKEND, image, mode)
            dt = round(time.perf_counter() - t0, 1)
            print(f"[{mode}] ({dt}s) {text}")
            if SAVE_DIR:
                _SEQ += 1
                try:
                    from app import quality as _q
                    _blur = f"_blur{_q.blur_score(image):.0f}"
                except Exception:
                    _blur = ""
                fn = os.path.join(SAVE_DIR, f"{_SEQ:03d}_{mode}_ok{_blur}.jpg")
                image.save(fn)
                with open(os.path.join(SAVE_DIR, "log.csv"), "a", encoding="utf-8") as f:
                    f.write(f'{_SEQ:03d},{mode},{dt},"{text}"\n')
                print(f"      saved → {fn}")
            self._send(200, json.dumps({"text": text, "seconds": dt}))
        except Exception as e:
            self._send(500, json.dumps({"error": str(e)}))

    def _ask(self):
        """[資服版] 語音指令／語音提問：錄音 → Whisper → 指令或問題（見 app/voice.py）"""
        try:
            from app import voice
            if not VOICE_READY:
                self._send(200, json.dumps({"action": "none", "heard": "",
                                            "text": "語音功能沒有啟動。"}))
                return
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n).decode("utf-8"))
            mode = payload.get("mode", config.DEFAULT_MODE)
            pcm = np.frombuffer(base64.b64decode(payload["audio"]), dtype="<i2").astype(np.float32) / 32768.0

            t0 = time.perf_counter()
            with GPU_LOCK:
                heard = voice.transcribe(pcm)
            print(f"[ask] 聽到：{heard or '（沒有內容）'}")
            if not heard:
                self._send(200, json.dumps({"action": "none", "heard": "",
                                            "text": "沒有聽清楚，請再說一次。"}))
                return

            cmd = voice.parse_command(heard)
            if cmd:
                kind, arg = cmd
                out = {"action": kind, "heard": heard, "text": voice.command_reply(kind, arg, mode)}
                if kind == "switch":
                    out["mode"] = arg
                elif kind == "continuous":
                    out["on"] = arg
                elif kind == "setting":           # 由手機調整並決定要說的話
                    out["key"], out["value"] = arg
                print(f"[ask] 指令：{kind} {arg}")
                self._send(200, json.dumps(out))
                return

            if not payload.get("image"):
                self._send(200, json.dumps({"action": "none", "heard": heard,
                                            "text": "相機還沒準備好。"}))
                return
            image = _decode_image(payload["image"])
            # 對話歷史由手機保存、每次一起送來；是追問才沿用，否則當成新話題
            history = payload.get("history") or []
            followup = bool(history) and voice.is_followup(heard)
            if not followup:
                history = []
            target = voice.question_mode(heard)
            reason = _quality_gate(image, target if target != "vqa" else "surrounding")
            if reason:
                self._send(200, json.dumps({"action": "answer", "heard": heard, "text": reason,
                                            "seconds": 0, "gated": True, "followup": followup}))
                return
            with GPU_LOCK:
                text, used = voice.answer(BACKEND, image, heard, mode, history)
            dt = round(time.perf_counter() - t0, 1)
            print(f"[ask:{used}{' 追問' if followup else ''}] ({dt}s) {text}")
            self._send(200, json.dumps({"action": "answer", "heard": heard, "text": text,
                                        "used": used, "seconds": dt, "followup": followup}))
        except Exception as e:
            self._send(500, json.dumps({"error": str(e)}))

    def log_message(self, *a):
        pass


def _decode_image(data_url):
    b64 = data_url.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _quality_gate(image, mode):
    """畫面品質檢查：畫面模糊或太暗 → 不推論。擋下時回傳要唸的提示，通過回傳 None。"""
    if not config.QUALITY_GATE:
        return None
    from app import quality
    q = config.get_quality(mode)
    ok, reason = quality.assess(image, q)
    try:      # [資服版] 印出實際分數，方便依真實畫面調整門檻
        score = (f"清晰度 {quality.blur_score(image):.0f}（門檻 {q['min_blur']:.0f}）"
                 f" 亮度 {quality.brightness(image):.0f} 解析度 {image.size[0]}x{image.size[1]}")
    except Exception:
        score = ""
    print(f"      {score}")
    if ok:
        return None
    print(f"[{mode}] (skip) {reason}")
    if SAVE_DIR:          # [資服版] 被擋下的畫面也存起來，用來校準門檻
        global _SEQ
        _SEQ += 1
        blur = quality.blur_score(image)
        image.save(os.path.join(SAVE_DIR, f"{_SEQ:03d}_{mode}_skip_blur{blur:.0f}.jpg"))
    return reason


def get_lan_ip():
    """找出連外網路介面的 IP（離線也能用 — UDP connect 不會真的送出封包）"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


class _Server(ThreadingHTTPServer):
    # http.server 預設開 SO_REUSEADDR：在 Windows 上會讓兩個程式同時綁同一個連接埠也不報錯，
    # 改成獨佔，連接埠被佔用時才會正確失敗、改試下一個
    allow_reuse_address = os.name != "nt"

    def server_bind(self):
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def bind_server(host, port, tries=20):
    """[資服版] 連接埠不能用（被佔用，或被 Windows／Hyper-V 保留而沒有權限）就往後一個一個試。"""
    last = None
    for p in range(port, port + tries):
        try:
            srv = _Server((host, p), Handler)
            if p != port:
                print(f"[info] 改用連接埠 {p}")
            return srv, p
        except OSError as e:
            last = e
            reason = "被其他程式佔用" if getattr(e, "winerror", None) == 10048 or e.errno == 98 else "沒有權限使用"
            print(f"[warn] 連接埠 {p} {reason}（{e.strerror or e}），改試 {p + 1}")
    raise SystemExit(f"[錯誤] 找不到可用的連接埠（試過 {port}～{port + tries - 1}）：{last}")


def get_tailscale_ip():
    """[資服版] 有裝 Tailscale 就回傳這台電腦的 100.x.x.x 位址，沒有就回傳 None"""
    try:
        r = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=5)
        ip = r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else ""
        return ip or None
    except Exception:
        return None


def get_tailscale_cert(certdir):
    """[資服版] Tailscale 管理頁開了 HTTPS 憑證的話，向 Tailscale 取得正式憑證（手機不會再跳警告，
    加到主畫面也能直接開）。回傳 (網域, cert, key)；沒開或失敗回傳 None，改用自簽憑證。"""
    try:
        r = subprocess.run(["tailscale", "status", "--json"], capture_output=True, timeout=10)
        domains = json.loads(r.stdout.decode("utf-8", "replace")).get("CertDomains") or []
    except Exception:
        return None
    if not domains:
        return None
    domain = domains[0]
    os.makedirs(certdir, exist_ok=True)
    cert = os.path.join(certdir, "tailscale.crt")
    key = os.path.join(certdir, "tailscale.key")
    try:   # 已有且未過期時 tailscale 會直接沿用，快過期才重新申請
        r = subprocess.run(["tailscale", "cert", "--cert-file", cert, "--key-file", key, domain],
                           capture_output=True, timeout=120)
    except Exception as e:
        print(f"[warn] 取得 Tailscale 憑證失敗，改用自簽憑證：{e}")
        return None
    if r.returncode != 0 or not (os.path.exists(cert) and os.path.exists(key)):
        print(f"[warn] 取得 Tailscale 憑證失敗，改用自簽憑證：{r.stderr.decode('utf-8', 'replace').strip()}")
        return None
    return domain, cert, key


def _make_cert_python(cert, key, ip):
    """[資服版] 用 cryptography 套件產生自簽憑證（Windows 通常沒有 openssl 指令）"""
    import datetime
    import ipaddress
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    pk = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "vlm-assistant")])
    now = datetime.datetime.now(datetime.timezone.utc)
    san = x509.SubjectAlternativeName([
        x509.IPAddress(ipaddress.ip_address(ip)),
        x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
        x509.DNSName("localhost"),
    ])
    c = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
         .public_key(pk.public_key()).serial_number(x509.random_serial_number())
         .not_valid_before(now - datetime.timedelta(days=1))
         .not_valid_after(now + datetime.timedelta(days=825))
         .add_extension(san, critical=False)
         .sign(pk, hashes.SHA256()))
    with open(key, "wb") as f:
        f.write(pk.private_bytes(serialization.Encoding.PEM,
                                 serialization.PrivateFormat.TraditionalOpenSSL,
                                 serialization.NoEncryption()))
    with open(cert, "wb") as f:
        f.write(c.public_bytes(serialization.Encoding.PEM))


def ensure_cert(certdir, ip):
    """還沒有憑證時，用 openssl 產生自簽憑證（把 IP 放進 SAN）"""
    os.makedirs(certdir, exist_ok=True)
    cert = os.path.join(certdir, "cert.pem")
    key = os.path.join(certdir, "key.pem")
    if os.path.exists(cert) and os.path.exists(key):
        return cert, key
    print(f"[info] 產生自簽憑證 → {certdir}/ ...")
    try:
        subprocess.run([
            "openssl", "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", key, "-out", cert, "-days", "825", "-nodes",
            "-subj", "/CN=vlm-assistant",
            "-addext", f"subjectAltName=IP:{ip},IP:127.0.0.1,DNS:localhost",
        ], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        _make_cert_python(cert, key, ip)      # [資服版] 沒有 openssl 指令時改用 cryptography 套件
    return cert, key


def main():
    global BACKEND, SAVE_DIR, VOICE_READY
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-voice", action="store_true",
                    help="[資服版] 不載入 Whisper（關閉語音指令／語音提問）")
    ap.add_argument("--backend", default=None, help="gemma_hf | smolvlm | gemma_npu")
    ap.add_argument("--device", default=None, help="cpu | igpu")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=0, help="default 8000 (http) / 8443 (https)")
    ap.add_argument("--https", action="store_true",
                    help="開啟 TLS（自簽憑證），手機的相機才能開啟")
    ap.add_argument("--certdir", default="certs")
    ap.add_argument("--save-frames", default=None,
                    help="把每張畫面和系統說的話存到這個資料夾（例如 results/field_1004）")
    ap.add_argument("--no-quality-gate", action="store_true",
                    help="關閉畫面品質檢查（模糊／太暗）— 展示或測試用")
    args = ap.parse_args()
    port = args.port or (8443 if args.https else 8000)

    if args.no_quality_gate:
        config.QUALITY_GATE = False
        print("[info] frame-quality gate = OFF")

    if args.save_frames:
        SAVE_DIR = args.save_frames
        os.makedirs(SAVE_DIR, exist_ok=True)
        print(f"[info] 會把每張畫面和系統說的話存到：{SAVE_DIR}/")

    print(f"[info] 載入模型（{args.backend or config.BACKEND}）...")
    BACKEND = build_backend(backend=args.backend, device=args.device)
    BACKEND.load()
    BACKEND.warmup()
    print(f"[info] 載入完成，實際使用：{BACKEND.name}")

    if not args.no_voice:
        print(f"[info] 載入語音辨識（{config.ASR_MODEL}）...")
        try:
            from app import voice
            voice.load_asr()
            VOICE_READY = True
            print("[info] 語音辨識載入完成")
        except Exception as e:
            print(f"[warn] 語音辨識載入失敗，語音功能關閉：{e}")

    ip = args.host if args.host not in ("0.0.0.0", "") else get_lan_ip()
    srv, port = bind_server(args.host, port)
    scheme = "http"
    ts = None
    if args.https:
        ts = get_tailscale_cert(args.certdir)
        cert, key = (ts[1], ts[2]) if ts else ensure_cert(args.certdir, ip)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(cert, key)
        srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
        scheme = "https"

    if ts:
        print(f"[info] 伺服器已啟動（Tailscale 正式憑證）")
        print(f"       手機開這個網址（在哪個網路都一樣，不會有憑證警告）：")
        print(f"       https://{ts[0]}:{port}")
        print(f"       想像 App 一樣用：Safari 打開後按「分享」→「加入主畫面」")
    else:
        print(f"[info] 伺服器已啟動：{scheme}://{ip}:{port}")
        print(f"       手機和電腦在同一個網路時開這個網址")
        ts_ip = get_tailscale_ip()
        if ts_ip:
            print(f"       手機在外面（用 Tailscale）時開：{scheme}://{ts_ip}:{port}")
        if args.https:
            print("       （自簽憑證：第一次連線按「進階」→「繼續前往」）")
    print("       按 Ctrl+C 停止")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[info] 伺服器已關閉")
        srv.shutdown()


if __name__ == "__main__":
    main()
