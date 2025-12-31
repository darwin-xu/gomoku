"""Web dashboard to manage Gomoku checkpoints, evaluation, training, and telemetry.

Run:
  python -m python_ai.dashboard --port 8787

Features:
- Lists checkpoints under python_ai/checkpoints using python_ai.inspect.inspect_checkpoint
- Cross-compares checkpoints via arena matches (python_ai.eval-style MCTS vs MCTS)
- Starts training jobs (python -m python_ai.train ...) with configurable options
- Receives training telemetry via POST (best-effort) and plots it

This is intentionally lightweight: Flask + a single HTML page + JSON APIs.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

from flask import Flask, Response, jsonify, request

from python_ai.inspect import inspect_checkpoint


ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = (ROOT / "python_ai" / "checkpoints").resolve()
def _now() -> float:
    return time.time()


def _safe_relpath(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


_DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Gomoku Models Dashboard</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 16px; }
    h2 { margin: 8px 0 12px; }
    .row { display: flex; gap: 12px; flex-wrap: wrap; align-items: stretch; }
    .card { border: 1px solid #ddd; border-radius: 10px; padding: 12px 14px; min-width: 320px; }
    .card.wide { min-width: 680px; flex: 1; }
    .small { color: #555; font-size: 12px; }
    button { padding: 6px 10px; }
    input[type=text], input[type=number] { padding: 6px 8px; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border-bottom: 1px solid #eee; padding: 6px 8px; text-align: left; font-size: 13px; }
    th { background: #fafafa; position: sticky; top: 0; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
    canvas { border: 1px solid #ddd; border-radius: 10px; max-width: 100%; }
    .pill { display: inline-block; padding: 2px 8px; border-radius: 999px; background: #f1f1f1; font-size: 12px; }
  </style>
</head>
<body>
  <h2>Gomoku Models Dashboard</h2>

  <div class="row">
    <div class="card wide">
      <div class="row" style="align-items:center; justify-content:space-between;">
        <div>
          <strong>Checkpoints</strong>
          <span class="small" id="modelsMeta"></span>
        </div>
        <div>
          <button id="refreshModels">Refresh</button>
        </div>
      </div>
      <div style="max-height: 340px; overflow: auto; margin-top: 8px;">
        <table id="modelsTable">
          <thead>
            <tr>
              <th style="width:28px"></th>
              <th>File</th>
              <th>Episode</th>
              <th>Channels</th>
              <th>Blocks</th>
              <th>Params</th>
              <th>Replay</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>
      <div class="small" style="margin-top:8px;">Select 2+ models to run a tournament.</div>
      <div class="small" style="margin-top:10px;">Selected model details</div>
      <pre id="modelInspect" class="mono" style="max-height: 220px; overflow:auto; border:1px solid #eee; border-radius:10px; padding:10px; background:#fafafa;"></pre>
    </div>

    <div class="card">
      <strong>Evaluate / Compare</strong>
      <div class="small" style="margin-top:6px;">Runs pairwise arena matches and assigns an Elo-like score.</div>
      <div style="margin-top:10px; display:flex; gap:8px; flex-wrap:wrap;">
        <label>Games <input id="evalGames" type="number" value="20" min="2" step="2" style="width:84px"/></label>
        <label>Sims <input id="evalSims" type="number" value="64" min="1" step="1" style="width:84px"/></label>
        <label>C_PUCT <input id="evalCPuct" type="number" value="1.5" min="0.1" step="0.1" style="width:84px"/></label>
      </div>
      <div style="margin-top:10px;">
        <button id="runEval">Run Tournament</button>
        <span class="pill" id="evalStatus">idle</span>
      </div>
      <div style="margin-top:10px; max-height:220px; overflow:auto;">
        <table id="evalTable">
          <thead>
            <tr><th>Model</th><th>Elo</th><th>W</th><th>L</th><th>D</th></tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>
    </div>

    <div class="card wide">
      <div class="row" style="align-items:center; justify-content:space-between;">
        <strong>Tournament Game Viewer</strong>
        <button id="refreshTournaments">Refresh</button>
      </div>
      <div class="small" style="margin-top:6px;">Review saved tournament games move by move.</div>
      <div class="row" style="margin-top:10px; gap:10px; align-items:flex-start;">
        <div style="min-width:280px;">
          <label class="small">Tournament
            <select id="tournamentSelect" style="min-width:240px;"></select>
          </label>
          <div style="margin-top:8px;">
            <label class="small">Game
              <select id="gameSelect" style="min-width:240px;"></select>
            </label>
          </div>
          <div style="margin-top:10px; max-height:180px; overflow:auto;">
            <table id="tournamentGamesTable" style="font-size:12px;">
              <thead>
                <tr><th>#</th><th>Match</th><th>Black</th><th>Winner</th><th>Moves</th></tr>
              </thead>
              <tbody></tbody>
            </table>
          </div>
        </div>
        <div>
          <canvas id="gameViewerCanvas" width="400" height="400"></canvas>
          <div style="margin-top:8px; display:flex; gap:8px; align-items:center;">
            <button id="gameViewerPrev">◀ Prev</button>
            <input id="gameViewerSlider" type="range" min="0" max="0" value="0" step="1" style="flex:1; min-width:150px;"/>
            <button id="gameViewerNext">Next ▶</button>
          </div>
          <div class="small" style="margin-top:6px;" id="gameViewerMeta">Select a tournament and game to view.</div>
        </div>
      </div>
    </div>

    <div class="card wide">
      <strong>Training</strong>
      <div class="small" style="margin-top:6px;">Starts python_ai.train as a background job.</div>
      <div class="row" style="margin-top:10px;">
        <label>Model path <input id="trainModelPath" type="text" value="./python_ai/checkpoints/policy_value.pt" style="width:420px"/></label>
        <label>Resume <input id="trainResume" type="checkbox" checked /></label>
      </div>
      <div class="row" style="margin-top:8px;">
        <label>Episodes <input id="trainEpisodes" type="number" value="200" min="1" step="1"/></label>
        <label>Games/ep <input id="trainGamesPerEp" type="number" value="1" min="1" step="1"/></label>
        <label>Sims <input id="trainSims" type="number" value="64" min="1" step="1"/></label>
        <label>Batch <input id="trainBatch" type="number" value="128" min="1" step="1"/></label>
        <label>Batches/ep <input id="trainBatchesPerEp" type="number" value="8" min="1" step="1"/></label>
        <label>LR <input id="trainLR" type="number" value="0.001" step="0.0001"/></label>
      </div>
      <div class="row" style="margin-top:8px;">
        <label>Replay size <input id="trainReplaySize" type="number" value="50000" min="1000" step="1000"/></label>
        <label>Min replay <input id="trainMinReplay" type="number" value="5000" min="0" step="500"/></label>
        <label>Temp <input id="trainTemp" type="number" value="1.0" step="0.05"/></label>
        <label>Temp decay <input id="trainTempDecay" type="number" value="0.995" step="0.001"/></label>
        <label>Min temp <input id="trainMinTemp" type="number" value="0.1" step="0.01"/></label>
      </div>
      <div class="row" style="margin-top:8px;">
        <label>Channels <input id="trainChannels" type="number" value="128" min="8" step="8"/></label>
        <label>Blocks <input id="trainBlocks" type="number" value="8" min="1" step="1"/></label>
        <label>Augment <input id="trainAugment" type="checkbox"/></label>
      </div>
      <div class="row" style="margin-top:8px;">
        <label>Save every <input id="trainSaveEvery" type="number" value="25" min="1" step="1"/></label>
        <label>Value loss w <input id="trainValueLossW" type="number" value="1.0" min="0" step="0.1"/></label>
        <label>C_PUCT <input id="trainCPuct" type="number" value="1.5" min="0.1" step="0.1"/></label>
        <label>Dir α <input id="trainDirAlpha" type="number" value="0.3" min="0" step="0.05"/></label>
        <label>Dir frac <input id="trainDirFrac" type="number" value="0.25" min="0" step="0.05"/></label>
      </div>
      <div class="row" style="margin-top:8px;">
        <label>Torch threads <input id="trainTorchThreads" type="number" value="0" min="0" step="1"/></label>
        <label>DL workers <input id="trainDLWorkers" type="number" value="0" min="0" step="1"/></label>
        <label class="small">Replay path <input id="trainReplayPath" type="text" value="" placeholder="(default: <model>.replay.npz)" style="width:240px"/></label>
        <label class="small">CoreML path <input id="trainCoreMLPath" type="text" value="" placeholder="(optional)" style="width:200px"/></label>
      </div>
      <div style="margin-top:10px;">
        <button id="startTrain">Start Training</button>
        <span class="pill" id="trainStatus">idle</span>
      </div>
      <div class="small" style="margin-top:8px;">Jobs</div>
      <div style="max-height: 180px; overflow:auto; margin-top: 6px;">
        <table id="jobsTable">
          <thead>
            <tr><th>ID</th><th>Type</th><th>Status</th><th>Started</th><th>Action</th></tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>
      <div class="small" style="margin-top:8px;">Selected job log</div>
      <pre id="jobLog" class="mono" style="max-height: 220px; overflow:auto; border:1px solid #eee; border-radius:10px; padding:10px; background:#fafafa;"></pre>
    </div>

    <div class="card wide">
      <div class="row" style="align-items:center; justify-content:space-between;">
        <strong>Telemetry</strong>
        <div>
          <label class="small">Auto-refresh <input id="teleAuto" type="checkbox" checked /></label>
          <label class="small" style="margin-left:10px">Interval (ms) <input id="teleInterval" type="number" value="1000" min="200" step="100" style="width:90px"/></label>
          <button id="teleRefresh" style="margin-left:10px">Refresh</button>
        </div>
      </div>
      <div class="small" style="margin-top:6px;">Shows in-memory telemetry posted by the selected training job.</div>
      <div class="row" style="margin-top:8px; align-items:center; gap:8px;">
        <label class="small">Training job
          <select id="teleJobSelect" style="min-width:180px;"></select>
        </label>
        <span class="small" id="teleJobHint"></span>
      </div>
      <div class="row" style="margin-top:10px;">
        <div class="card" style="min-width:240px;">
          <div><strong>Status</strong>: <span id="teleStatus">-</span></div>
          <div><strong>Episode</strong>: <span id="teleEpisode">-</span></div>
          <div><strong>Replay</strong>: <span id="teleReplay">-</span></div>
          <div class="small" id="teleUpdated">-</div>
        </div>
      </div>
      <h3 style="margin:10px 0 6px;">Loss (policy/value)</h3>
      <canvas id="loss" width="1100" height="280"></canvas>
      <h3 style="margin:10px 0 6px;">Episode time (sec)</h3>
      <canvas id="time" width="1100" height="220"></canvas>
      <h3 style="margin:10px 0 6px;">Policy snapshot (latest)</h3>
      <div class="row" style="gap:10px; align-items:flex-start; flex-wrap:wrap;">
        <canvas id="policyCanvas" width="360" height="360"></canvas>
        <div class="small" id="policyMeta" style="min-width:220px;">Waiting for telemetry...</div>
      </div>
      <h3 style="margin:10px 0 6px;">Self-play trace (latest game)</h3>
      <div class="row" style="gap:10px; align-items:flex-start; flex-wrap:wrap;">
        <canvas id="traceCanvas" width="360" height="360"></canvas>
        <div style="display:flex; flex-direction:column; gap:6px; min-width:240px;">
          <input id="traceSlider" type="range" min="0" max="0" value="0" step="1" />
          <div class="small" id="traceMeta">Waiting for telemetry...</div>
        </div>
      </div>
    </div>
  </div>

<script>
const modelsTableBody = document.querySelector('#modelsTable tbody');
const evalTableBody = document.querySelector('#evalTable tbody');
const jobsTableBody = document.querySelector('#jobsTable tbody');
const jobLogEl = document.getElementById('jobLog');
const modelInspectEl = document.getElementById('modelInspect');

let models = [];
let selectedJobId = null;
let runningTrainJobs = [];

function fmtInt(n) {
  if (n === null || n === undefined) return '-';
  const x = Number(n);
  if (!Number.isFinite(x)) return '-';
  return x.toLocaleString();
}

async function apiGet(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

async function apiPost(path, body) {
  const res = await fetch(path, { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body || {}) });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

function renderModels() {
  modelsTableBody.innerHTML = '';
  for (const m of models) {
    const tr = document.createElement('tr');
    const chk = document.createElement('input');
    chk.type = 'checkbox';
    chk.dataset.path = m.path;

    const td0 = document.createElement('td');
    td0.appendChild(chk);

    const td1 = document.createElement('td');
    td1.textContent = m.relpath;
    td1.className = 'mono';
    td1.style.cursor = 'pointer';
    td1.title = 'Click to view full inspect details';
    td1.addEventListener('click', () => {
      try {
        modelInspectEl.textContent = JSON.stringify(m.inspect || {}, null, 2);
      } catch {
        modelInspectEl.textContent = String(m.inspect || '');
      }
    });

    const td2 = document.createElement('td');
    td2.textContent = m.inspect?.episode ?? '-';

    const td3 = document.createElement('td');
    td3.textContent = m.inspect?.channels ?? '-';

    const td4 = document.createElement('td');
    td4.textContent = m.inspect?.blocks ?? '-';

    const td5 = document.createElement('td');
    td5.textContent = fmtInt(m.inspect?.param_count);

    const td6 = document.createElement('td');
    td6.textContent = m.inspect?.replay_sidecar ? 'yes' : 'no';

    tr.append(td0, td1, td2, td3, td4, td5, td6);
    modelsTableBody.appendChild(tr);
  }
  document.getElementById('modelsMeta').textContent = `(${models.length} found in python_ai/checkpoints)`;
}

async function refreshModels() {
  models = (await apiGet('/api/models')).models;
  renderModels();
}

function getSelectedModelPaths() {
  const checks = Array.from(document.querySelectorAll('#modelsTable tbody input[type=checkbox]'));
  return checks.filter(c => c.checked).map(c => c.dataset.path);
}

async function runEval() {
  const paths = getSelectedModelPaths();
  if (paths.length < 2) {
    alert('Select at least 2 models.');
    return;
  }
  document.getElementById('evalStatus').textContent = 'running';
  const job = await apiPost('/api/eval/tournament', {
    models: paths,
    games: Number(document.getElementById('evalGames').value) || 20,
    sims: Number(document.getElementById('evalSims').value) || 64,
    c_puct: Number(document.getElementById('evalCPuct').value) || 1.5,
  });
  selectedJobId = job.job_id;
  await refreshJobs();
}

async function startTrain() {
  document.getElementById('trainStatus').textContent = 'starting';
  const body = {
    model_path: document.getElementById('trainModelPath').value,
    resume: document.getElementById('trainResume').checked,
    episodes: Number(document.getElementById('trainEpisodes').value) || 200,
    games_per_episode: Number(document.getElementById('trainGamesPerEp').value) || 1,
    simulations: Number(document.getElementById('trainSims').value) || 64,
    batch_size: Number(document.getElementById('trainBatch').value) || 128,
    batches_per_episode: Number(document.getElementById('trainBatchesPerEp').value) || 8,
    lr: Number(document.getElementById('trainLR').value) || 1e-3,
    replay_size: Number(document.getElementById('trainReplaySize').value) || 50000,
    min_replay: Number(document.getElementById('trainMinReplay').value) || 5000,
    temperature: Number(document.getElementById('trainTemp').value) || 1.0,
    temperature_decay: Number(document.getElementById('trainTempDecay').value) || 0.995,
    min_temperature: Number(document.getElementById('trainMinTemp').value) || 0.1,
    channels: Number(document.getElementById('trainChannels').value) || 128,
    blocks: Number(document.getElementById('trainBlocks').value) || 8,
    augment: document.getElementById('trainAugment').checked,
    save_every: Number(document.getElementById('trainSaveEvery').value) || 25,
    value_loss_weight: Number(document.getElementById('trainValueLossW').value) || 1.0,
    c_puct: Number(document.getElementById('trainCPuct').value) || 1.5,
    dirichlet_alpha: Number(document.getElementById('trainDirAlpha').value) || 0.3,
    dirichlet_frac: Number(document.getElementById('trainDirFrac').value) || 0.25,
    torch_threads: Number(document.getElementById('trainTorchThreads').value) || 0,
    dataloader_workers: Number(document.getElementById('trainDLWorkers').value) || 0,
    replay_path: document.getElementById('trainReplayPath').value,
    coreml_path: document.getElementById('trainCoreMLPath').value,
  };
  const job = await apiPost('/api/train/start', body);
  selectedJobId = job.job_id;
  await refreshJobs();
  document.getElementById('trainStatus').textContent = 'running';
}

async function refreshJobs() {
  const data = await apiGet('/api/jobs');
  runningTrainJobs = data.jobs.filter(j => j.type === 'train' && j.status === 'running');
  jobsTableBody.innerHTML = '';
  for (const j of data.jobs) {
    const tr = document.createElement('tr');
    const tdId = document.createElement('td');
    tdId.textContent = j.id;
    tdId.className = 'mono';
    const tdType = document.createElement('td');
    tdType.textContent = j.type;
    const tdStatus = document.createElement('td');
    tdStatus.textContent = j.status;
    const tdStarted = document.createElement('td');
    tdStarted.textContent = new Date(j.started_at * 1000).toLocaleTimeString();

    const tdAction = document.createElement('td');
    const btnView = document.createElement('button');
    btnView.textContent = 'View';
    btnView.addEventListener('click', async () => { selectedJobId = j.id; await refreshLog(); });
    tdAction.appendChild(btnView);

    if (j.type === 'train' && j.status === 'running') {
      const btnStop = document.createElement('button');
      btnStop.style.marginLeft = '6px';
      btnStop.textContent = 'Stop';
      btnStop.addEventListener('click', async () => {
        await apiPost(`/api/jobs/${j.id}/stop`, {});
        await refreshJobs();
      });
      tdAction.appendChild(btnStop);
    }

    tr.append(tdId, tdType, tdStatus, tdStarted, tdAction);
    jobsTableBody.appendChild(tr);
  }
  if (selectedJobId) await refreshLog();

  // Populate telemetry job selector with running training jobs
  const teleSel = document.getElementById('teleJobSelect');
  const prevSel = teleSel.value;
  teleSel.innerHTML = '';
  for (const j of runningTrainJobs) {
    const opt = document.createElement('option');
    opt.value = j.id;
    opt.textContent = `${j.id} (${j.status})`;
    teleSel.appendChild(opt);
  }
  await appendTelemetryJobOptions(teleSel, prevSel);
  document.getElementById('teleJobHint').textContent = teleSel.value ? '' : 'Select a telemetry job id (CLI runs auto-appear once they post).';

  // Eval status pill
  const evalJob = data.jobs.find(j => j.id === selectedJobId && j.type === 'eval');
  if (evalJob) document.getElementById('evalStatus').textContent = evalJob.status;
}

async function appendTelemetryJobOptions(teleSel, prevSel) {
  const existing = new Set(Array.from(teleSel.options).map(o => o.value));
  try {
    const data = await apiGet('/api/telemetry/jobs');
    const jobs = data.jobs || [];
    for (const j of jobs) {
      if (!j.job_id || existing.has(j.job_id)) continue;
      const opt = document.createElement('option');
      opt.value = j.job_id;
      opt.textContent = `${j.job_id} (telemetry)`;
      teleSel.appendChild(opt);
    }
  } catch (e) {
    console.error('Failed to load telemetry jobs', e);
  }

  // Choose previously selected if still present; else keep current; else first option
  if (prevSel && Array.from(teleSel.options).some(o => o.value === prevSel)) {
    teleSel.value = prevSel;
    selectedJobId = prevSel;
  } else if (!teleSel.value && teleSel.options.length > 0) {
    teleSel.value = teleSel.options[0].value;
    selectedJobId = teleSel.value;
  }

  if (teleSel.options.length === 0) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = 'No telemetry jobs yet';
    teleSel.appendChild(opt);
    teleSel.value = '';
  }
}

async function refreshLog() {
  if (!selectedJobId) return;
  const data = await apiGet(`/api/jobs/${selectedJobId}/log?limit=400`);
  jobLogEl.textContent = data.text || '';

  // If this is an eval job and it has results, render them
  if (data.job && data.job.type === 'eval' && data.job.result) {
    renderEvalTable(data.job.result);
  }
}

function renderEvalTable(result) {
  const rows = result.ranking || [];
  evalTableBody.innerHTML = '';
  for (const r of rows) {
    const tr = document.createElement('tr');
    const tdM = document.createElement('td');
    tdM.textContent = r.model;
    tdM.className = 'mono';
    const tdE = document.createElement('td');
    tdE.textContent = (r.elo ?? 0).toFixed(1);
    const tdW = document.createElement('td');
    tdW.textContent = String(r.wins ?? 0);
    const tdL = document.createElement('td');
    tdL.textContent = String(r.losses ?? 0);
    const tdD = document.createElement('td');
    tdD.textContent = String(r.draws ?? 0);
    tr.append(tdM, tdE, tdW, tdL, tdD);
    evalTableBody.appendChild(tr);
  }
}

// Tournament Game Viewer
const tournamentSelect = document.getElementById('tournamentSelect');
const gameSelect = document.getElementById('gameSelect');
const tournamentGamesTableBody = document.querySelector('#tournamentGamesTable tbody');
const gameViewerCanvas = document.getElementById('gameViewerCanvas');
const gameViewerSlider = document.getElementById('gameViewerSlider');
const gameViewerMeta = document.getElementById('gameViewerMeta');

let currentTournamentData = null;
let currentGameData = null;
let currentMoveIndex = 0;

async function refreshTournaments() {
  const data = await apiGet('/api/tournament/list');
  const tournaments = data.tournaments || [];
  tournamentSelect.innerHTML = '';
  if (tournaments.length === 0) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = 'No tournaments yet';
    tournamentSelect.appendChild(opt);
    return;
  }
  for (const t of tournaments) {
    const opt = document.createElement('option');
    opt.value = t.tournament_id;
    opt.textContent = `${t.tournament_id} (${t.total_games} games)`;
    tournamentSelect.appendChild(opt);
  }
  if (tournaments.length > 0) {
    await loadTournament(tournaments[0].tournament_id);
  }
}

async function loadTournament(tournamentId) {
  if (!tournamentId) return;
  try {
    const data = await apiGet(`/api/tournament/${tournamentId}`);
    currentTournamentData = data;
    renderTournamentGames(data.games || []);
    if (data.games && data.games.length > 0) {
      populateGameSelect(data.games);
      await loadGame(0);
    }
  } catch (e) {
    console.error('Failed to load tournament:', e);
    currentTournamentData = null;
  }
}

function populateGameSelect(games) {
  gameSelect.innerHTML = '';
  for (let i = 0; i < games.length; i++) {
    const g = games[i];
    const opt = document.createElement('option');
    opt.value = String(i);
    opt.textContent = `#${i}: ${g.model_a} vs ${g.model_b} (${g.winner_name})`;
    gameSelect.appendChild(opt);
  }
}

function renderTournamentGames(games) {
  tournamentGamesTableBody.innerHTML = '';
  for (const g of games) {
    const tr = document.createElement('tr');
    tr.style.cursor = 'pointer';
    tr.addEventListener('click', () => loadGame(g.game_index));

    const tdIdx = document.createElement('td');
    tdIdx.textContent = String(g.game_index);

    const tdMatch = document.createElement('td');
    tdMatch.textContent = g.match || `${g.model_a} vs ${g.model_b}`;
    tdMatch.style.fontSize = '11px';

    const tdBlack = document.createElement('td');
    tdBlack.textContent = g.a_is_black ? 'A' : 'B';

    const tdWinner = document.createElement('td');
    tdWinner.textContent = g.winner_name;
    tdWinner.style.fontWeight = 'bold';
    tdWinner.style.color = g.winner_name === 'A' ? '#2ca02c' : (g.winner_name === 'B' ? '#d62728' : '#666');

    const tdMoves = document.createElement('td');
    tdMoves.textContent = String(g.total_moves);

    tr.append(tdIdx, tdMatch, tdBlack, tdWinner, tdMoves);
    tournamentGamesTableBody.appendChild(tr);
  }
}

async function loadGame(gameIndex) {
  if (!currentTournamentData || !currentTournamentData.games) return;
  const games = currentTournamentData.games;
  if (gameIndex < 0 || gameIndex >= games.length) return;

  currentGameData = games[gameIndex];
  currentMoveIndex = currentGameData.moves.length - 1;
  gameSelect.value = String(gameIndex);

  gameViewerSlider.max = String(Math.max(0, currentGameData.moves.length - 1));
  gameViewerSlider.value = String(currentMoveIndex);

  renderGameAt(currentMoveIndex);
}

function renderGameAt(moveIndex) {
  if (!gameViewerCanvas || !currentGameData) return;

  const size = currentGameData.board_size || 15;
  const board = new Array(size * size).fill(0);
  const moves = currentGameData.moves || [];
  const clampedIndex = Math.max(-1, Math.min(moveIndex, moves.length - 1));

  for (let i = 0; i <= clampedIndex; i++) {
    const m = moves[i];
    if (m) board[m.row * size + m.col] = m.player;
  }

  const lastMove = clampedIndex >= 0 ? moves[clampedIndex] : null;
  drawBoard(gameViewerCanvas, { boardSize: size, board, lastMove });

  const moveInfo = clampedIndex >= 0 && moves[clampedIndex]
    ? `Move ${clampedIndex + 1}: ${moves[clampedIndex].agent} plays (${moves[clampedIndex].row}, ${moves[clampedIndex].col})`
    : 'Start position';

  gameViewerMeta.innerHTML = `
    <strong>${currentGameData.model_a}</strong> (A) vs <strong>${currentGameData.model_b}</strong> (B)<br>
    A is ${currentGameData.a_is_black ? 'Black ●' : 'White ○'} | Winner: <strong>${currentGameData.winner_name}</strong><br>
    ${moveInfo} | Total: ${moves.length} moves
  `;
}

document.getElementById('refreshTournaments').addEventListener('click', refreshTournaments);
tournamentSelect.addEventListener('change', () => loadTournament(tournamentSelect.value));
gameSelect.addEventListener('change', () => loadGame(Number(gameSelect.value)));
gameViewerSlider.addEventListener('input', () => {
  currentMoveIndex = Number(gameViewerSlider.value);
  renderGameAt(currentMoveIndex);
});
document.getElementById('gameViewerPrev').addEventListener('click', () => {
  if (currentMoveIndex > -1) {
    currentMoveIndex--;
    gameViewerSlider.value = String(Math.max(0, currentMoveIndex));
    renderGameAt(currentMoveIndex);
  }
});
document.getElementById('gameViewerNext').addEventListener('click', () => {
  if (currentGameData && currentMoveIndex < currentGameData.moves.length - 1) {
    currentMoveIndex++;
    gameViewerSlider.value = String(currentMoveIndex);
    renderGameAt(currentMoveIndex);
  }
});

// Telemetry
const lossCanvas = document.getElementById('loss');
const timeCanvas = document.getElementById('time');
const policyCanvas = document.getElementById('policyCanvas');
const policyMeta = document.getElementById('policyMeta');
const traceCanvas = document.getElementById('traceCanvas');
const traceSlider = document.getElementById('traceSlider');
const traceMeta = document.getElementById('traceMeta');

let latestTrace = null;

function drawBoard(canvas, cfg) {
  if (!canvas || !cfg) return;
  const size = cfg.boardSize || 15;
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0,0,w,h);
  ctx.fillStyle = '#fff';
  ctx.fillRect(0,0,w,h);

  const pad = 12;
  const cell = Math.min((w - 2*pad) / size, (h - 2*pad) / size);
  const offsetX = (w - cell * size) / 2;
  const offsetY = (h - cell * size) / 2;

  // Heatmap background
  if (cfg.heatmap && cfg.heatmap.length === size*size) {
    const maxVal = Math.max(...cfg.heatmap, 0.0001);
    for (let r = 0; r < size; r++) {
      for (let c = 0; c < size; c++) {
        const v = cfg.heatmap[r*size + c];
        if (v <= 0) continue;
        const alpha = Math.min(0.9, v / maxVal);
        ctx.fillStyle = `rgba(255,99,71,${alpha})`;
        ctx.fillRect(offsetX + c*cell, offsetY + r*cell, cell, cell);
      }
    }
  }

  ctx.strokeStyle = '#bbb';
  for (let i = 0; i <= size; i++) {
    const y = offsetY + i * cell;
    const x = offsetX + i * cell;
    ctx.beginPath();
    ctx.moveTo(offsetX, y);
    ctx.lineTo(offsetX + size*cell, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, offsetY);
    ctx.lineTo(x, offsetY + size*cell);
    ctx.stroke();
  }

  const board = cfg.board || [];
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      const v = board[r*size + c] || 0;
      if (!v) continue;
      ctx.beginPath();
      const cx = offsetX + (c + 0.5) * cell;
      const cy = offsetY + (r + 0.5) * cell;
      const radius = cell * 0.35;
      ctx.fillStyle = v === 1 ? '#222' : '#f4f4f4';
      ctx.strokeStyle = '#222';
      ctx.lineWidth = 1.5;
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
  }

  if (cfg.lastMove) {
    const { row, col } = cfg.lastMove;
    ctx.strokeStyle = '#ffd54f';
    ctx.lineWidth = 3;
    const x0 = offsetX + col * cell;
    const y0 = offsetY + row * cell;
    ctx.strokeRect(x0 + 2, y0 + 2, cell - 4, cell - 4);
  }
}

function renderPolicySnapshot(snap) {
  if (!snap || !policyCanvas) return;
  const size = snap.board_size || 15;
  const board = Array.isArray(snap.board) ? snap.board : [];
  const policy = Array.isArray(snap.policy) ? snap.policy : [];
  if (!policy.length) {
    policyMeta.textContent = 'No policy snapshot yet';
    return;
  }
  drawBoard(policyCanvas, { boardSize: size, board, heatmap: policy });
  let topIdx = 0;
  let topVal = -1;
  for (let i = 0; i < policy.length; i++) {
    if (policy[i] > topVal) {
      topVal = policy[i];
      topIdx = i;
    }
  }
  const row = Math.floor(topIdx / size);
  const col = topIdx % size;
  policyMeta.textContent = `Player ${snap.player_to_move || 1} → top move (${row}, ${col}) ${(topVal*100).toFixed(1)}%`;
}

function setTrace(trace) {
  latestTrace = trace || null;
  const moves = (trace && Array.isArray(trace.moves)) ? trace.moves : [];
  const maxStep = Math.max(0, moves.length - 1);
  if (traceSlider) {
    traceSlider.max = String(maxStep);
    traceSlider.value = String(maxStep);
  }
  renderTraceAt(Number(maxStep));
}

function renderTraceAt(step) {
  if (!traceCanvas) return;
  if (!latestTrace || !Array.isArray(latestTrace.moves) || latestTrace.moves.length === 0) {
    const ctx = traceCanvas.getContext('2d');
    ctx.clearRect(0,0,traceCanvas.width, traceCanvas.height);
    traceMeta.textContent = 'Waiting for telemetry...';
    return;
  }
  const size = latestTrace.board_size || 15;
  const board = new Array(size*size).fill(0);
  const clampedStep = Math.max(0, Math.min(step, latestTrace.moves.length - 1));
  for (let i = 0; i <= clampedStep; i++) {
    const m = latestTrace.moves[i];
    if (!m) continue;
    board[m.row * size + m.col] = m.player;
  }
  const lastMove = latestTrace.moves[clampedStep];
  drawBoard(traceCanvas, { boardSize: size, board, lastMove });
  const winner = latestTrace.winner === 0 ? 'draw' : (latestTrace.winner || 'unknown');
  traceMeta.textContent = `Move ${clampedStep + 1} / ${latestTrace.moves.length} • winner: ${winner}`;
}

function drawSeries(canvas, series, opts) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0,0,w,h);
  ctx.fillStyle = '#fff';
  ctx.fillRect(0,0,w,h);

  const pad = 30;
  const xs = series.map(p => p.x);
  const ysAll = series.flatMap(p => p.y.filter(v => Number.isFinite(v)));

  if (xs.length < 2 || ysAll.length === 0) {
    ctx.fillStyle = '#666';
    ctx.fillText('No data yet', pad, pad);
    return;
  }

  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  let yMin = Math.min(...ysAll);
  let yMax = Math.max(...ysAll);
  if (yMin === yMax) { yMin -= 1; yMax += 1; }

  ctx.strokeStyle = '#ddd';
  ctx.beginPath();
  ctx.moveTo(pad, pad);
  ctx.lineTo(pad, h - pad);
  ctx.lineTo(w - pad, h - pad);
  ctx.stroke();

  function sx(x) { return pad + (x - xMin) * (w - 2*pad) / (xMax - xMin); }
  function sy(y) { return (h - pad) - (y - yMin) * (h - 2*pad) / (yMax - yMin); }

  ctx.fillStyle = '#666';
  ctx.font = '12px system-ui';
  ctx.fillText(yMax.toFixed(2), 4, pad + 4);
  ctx.fillText(yMin.toFixed(2), 4, h - pad);

  const colors = opts.colors;
  const labels = opts.labels;
  for (let lineIdx = 0; lineIdx < labels.length; lineIdx++) {
    ctx.strokeStyle = colors[lineIdx];
    ctx.lineWidth = 2;
    ctx.beginPath();
    let started = false;
    for (const p of series) {
      const y = p.y[lineIdx];
      if (!Number.isFinite(y)) continue;
      const X = sx(p.x);
      const Y = sy(y);
      if (!started) { ctx.moveTo(X, Y); started = true; }
      else { ctx.lineTo(X, Y); }
    }
    ctx.stroke();
  }

  ctx.fillStyle = '#111';
  ctx.fillText(opts.title, pad, 18);
}

function parseFloatOrNaN(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : NaN;
}

async function refreshTelemetry() {
  const jobId = document.getElementById('teleJobSelect').value || selectedJobId;
  const jobQuery = jobId ? `&job_id=${encodeURIComponent(jobId)}` : '';
  const data = await apiGet(`/api/telemetry?limit=400${jobQuery}`);
  const points = data.points || [];
  if (points.length) {
    const last = points[points.length - 1];
    document.getElementById('teleStatus').textContent = last.status ?? '-';
    document.getElementById('teleEpisode').textContent = String(last.episode ?? '-');
    document.getElementById('teleReplay').textContent = `${last.replay_size ?? '-'} / ${last.min_replay ?? '-'}`;
    document.getElementById('teleUpdated').textContent = `Updated: ${new Date((last.ts||Date.now()/1000)*1000).toLocaleString()}`;
  }

  const lossSeries = points.map(p => ({ x: p.episode, y: [parseFloatOrNaN(p.policy_loss), parseFloatOrNaN(p.value_loss)] }));
  drawSeries(lossCanvas, lossSeries, { title: 'Loss', labels: ['policy_loss','value_loss'], colors: ['#1f77b4', '#ff7f0e'] });

  const timeSeries = points.map(p => ({ x: p.episode, y: [parseFloatOrNaN(p.episode_sec), parseFloatOrNaN(p.avg_ep_sec)] }));
  drawSeries(timeCanvas, timeSeries, { title: 'Time', labels: ['episode_sec','avg_ep_sec'], colors: ['#2ca02c', '#d62728'] });

  const lastSnap = [...points].reverse().find(p => p.policy_snapshot);
  if (lastSnap && lastSnap.policy_snapshot) {
    renderPolicySnapshot(lastSnap.policy_snapshot);
  }

  const lastTrace = [...points].reverse().find(p => p.self_play_trace);
  if (lastTrace && lastTrace.self_play_trace) {
    setTrace(lastTrace.self_play_trace);
  }
}

let teleTimer = null;
function scheduleTelemetry() {
  if (teleTimer) clearInterval(teleTimer);
  if (!document.getElementById('teleAuto').checked) return;
  const interval = Math.max(200, Number(document.getElementById('teleInterval').value) || 1000);
  teleTimer = setInterval(refreshTelemetry, interval);
}

document.getElementById('refreshModels').addEventListener('click', refreshModels);
document.getElementById('runEval').addEventListener('click', runEval);
document.getElementById('startTrain').addEventListener('click', startTrain);
document.getElementById('teleRefresh').addEventListener('click', refreshTelemetry);
document.getElementById('teleAuto').addEventListener('change', scheduleTelemetry);
document.getElementById('teleInterval').addEventListener('change', scheduleTelemetry);
document.getElementById('teleJobSelect').addEventListener('change', () => {
  selectedJobId = document.getElementById('teleJobSelect').value || selectedJobId;
  refreshTelemetry();
});
if (traceSlider) {
  traceSlider.addEventListener('input', () => renderTraceAt(Number(traceSlider.value)));
}

refreshModels();
refreshTournaments();
refreshJobs().then(() => refreshTelemetry().then(scheduleTelemetry));
setInterval(refreshJobs, 1000);
</script>
</body>
</html>
"""


@dataclass
class Job:
    id: str
    type: str  # 'train' | 'eval'
    status: str  # 'running' | 'completed' | 'failed' | 'stopped'
    started_at: float
    updated_at: float
    log_path: Path
    pid: Optional[int] = None
    result: Optional[Dict[str, Any]] = None


class TelemetryStore:
    def __init__(self, *, keep_points: int = 5000) -> None:
        self._lock = threading.Lock()
        self._keep_points = int(keep_points)
        self._points: Dict[str, Deque[Dict[str, Any]]] = {}

    def append(self, job_id: str, point: Dict[str, Any]) -> None:
        with self._lock:
            if job_id not in self._points:
                self._points[job_id] = deque(maxlen=self._keep_points)
            self._points[job_id].append(point)

    def get(self, job_id: str, limit: int = 400) -> List[Dict[str, Any]]:
        with self._lock:
            pts = list(self._points.get(job_id, deque()))
        return pts[-int(limit):]

    def list_job_ids(self) -> List[Dict[str, Any]]:
      with self._lock:
        items = []
        for jid, pts in self._points.items():
          last_ts = 0.0
          if pts:
            last = pts[-1]
            last_ts = float(last.get("ts", 0.0) or 0.0)
          items.append({"job_id": jid, "points": len(pts), "last_ts": last_ts})
      items.sort(key=lambda x: x.get("last_ts", 0.0), reverse=True)
      return items


class JobManager:
    def __init__(self, root: Path, *, dashboard_base_url: str, telemetry: TelemetryStore) -> None:
        self._root = root
        self._dashboard_base_url = str(dashboard_base_url).rstrip("/")
        self._telemetry = telemetry
        self._lock = threading.Lock()
        self._jobs: Dict[str, Job] = {}

    def _new_id(self, prefix: str) -> str:
        return f"{prefix}-{int(_now()*1000)}-{random.randint(1000, 9999)}"

    def list(self) -> List[Dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())
        return [
            {
                "id": j.id,
                "type": j.type,
                "status": j.status,
                "started_at": j.started_at,
                "updated_at": j.updated_at,
                "log_path": _safe_relpath(j.log_path),
                "pid": j.pid,
                "result": j.result,
            }
            for j in sorted(jobs, key=lambda x: x.started_at, reverse=True)
        ]

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def create_train_job(self, args: Dict[str, Any]) -> Job:
        job_id = self._new_id("train")
        logs_dir = (CHECKPOINT_DIR / "jobs").resolve()
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"{job_id}.log"

        cmd = [
            sys.executable,
            "-m",
            "python_ai.train",
            "--model-path",
            str(args.get("model_path", "./python_ai/checkpoints/policy_value.pt")),
            "--episodes",
            str(int(args.get("episodes", 200))),
            "--games-per-episode",
            str(int(args.get("games_per_episode", 1))),
            "--simulations",
            str(int(args.get("simulations", 64))),
            "--batch-size",
            str(int(args.get("batch_size", 128))),
            "--batches-per-episode",
            str(int(args.get("batches_per_episode", 8))),
            "--lr",
            str(float(args.get("lr", 1e-3))),
            "--replay-size",
            str(int(args.get("replay_size", 50000))),
            "--min-replay",
            str(int(args.get("min_replay", 5000))),
            "--temperature",
            str(float(args.get("temperature", 1.0))),
            "--temperature-decay",
            str(float(args.get("temperature_decay", 0.995))),
            "--min-temperature",
            str(float(args.get("min_temperature", 0.1))),
            "--channels",
            str(int(args.get("channels", 128))),
            "--blocks",
            str(int(args.get("blocks", 8))),
            "--save-every",
            str(int(args.get("save_every", 25))),
            "--value-loss-weight",
            str(float(args.get("value_loss_weight", 1.0))),
            "--c-puct",
            str(float(args.get("c_puct", 1.5))),
            "--dirichlet-alpha",
            str(float(args.get("dirichlet_alpha", 0.3))),
            "--dirichlet-frac",
            str(float(args.get("dirichlet_frac", 0.25))),
            "--torch-threads",
            str(int(args.get("torch_threads", 0))),
            "--dataloader-workers",
            str(int(args.get("dataloader_workers", 0))),
            "--dashboard-url",
            self._dashboard_base_url,
            "--dashboard-job-id",
            job_id,
        ]

        if bool(args.get("resume")):
            cmd.append("--resume")
        if bool(args.get("augment")):
            cmd.append("--augment")

        replay_path = str(args.get("replay_path") or "").strip()
        if replay_path:
            cmd.extend(["--replay-path", replay_path])

        coreml_path = str(args.get("coreml_path") or "").strip()
        if coreml_path:
            cmd.extend(["--coreml-path", coreml_path])

        job = Job(
            id=job_id,
            type="train",
            status="running",
            started_at=_now(),
            updated_at=_now(),
            log_path=log_path,
            pid=None,
        )

        with log_path.open("w", encoding="utf-8") as lf:
            lf.write("$ " + " ".join(cmd) + "\n\n")
            lf.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=lf,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

        job.pid = int(proc.pid)

        def watcher() -> None:
            rc = proc.wait()
            with self._lock:
                j = self._jobs.get(job_id)
                if not j:
                    return
                if j.status == "stopped":
                    j.updated_at = _now()
                    return
                j.status = "completed" if rc == 0 else "failed"
                j.updated_at = _now()

        threading.Thread(target=watcher, daemon=True).start()

        with self._lock:
            self._jobs[job_id] = job
        return job

    def stop_job(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
        if not job or job.type != "train" or not job.pid:
            return False

        try:
            os.kill(job.pid, 15)
        except Exception:
            return False

        with self._lock:
            job.status = "stopped"
            job.updated_at = _now()
        return True

    def create_eval_job(self, models: List[str], *, games: int, sims: int, c_puct: float) -> Job:
        job_id = self._new_id("eval")
        logs_dir = (CHECKPOINT_DIR / "jobs").resolve()
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"{job_id}.log"

        job = Job(
            id=job_id,
            type="eval",
            status="running",
            started_at=_now(),
            updated_at=_now(),
            log_path=log_path,
        )

        with self._lock:
            self._jobs[job_id] = job

        def runner() -> None:
            try:
                result = run_tournament(models=models, games=games, sims=sims, c_puct=c_puct, log_path=log_path)
                with self._lock:
                    j = self._jobs.get(job_id)
                    if j:
                        j.result = result
                        j.status = "completed"
                        j.updated_at = _now()
            except Exception as e:
                with log_path.open("a", encoding="utf-8") as lf:
                    lf.write(f"\nERROR: {type(e).__name__}: {e}\n")
                with self._lock:
                    j = self._jobs.get(job_id)
                    if j:
                        j.status = "failed"
                        j.updated_at = _now()

        threading.Thread(target=runner, daemon=True).start()
        return job


def _elo_expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def run_tournament(*, models: List[str], games: int, sims: int, c_puct: float, log_path: Path) -> Dict[str, Any]:
    """Pairwise round-robin with a simple Elo update. Saves all games for later review."""
    import torch

    from python_ai.eval import MCTSAgent, _play_game_with_record
    from python_ai.model import get_device

    device_cfg = get_device()
    device = device_cfg.device

    paths = [Path(m).resolve() for m in models]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(str(p))

    agents = {str(p): MCTSAgent(p, device=device, sims=int(sims), c_puct=float(c_puct)) for p in paths}

    rating: Dict[str, float] = {str(p): 1000.0 for p in paths}
    record: Dict[str, Dict[str, int]] = {str(p): {"wins": 0, "losses": 0, "draws": 0} for p in paths}

    k_factor = 24.0

    # Collect all games for later review
    all_games: List[Dict[str, Any]] = []
    tournament_id = log_path.stem  # Use job id as tournament id

    with log_path.open("a", encoding="utf-8") as lf:
        lf.write(f"Device: {device}\n")
        lf.write(f"Models: {len(paths)}  games_per_pair={int(games)} sims={int(sims)} c_puct={float(c_puct)}\n\n")
        lf.flush()

        game_index = 0
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                a_key = str(paths[i])
                b_key = str(paths[j])
                lf.write(f"Match: {Path(a_key).name} vs {Path(b_key).name}\n")
                lf.flush()

                a = agents[a_key]
                b = agents[b_key]

                for g in range(int(games)):
                    a_is_black = (g % 2 == 0)
                    winner, game_rec = _play_game_with_record(
                        a, b,
                        a_is_black=a_is_black,
                        model_a_name=Path(a_key).name,
                        model_b_name=Path(b_key).name,
                    )

                    # Store game record
                    game_data = game_rec.to_dict()
                    game_data["game_index"] = game_index
                    game_data["tournament_id"] = tournament_id
                    game_data["match"] = f"{Path(a_key).name} vs {Path(b_key).name}"
                    game_data["game_in_match"] = g
                    all_games.append(game_data)
                    game_index += 1

                    if winner == 1:
                        record[a_key]["wins"] += 1
                        record[b_key]["losses"] += 1
                        score_a, score_b = 1.0, 0.0
                    elif winner == -1:
                        record[a_key]["losses"] += 1
                        record[b_key]["wins"] += 1
                        score_a, score_b = 0.0, 1.0
                    else:
                        record[a_key]["draws"] += 1
                        record[b_key]["draws"] += 1
                        score_a, score_b = 0.5, 0.5

                    exp_a = _elo_expected(rating[a_key], rating[b_key])
                    exp_b = 1.0 - exp_a
                    rating[a_key] += k_factor * (score_a - exp_a)
                    rating[b_key] += k_factor * (score_b - exp_b)

                lf.write(
                    f"  done: {Path(a_key).name} Elo={rating[a_key]:.1f}  {Path(b_key).name} Elo={rating[b_key]:.1f}\n\n"
                )
                lf.flush()

    ranking = [
        {
            "model": _safe_relpath(Path(k)),
            "elo": float(v),
            "wins": int(record[k]["wins"]),
            "losses": int(record[k]["losses"]),
            "draws": int(record[k]["draws"]),
        }
        for k, v in rating.items()
    ]
    ranking.sort(key=lambda r: r["elo"], reverse=True)

    # Save all games to JSON file
    games_file = log_path.with_suffix(".games.json")
    games_data = {
        "tournament_id": tournament_id,
        "models": [_safe_relpath(Path(p)) for p in paths],
        "games_per_pair": games,
        "sims": sims,
        "c_puct": c_puct,
        "total_games": len(all_games),
        "ranking": ranking,
        "games": all_games,
    }
    with games_file.open("w", encoding="utf-8") as f:
        json.dump(games_data, f, indent=2)

    return {"ranking": ranking, "record": record, "games_file": str(games_file)}


def create_app(*, dashboard_host: str, dashboard_port: int) -> Flask:
    app = Flask(__name__)
    # If the server is bound to a wildcard address, use a loopback address for
    # local clients (training subprocesses) to POST telemetry to.
    post_host = str(dashboard_host)
    if post_host in {"0.0.0.0", "::"}:
        post_host = "127.0.0.1"
    base_url = f"http://{post_host}:{int(dashboard_port)}"
    telemetry_store = TelemetryStore(keep_points=5000)
    jobs = JobManager(ROOT, dashboard_base_url=base_url, telemetry=telemetry_store)

    @app.get("/")
    def index() -> Response:
        return Response(_DASHBOARD_HTML, mimetype="text/html")

    @app.get("/api/models")
    def api_models() -> Response:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        pts = sorted(CHECKPOINT_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        out = []
        for p in pts:
            try:
                info = inspect_checkpoint(p)
            except Exception as e:
                info = {"path": str(p), "error": f"{type(e).__name__}: {e}"}
            st = p.stat()
            out.append(
                {
                    "path": str(p.resolve()),
                    "relpath": _safe_relpath(p),
                    "mtime": float(st.st_mtime),
                    "size": int(st.st_size),
                    "inspect": info,
                }
            )
        return jsonify({"models": out})

    @app.post("/api/eval/tournament")
    def api_eval_tournament() -> Response:
        body = request.get_json(force=True, silent=True) or {}
        models = body.get("models") or []
        if not isinstance(models, list) or len(models) < 2:
            return jsonify({"error": "models must be a list of >=2"}), 400
        games = int(body.get("games", 20))
        sims = int(body.get("sims", 64))
        c_puct = float(body.get("c_puct", 1.5))

        job = jobs.create_eval_job([str(m) for m in models], games=games, sims=sims, c_puct=c_puct)
        return jsonify({"job_id": job.id})

    @app.get("/api/tournament/list")
    def api_tournament_list() -> Response:
        """List all saved tournament game files."""
        jobs_dir = (CHECKPOINT_DIR / "jobs").resolve()
        if not jobs_dir.exists():
            return jsonify({"tournaments": []})
        game_files = sorted(jobs_dir.glob("*.games.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        tournaments = []
        for gf in game_files:
            try:
                with gf.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                tournaments.append({
                    "file": str(gf.name),
                    "tournament_id": data.get("tournament_id", gf.stem.replace(".games", "")),
                    "models": data.get("models", []),
                    "total_games": data.get("total_games", 0),
                    "games_per_pair": data.get("games_per_pair", 0),
                    "sims": data.get("sims", 0),
                    "mtime": gf.stat().st_mtime,
                })
            except Exception:
                continue
        return jsonify({"tournaments": tournaments})

    @app.get("/api/tournament/<tournament_id>")
    def api_tournament_detail(tournament_id: str) -> Response:
        """Get full tournament data including all games."""
        jobs_dir = (CHECKPOINT_DIR / "jobs").resolve()
        # Try to find the games file
        game_file = jobs_dir / f"{tournament_id}.games.json"
        if not game_file.exists():
            # Try with eval- prefix
            game_file = jobs_dir / f"eval-{tournament_id}.games.json"
        if not game_file.exists():
            return jsonify({"error": "Tournament not found"}), 404
        try:
            with game_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": f"Failed to load: {e}"}), 500

    @app.get("/api/tournament/<tournament_id>/game/<int:game_index>")
    def api_tournament_game(tournament_id: str, game_index: int) -> Response:
        """Get a single game from a tournament."""
        jobs_dir = (CHECKPOINT_DIR / "jobs").resolve()
        game_file = jobs_dir / f"{tournament_id}.games.json"
        if not game_file.exists():
            game_file = jobs_dir / f"eval-{tournament_id}.games.json"
        if not game_file.exists():
            return jsonify({"error": "Tournament not found"}), 404
        try:
            with game_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            games = data.get("games", [])
            if game_index < 0 or game_index >= len(games):
                return jsonify({"error": "Game index out of range"}), 404
            return jsonify({"game": games[game_index], "total_games": len(games)})
        except Exception as e:
            return jsonify({"error": f"Failed to load: {e}"}), 500

    @app.post("/api/train/start")
    def api_train_start() -> Response:
        body = request.get_json(force=True, silent=True) or {}
        job = jobs.create_train_job(body)
        return jsonify({"job_id": job.id})

    @app.post("/api/telemetry/<job_id>")
    def api_telemetry_ingest(job_id: str) -> Response:
        point = request.get_json(force=True, silent=True) or {}
        if not isinstance(point, dict):
            return jsonify({"error": "point must be an object"}), 400
        point = dict(point)
        point["job_id"] = job_id
        telemetry_store.append(job_id, point)
        return jsonify({"ok": True})

    @app.get("/api/jobs")
    def api_jobs() -> Response:
        return jsonify({"jobs": jobs.list()})

    @app.get("/api/jobs/<job_id>/log")
    def api_job_log(job_id: str) -> Response:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "job not found"}), 404
        limit = int(request.args.get("limit", "400"))
        text = ""
        try:
            if job.log_path.exists():
                with job.log_path.open("r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()[-limit:]
                    text = "".join(lines)
        except Exception:
            text = ""
        return jsonify({"job": {
            "id": job.id,
            "type": job.type,
            "status": job.status,
            "started_at": job.started_at,
            "updated_at": job.updated_at,
            "result": job.result,
        }, "text": text})

    @app.post("/api/jobs/<job_id>/stop")
    def api_job_stop(job_id: str) -> Response:
        ok = jobs.stop_job(job_id)
        return jsonify({"ok": bool(ok)})

    @app.get("/api/telemetry")
    def api_telemetry() -> Response:
        limit = int(request.args.get("limit", "400"))
        job_id = request.args.get("job_id", "")

        if job_id:
            points = telemetry_store.get(job_id, limit=limit)
            return jsonify({"job_id": job_id, "points": points})

        return jsonify({"job_id": None, "points": []})

    @app.get("/api/telemetry/jobs")
    def api_telemetry_jobs() -> Response:
      jobs = telemetry_store.list_job_ids()
      return jsonify({"jobs": jobs})

    return app


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gomoku dashboard")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8787)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(dashboard_host=str(args.host), dashboard_port=int(args.port))
    app.run(host=str(args.host), port=int(args.port), debug=False, threaded=True)


if __name__ == "__main__":
    main()
