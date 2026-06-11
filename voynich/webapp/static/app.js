/* Voynich Decipherment Workbench — frontend. Vanilla JS, polling API. */
"use strict";

const $ = (sel) => document.querySelector(sel);

let openRunId = null;
let detailTimer = null;

/* ---------- tabs ---------- */
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tabpane").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    $("#tab-" + btn.dataset.tab).classList.add("active");
  });
});

/* ---------- data status ---------- */
async function refreshData() {
  const st = await fetch("/api/data/status").then((r) => r.json());
  const pill = $("#data-pill");
  if (st.ready) {
    pill.textContent = "data ready";
    pill.className = "pill ok";
  } else if (st.download && st.download.status === "running") {
    pill.textContent = "downloading data…";
    pill.className = "pill";
  } else {
    pill.textContent = "data missing — see Data tab";
    pill.className = "pill bad";
  }
  const list = $("#data-list");
  const items = [["Voynich interlinear transcription", st.voynich]];
  for (const [lang, ok] of Object.entries(st.references)) {
    items.push(["Reference corpus: " + lang, ok]);
  }
  list.innerHTML = items
    .map(([name, ok]) => `<li class="${ok ? "ok" : "missing"}">${name}</li>`)
    .join("");
  if (st.download && st.download.status === "error") {
    list.innerHTML += `<li class="missing">Download error: ${st.download.error}</li>`;
  }
}

$("#download-data").addEventListener("click", async () => {
  await fetch("/api/data/download", { method: "POST" });
  const poll = setInterval(async () => {
    await refreshData();
    const st = await fetch("/api/data/status").then((r) => r.json());
    if (st.download.status !== "running") clearInterval(poll);
  }, 1500);
});

/* ---------- starting runs ---------- */
function formConfig(form) {
  const cfg = {};
  for (const el of form.elements) {
    if (!el.name) continue;
    if (el.type === "checkbox") {
      cfg[el.name] = el.checked;
    } else {
      cfg[el.name] = el.value === "" ? null : el.value;
    }
  }
  return cfg;
}

async function startRun(kind, form) {
  const resp = await fetch("/api/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ kind, config: formConfig(form) }),
  });
  const body = await resp.json();
  if (!resp.ok) {
    alert(body.error || "failed to start run");
    return;
  }
  await refreshRuns();
  openDetail(body.id);
}

$("#start-solve").addEventListener("click", () => startRun("solve", $("#solve-form")));
$("#start-sweep").addEventListener("click", () => startRun("sweep", $("#solve-form")));
$("#start-bench").addEventListener("click", () => startRun("benchmark", $("#bench-form")));

/* ---------- run list ---------- */
function describeRun(r) {
  const c = r.config || {};
  if (r.kind === "solve") {
    return `solve · Currier ${c.currier_language || "?"} vs ${c.reference} · ${c.hypothesis} · ${c.iterations}×${c.restarts}`;
  }
  if (r.kind === "sweep") {
    const n = (c.references || []).length || "all";
    return `sweep · Currier ${c.currier_language || "?"} · ${c.hypothesis} · ${n} languages · ${c.iterations}×${c.restarts}`;
  }
  return `benchmark · ${c.reference} · ${c.mode || "substitution"} · ${c.cipher_chars} chars · ${c.iterations}×${c.restarts}`;
}

async function refreshRuns() {
  const runs = await fetch("/api/runs").then((r) => r.json());
  const list = $("#runs-list");
  if (!runs.length) {
    list.innerHTML = '<p class="hint">No runs yet.</p>';
    return;
  }
  list.innerHTML = runs
    .sort((a, b) => b.id - a.id)
    .map((r) => {
      const p = r.progress || {};
      const pct = p.total_iterations
        ? Math.round(
            (100 * ((p.restart || 0) * p.total_iterations + p.iteration)) /
              (p.total_iterations * (p.restarts || 1))
          )
        : 0;
      const prog = r.status === "running" ? ` · ${pct}%` : "";
      return `<div class="runrow" data-id="${r.id}">
        <div><b>#${r.id}</b> ${describeRun(r)}
          <div class="meta">${new Date(r.started).toLocaleTimeString()}${r.error ? " · " + r.error : ""}</div>
        </div>
        <span class="status ${r.status}">${r.status}${prog}</span>
      </div>`;
    })
    .join("");
  list.querySelectorAll(".runrow").forEach((row) => {
    row.addEventListener("click", () => openDetail(parseInt(row.dataset.id, 10)));
  });
}

/* ---------- run detail ---------- */
function openDetail(id) {
  openRunId = id;
  $("#run-detail").classList.remove("hidden");
  if (detailTimer) clearInterval(detailTimer);
  pollDetail();
  detailTimer = setInterval(pollDetail, 1000);
  $("#run-detail").scrollIntoView({ behavior: "smooth" });
}

$("#close-detail").addEventListener("click", () => {
  openRunId = null;
  $("#run-detail").classList.add("hidden");
  if (detailTimer) clearInterval(detailTimer);
});

$("#stop-run").addEventListener("click", () => {
  if (openRunId) fetch(`/api/runs/${openRunId}/stop`, { method: "POST" });
});

async function pollDetail() {
  if (openRunId === null) return;
  const r = await fetch(`/api/runs/${openRunId}`).then((x) => x.json());
  $("#detail-title").textContent = `Run #${r.id} — ${describeRun(r)} [${r.status}]`;
  $("#stop-run").classList.toggle("hidden", r.status !== "running");

  const p = r.progress || {};
  if (p.total_iterations) {
    let total = p.total_iterations * (p.restarts || 1);
    let done = (p.restart || 0) * p.total_iterations + p.iteration;
    let prefix = "";
    if (p.n_langs) {
      done += (p.lang_index || 0) * total;
      total *= p.n_langs;
      prefix = `language ${(p.lang_index || 0) + 1}/${p.n_langs} (${p.language}) · `;
    }
    const phase = p.phase === "polishing" ? " · polishing key" : "";
    $("#bar-fill").style.width = `${(100 * done) / total}%`;
    $("#progress-text").textContent =
      prefix +
      `restart ${(p.restart || 0) + 1}/${p.restarts} · iteration ${p.iteration.toLocaleString()}/${p.total_iterations.toLocaleString()}` +
      ` · T=${(p.temperature || 0).toExponential(2)} · best score ${(p.best_score || 0).toFixed(4)}${phase}`;
  }
  drawChart(r.history || []);
  renderResult(r);
  if (r.status !== "running" && detailTimer) {
    clearInterval(detailTimer);
    detailTimer = null;
    refreshRuns();
  }
}

/* ---------- chart ---------- */
function drawChart(hist) {
  const cv = $("#chart");
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (hist.length < 2) return;
  const xs = hist.map((h) => h[0]);
  const ys = hist.map((h) => h[1]);
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const ymin = Math.min(...ys), ymax = Math.max(...ys);
  const pad = 44, W = cv.width, H = cv.height;
  const X = (x) => pad + ((x - xmin) / (xmax - xmin || 1)) * (W - pad - 12);
  const Y = (y) => H - 26 - ((y - ymin) / (ymax - ymin || 1)) * (H - 26 - 12);

  ctx.strokeStyle = "#3a342a";
  ctx.fillStyle = "#9b927f";
  ctx.font = "11px Georgia";
  for (let i = 0; i <= 4; i++) {
    const yv = ymin + ((ymax - ymin) * i) / 4;
    const ypix = Y(yv);
    ctx.beginPath(); ctx.moveTo(pad, ypix); ctx.lineTo(W - 12, ypix); ctx.stroke();
    ctx.fillText(yv.toFixed(3), 4, ypix + 4);
  }
  ctx.fillText("best score (bits/char) vs iteration", pad, H - 8);

  ctx.strokeStyle = "#c9a227";
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  hist.forEach((h, i) => {
    if (i === 0) ctx.moveTo(X(h[0]), Y(h[1]));
    else ctx.lineTo(X(h[0]), Y(h[1]));
  });
  ctx.stroke();
  ctx.lineWidth = 1;
}

/* ---------- result rendering ---------- */
function scoreBoxes(items) {
  return `<div class="scores">${items
    .map(([l, v]) => `<div class="scorebox"><div class="v">${v}</div><div class="l">${l}</div></div>`)
    .join("")}</div>`;
}

function renderResult(r) {
  const el = $("#detail-body");
  if (r.status === "error") {
    el.innerHTML = `<div class="verdict">Run failed: ${r.error}</div>`;
    return;
  }
  const res = r.result;
  if (!res) { el.innerHTML = ""; return; }

  if (r.kind === "sweep") {
    const best = res.table[0] || {};
    el.innerHTML =
      `<div class="verdict">${escapeHtml(res.note)}</div>` +
      `<h3>Ranked results (${res.table.length} languages)</h3>
       <table><tr><th>#</th><th>reference</th><th>family</th><th>gap closed</th>
       <th>held-out</th><th>floor</th><th>ceiling</th><th>decoded sample</th></tr>${res.table
         .map(
           (row, i) =>
             `<tr><td class="num">${i + 1}</td><td>${escapeHtml(row.label)}</td>
              <td>${escapeHtml(row.family)}</td>
              <td class="num"><b>${(row.gap_closed * 100).toFixed(1)}%</b></td>
              <td class="num">${row.test_heldout.toFixed(3)}</td>
              <td class="num">${row.random_key_floor.toFixed(3)}</td>
              <td class="num">${row.reference_ceiling.toFixed(3)}</td>
              <td class="mono">${escapeHtml(row.sample)}</td></tr>`
         )
         .join("")}</table>` +
      (res.saved_to ? `<p class="hint">Saved to ${escapeHtml(res.saved_to)}</p>` : "");
    return;
  }

  if (r.kind === "benchmark") {
    el.innerHTML =
      scoreBoxes([
        ["letters recovered", (res.accuracy * 100).toFixed(1) + "%"],
        ["solver score", res.best_score.toFixed(4)],
        ["true-key score", res.true_key_score.toFixed(4)],
        ["random-key floor", res.random_key_score_mean.toFixed(4)],
        ["elapsed", res.elapsed_sec.toFixed(1) + "s"],
      ]) +
      `<div class="verdict">${
        res.accuracy > 0.95
          ? "The annealer recovered the cipher — the machinery works on solvable problems."
          : res.accuracy > 0.7
          ? "Partial recovery; raise iterations/restarts or cipher length."
          : "Recovery failed — increase iterations, restarts or cipher length."
      }</div>` +
      `<h3>Decoded vs truth (first 300 chars)</h3>
       <div class="sampletext mono">${escapeHtml(res.decoded_preview)}</div>
       <details><summary>plaintext</summary>
       <div class="sampletext mono">${escapeHtml(res.plaintext_preview)}</div></details>`;
    return;
  }

  const s = res.scores;
  el.innerHTML =
    scoreBoxes([
      ["held-out score", s.test_heldout.toFixed(4)],
      ["train best", s.train_best.toFixed(4)],
      ["random-key floor", s.random_key_floor.toFixed(4)],
      ["language ceiling", s.reference_ceiling.toFixed(4)],
      ["gap closed", (s.gap_closed * 100).toFixed(1) + "%"],
    ]) +
    `<div class="verdict">${escapeHtml(res.verdict)}</div>` +
    `<h3>Decoded held-out sample</h3>
     <table><tr><th>folio.line</th><th>decoded</th></tr>${(res.decoded_sample || [])
       .map((d) => `<tr><td class="mono">${escapeHtml(d.ref)}</td><td class="mono">${escapeHtml(d.text)}</td></tr>`)
       .join("")}</table>` +
    `<details><summary>Best key (${(res.key || []).length} tokens)</summary>
     <div class="keygrid">${(res.key || [])
       .map((k) => {
         const states = Object.entries(k).filter(([n]) => n !== "token");
         const letters = states.map(([n, v]) => (states.length > 1 ? `${n.slice(0, 1)}:${v}` : v)).join(" ");
         return `<div class="keycell">${escapeHtml(k.token)} &rarr; <b>${escapeHtml(letters)}</b></div>`;
       })
       .join("")}</div></details>` +
    (res.saved_to ? `<p class="hint">Saved to ${escapeHtml(res.saved_to)}</p>` : "");
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

/* ---------- boot ---------- */
refreshData();
refreshRuns();
setInterval(refreshRuns, 3000);
setInterval(refreshData, 10000);
