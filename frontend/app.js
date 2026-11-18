// Interface de l'agent d'audit : dialogue avec l'API FastAPI (src/api/main.py).
//   POST /audit             dépose un contrat, renvoie un job_id
//   GET  /audit/{id}        statut + résultat détaillé
//   GET  /audit/{id}/rapport page HTML du rapport
//   GET  /audits            historique
//   POST /feedback          correction d'un juriste
// Aucune bibliothèque externe : JavaScript standard.

const API = "";                      // même serveur que l'interface
const POLL_MS = 1500;

const $ = (id) => document.getElementById(id);
const LIBELLE = { presente: "Présente", manquante: "Manquante", a_verifier: "À vérifier" };
const BADGE = { presente: "bg-green-lt", manquante: "bg-red-lt", a_verifier: "bg-yellow-lt", ok: "bg-green-lt" };
const ICON = (name, cls = "icon") => `<svg class="${cls}"><use href="vendor/icons.svg#i-${name}"/></svg>`;
const GRAVITE = { high: "élevée", medium: "moyenne", low: "faible" };

let selectedFile = null;
let currentJob = null;
let currentResult = null;
let pollTimer = null;
let feedbackTarget = null;

// ---------- Outils ----------

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function cut(s, n) {
  s = String(s ?? "");
  return s.length <= n ? s : s.slice(0, n).replace(/\s+\S*$/, "") + " …";
}

function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (t.hidden = true), 3000);
}

function show(view) {
  for (const v of ["welcome", "progress", "error", "result"]) $(`view-${v}`).hidden = v !== view;
  $("btn-report").hidden = view !== "result";
  $("nav-new").classList.toggle("active", view === "welcome");
  if (view === "welcome") $("page-title").textContent = "Nouvel audit";
}

function badge(statut, texte) {
  return `<span class="badge ${BADGE[statut] || "bg-secondary-lt"}">${texte ?? LIBELLE[statut]}</span>`;
}

function confBar(c) {
  if (c == null) return "";
  const pct = Math.round(c * 100);
  return `<span class="conf"><span class="progress progress-sm"><span class="progress-bar" style="width:${pct}%"></span></span>${pct} %</span>`;
}

async function api(path, options) {
  const res = await fetch(API + path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) { /* réponse non JSON */ }
    throw new Error(detail);
  }
  return res.json();
}

// ---------- État du service ----------

async function checkHealth() {
  const el = $("api-status");
  try {
    await api("/health");
    el.className = "api-status up";
    el.querySelector(".label").textContent = "Service en ligne";
  } catch (_) {
    el.className = "api-status down";
    el.querySelector(".label").textContent = "Service injoignable";
  }
}

// ---------- Choix du fichier ----------

function pickFile(file) {
  if (!file) return;
  if (!/\.(pdf|docx|txt)$/i.test(file.name)) {
    toast("Formats acceptés : PDF, Word (.docx) ou texte (.txt)");
    return;
  }
  selectedFile = file;
  $("file-name").textContent = `${file.name} (${(file.size / 1024).toFixed(0)} Ko)`;
  $("file-picked").hidden = false;
  $("btn-audit").disabled = false;
}

function initDropzone() {
  const dz = $("dropzone");
  $("file-input").addEventListener("change", (e) => pickFile(e.target.files[0]));
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("over"); }));
  dz.addEventListener("drop", (e) => pickFile(e.dataTransfer.files[0]));
}

// ---------- Lancer un audit ----------

async function startAudit() {
  if (!selectedFile) return;
  $("btn-audit").disabled = true;
  setStep(0, `Envoi de ${selectedFile.name}…`);
  show("progress");
  try {
    const form = new FormData();
    form.append("file", selectedFile);
    const job = await api("/audit", { method: "POST", body: form });
    currentJob = job.job_id;
    loadHistory();
    poll();
  } catch (err) {
    showError(err.message);
  } finally {
    $("btn-audit").disabled = false;
  }
}

function setStep(step, msg) {
  document.querySelectorAll("#stepper li").forEach((li) => {
    li.classList.toggle("active", Number(li.dataset.step) === Math.min(step, 3));
  });
  if (msg) $("progress-msg").textContent = msg;
}

async function poll() {
  clearTimeout(pollTimer);
  try {
    const job = await api(`/audit/${currentJob}`);
    $("progress-title").textContent = `Analyse de ${job.document}`;
    $("page-title").textContent = job.document;
    if (job.status === "en_attente") setStep(1, "Lecture du document…");
    else if (job.status === "en_cours") setStep(2, "Classification des clauses par Legal-BERT… (plus long au premier audit : chargement du modèle)");
    else if (job.status === "termine") { setStep(4); renderResult(job); loadHistory(); return; }
    else if (job.status === "erreur") { showError(job.erreur || "Erreur inconnue"); loadHistory(); return; }
    pollTimer = setTimeout(poll, POLL_MS);
  } catch (err) {
    showError(err.message);
  }
}

function showError(msg) {
  $("error-msg").textContent = msg;
  show("error");
}

// ---------- Affichage du résultat ----------

function renderResult(job) {
  currentResult = job.resultat;
  const r = job.resultat;
  const att = r.attendues;
  const count = (s) => att.filter((a) => a.statut === s).length;

  $("page-title").textContent = job.document;
  const date = new Date(job.created_at).toLocaleString("fr-FR");
  $("res-meta").textContent = `Audité le ${date}` + (job.duree_s != null ? ` · ${job.duree_s} s` : "") +
    ` · ${r.stats.segments} paragraphes analysés · ${r.stats.categories} types de clauses`;
  $("btn-report").href = `${API}/audit/${job.job_id}/rapport`;

  $("kpi-clauses").textContent = r.stats.clauses;
  $("kpi-missing").textContent = count("manquante");
  $("kpi-check").textContent = count("a_verifier");
  $("kpi-rgpd").textContent = r.stats.rgpd;

  renderAttendues("all");
  renderClauses("");
  renderIdentite();
  $("report-frame").src = "about:blank";
  selectTab("attendues");
  show("result");
}

function renderAttendues(filter) {
  document.querySelectorAll("#filters .btn").forEach((c) => c.classList.toggle("active", c.dataset.filter === filter));
  const items = currentResult.attendues.filter((a) => filter === "all" || a.statut === filter);
  $("list-attendues").innerHTML = items.length ? items.map((a, i) => {
    const p = a.preuve;
    const preuve = p
      ? `<div class="excerpt">${esc(cut(p.texte, 320))}</div>`
      : `<div class="text-secondary mt-2">${a.statut === "manquante"
          ? "Aucun passage trouvé dans le contrat."
          : "Non trouvée automatiquement : à relire par un humain."}</div>`;
    return `<div class="list-group-item py-3">
      <div class="d-flex justify-content-between align-items-start gap-3">
        <div>
          <div class="fw-bold">${esc(a.titre)}</div>
          <div class="text-secondary small">${esc(a.categorie)} · gravité ${esc(GRAVITE[a.gravite] || a.gravite)}</div>
        </div>
        ${badge(a.statut)}
      </div>
      <div class="text-secondary mt-2 small">${esc(a.regle)}</div>
      ${preuve}
      <div class="d-flex justify-content-between align-items-center mt-2 small text-secondary">
        <span>${p ? `Trouvée par : ${esc(p.source)} ${confBar(p.confiance)}` : ""}</span>
        <button class="btn btn-sm btn-ghost-primary" data-fb="${i}">${ICON("edit")} Corriger</button>
      </div>
    </div>`;
  }).join("") : `<div class="list-group-item text-secondary">Aucune clause dans cette catégorie.</div>`;

  $("list-attendues").querySelectorAll("[data-fb]").forEach((b) =>
    b.addEventListener("click", () => openFeedback(items[Number(b.dataset.fb)])));
}

function renderClauses(query) {
  const q = query.trim().toLowerCase();
  const items = currentResult.clauses.filter((c) => !q || c.texte.toLowerCase().includes(q) || c.categorie.toLowerCase().includes(q));
  $("list-clauses").innerHTML = items.length ? items.map((c) => `
    <div class="list-group-item py-3">
      <div class="d-flex justify-content-between align-items-center gap-3">
        <div class="fw-bold"><span class="badge bg-blue-lt me-2">§${c.n}</span>${esc(c.categorie)}</div>
        ${confBar(c.confiance)}
      </div>
      <div class="excerpt">${esc(cut(c.texte, 400))}</div>
      ${c.regle ? `<div class="small text-secondary mt-2"><b>Règle ${esc(c.regle.id)} — ${esc(c.regle.titre)}</b> : ${esc(cut(c.regle.regle, 260))}</div>` : ""}
      ${c.rgpd ? `<div class="small mt-2"><span class="badge bg-purple-lt me-1">${ICON("shield-lock", "icon icon-sm")} Données personnelles</span>
        Contrôle ${esc(c.rgpd.id)} · ${esc(c.rgpd.theme)} — ${esc(c.rgpd.refs.join(", "))}. <b>À vérifier par un humain.</b></div>` : ""}
    </div>`).join("") : `<div class="list-group-item text-secondary">Aucune clause ne correspond.</div>`;
}

function renderIdentite() {
  $("list-identite").innerHTML = currentResult.identite.map((i) => `
    <div class="list-group-item py-3">
      <div class="d-flex justify-content-between align-items-center">
        <div class="fw-bold">${esc(i.categorie)}</div>
        ${i.trouve ? badge("ok", "Trouvé") : badge("a_verifier")}
      </div>
      ${i.trouve ? `<div class="excerpt">${esc(cut(i.texte, 300))}</div>` : `<div class="text-secondary mt-2">Non trouvé automatiquement.</div>`}
    </div>`).join("");
}

function selectTab(name) {
  document.querySelectorAll("#tabs .nav-link").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".tab-panel").forEach((p) => (p.hidden = p.id !== `tab-${name}`));
  if (name === "rapport" && currentJob) {
    const src = `${API}/audit/${currentJob}/rapport`;
    if (!$("report-frame").src.endsWith(src)) $("report-frame").src = src;
  }
}

// ---------- Historique ----------

async function loadHistory() {
  try {
    const jobs = await api("/audits");
    const box = $("history");
    if (!jobs.length) { box.innerHTML = `<div class="history-empty">Aucun audit pour l'instant.</div>`; return; }
    const etat = { termine: ["ok", "Terminé"], erreur: ["manquante", "Erreur"], en_cours: ["a_verifier", "En cours"], en_attente: ["a_verifier", "En attente"] };
    box.innerHTML = jobs.map((j) => `
      <div class="history-item ${j.job_id === currentJob ? "current" : ""}" data-id="${j.job_id}">
        <div style="min-width:0">
          <div class="history-name">${esc(j.document)}</div>
          <div class="history-date">${new Date(j.created_at).toLocaleString("fr-FR")}</div>
        </div>
        ${badge(...(etat[j.status] || ["a_verifier", j.status]))}
      </div>`).join("");
    box.querySelectorAll("[data-id]").forEach((el) => el.addEventListener("click", () => {
      currentJob = el.dataset.id;
      show("progress");
      poll();
      loadHistory();
    }));
  } catch (_) { /* service injoignable : déjà signalé dans le menu */ }
}

// ---------- Mode sombre ----------

function applyTheme(theme) {
  document.documentElement.setAttribute("data-bs-theme", theme);
  document.body.setAttribute("data-bs-theme", theme);
  $("btn-theme").innerHTML = theme === "dark" ? `${ICON("sun")} <span>Mode clair</span>` : `${ICON("moon")} <span>Mode sombre</span>`;
  try { localStorage.setItem("audit-theme", theme); } catch (_) { /* stockage indisponible */ }
}

function initTheme() {
  let theme = "light";
  try { theme = localStorage.getItem("audit-theme") || "light"; } catch (_) { /* stockage indisponible */ }
  applyTheme(theme);
  $("btn-theme").addEventListener("click", () =>
    applyTheme(document.documentElement.getAttribute("data-bs-theme") === "dark" ? "light" : "dark"));
}

// ---------- Correction d'un juriste ----------

function openFeedback(item) {
  feedbackTarget = item;
  $("fb-clause").textContent = `${item.titre} — résultat actuel : ${LIBELLE[item.statut]}`;
  $("fb-verdict").value = item.statut;
  $("fb-comment").value = "";
  $("feedback-dialog").showModal();
}

async function sendFeedback() {
  const body = {
    job_id: currentJob,
    clause_ref: feedbackTarget.categorie,
    verdict_corrige: $("fb-verdict").value,
    commentaire: $("fb-comment").value || null,
  };
  try {
    await api("/feedback", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    toast("Correction enregistrée : elle servira au réentraînement.");
  } catch (err) {
    toast(`Échec de l'enregistrement : ${err.message}`);
  }
}

// ---------- Démarrage ----------

document.addEventListener("DOMContentLoaded", () => {
  initDropzone();
  $("btn-audit").addEventListener("click", startAudit);
  $("btn-refresh").addEventListener("click", loadHistory);
  document.querySelectorAll("#tabs .nav-link").forEach((t) => t.addEventListener("click", (e) => { e.preventDefault(); selectTab(t.dataset.tab); }));
  document.querySelectorAll("#filters .btn").forEach((c) => c.addEventListener("click", () => renderAttendues(c.dataset.filter)));
  $("nav-new").addEventListener("click", (e) => { e.preventDefault(); clearTimeout(pollTimer); show("welcome"); });
  initTheme();
  $("search-clauses").addEventListener("input", (e) => renderClauses(e.target.value));
  $("feedback-form").addEventListener("submit", (e) => {
    if (e.submitter && e.submitter.value === "send") sendFeedback();
  });
  checkHealth();
  loadHistory();
  setInterval(checkHealth, 15000);
});
