const state = { sessionId: null, profiles: [] };

// ---------- helpers ----------
async function api(path, options = {}) {
  const resp = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!resp.ok) {
    const body = await resp.text();
    throw new Error(`${resp.status}: ${body}`);
  }
  return resp.status === 204 ? null : resp.json();
}

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "text") node.textContent = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const child of children) node.appendChild(child);
  return node;
}

function showModal(content) {
  document.getElementById("modal-content").textContent =
    typeof content === "string" ? content : JSON.stringify(content, null, 2);
  document.getElementById("modal").classList.remove("hidden");
}
document.getElementById("modal-close").onclick = () => document.getElementById("modal").classList.add("hidden");

// ---------- navigation ----------
document.querySelectorAll(".nav-item").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`view-${btn.dataset.view}`).classList.add("active");
    if (btn.dataset.view === "agents") loadAgents();
    if (btn.dataset.view === "tools") loadTools();
    if (btn.dataset.view === "mcp") loadMcp();
    if (btn.dataset.view === "memory") loadMemory();
    if (btn.dataset.view === "settings") loadModels();
    if (btn.dataset.view === "github" && !document.getElementById("github-results").dataset.loaded) {
      searchGithub("");
    }
  });
});

// ---------- profiles dropdown ----------
async function loadProfileSelect() {
  const profiles = await api("/api/agents");
  state.profiles = profiles;
  const select = document.getElementById("profile-select");
  select.innerHTML = "";
  select.appendChild(el("option", { value: "", text: "default" }));
  for (const p of profiles) select.appendChild(el("option", { value: p.name, text: p.name }));
}

// ---------- chat ----------
const messagesEl = document.getElementById("messages");
const statusLogEl = document.getElementById("status-log");

function addMessage(role, text) {
  messagesEl.appendChild(el("div", { class: `msg ${role}`, text }));
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function renderStatusTimeline(result) {
  statusLogEl.innerHTML = "";
  const lines = ["Planning..."];
  for (const step of result.plan || []) lines.push(`  • ${step}`);
  for (const tc of result.tool_calls || []) {
    lines.push(`Using tool: ${tc.name}`);
    lines.push(tc.ok ? "Received result." : `Tool error: ${tc.error}`);
  }
  lines.push("Generating answer...");
  for (const line of lines) statusLogEl.appendChild(el("div", { text: line }));
}

document.getElementById("chat-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = document.getElementById("chat-input");
  const message = input.value.trim();
  if (!message) return;
  input.value = "";
  addMessage("user", message);
  statusLogEl.innerHTML = "";
  statusLogEl.appendChild(el("div", { text: "Planning..." }));

  const profile = document.getElementById("profile-select").value || undefined;
  try {
    const result = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({ message, session_id: state.sessionId, profile }),
    });
    state.sessionId = result.session_id;
    renderStatusTimeline(result);
    addMessage("assistant", result.final_answer || `[status: ${result.status}]`);
  } catch (err) {
    addMessage("assistant", `Error: ${err.message}`);
  }
});

// ---------- agents ----------
async function loadAgents() {
  const profiles = await api("/api/agents");
  const list = document.getElementById("agents-list");
  list.innerHTML = "";
  for (const p of profiles) {
    list.appendChild(
      el("div", { class: "card" }, [
        el("h4", { text: p.name }),
        el("p", { text: p.system_prompt }),
        el("div", { class: "badges" }, [
          el("span", { class: "badge", text: `model: ${p.model || "default"}` }),
          el("span", { class: "badge", text: `temp: ${p.temperature}` }),
          el("span", { class: "badge", text: `steps: ${p.max_steps || "default"}` }),
        ]),
        el("div", { class: "card-actions" }, [
          el("button", { text: "Delete", onclick: async () => { await api(`/api/agents/${encodeURIComponent(p.name)}`, { method: "DELETE" }); loadAgents(); loadProfileSelect(); } }),
        ]),
      ])
    );
  }
  loadProfileSelect();
}

document.getElementById("agent-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const payload = {
    name: fd.get("name"),
    system_prompt: fd.get("system_prompt") || "You are a helpful, precise AI agent.",
    model: fd.get("model") || null,
    temperature: parseFloat(fd.get("temperature") || "0.2"),
    tools: (fd.get("tools") || "").split(",").map((s) => s.trim()).filter(Boolean),
    memory_enabled: fd.get("memory_enabled") === "on",
    max_steps: fd.get("max_steps") ? parseInt(fd.get("max_steps"), 10) : null,
  };
  await api("/api/agents", { method: "POST", body: JSON.stringify(payload) });
  e.target.reset();
  loadAgents();
});

// ---------- tools ----------
async function loadTools() {
  const tools = await api("/api/tools");
  const list = document.getElementById("tools-list");
  list.innerHTML = "";
  for (const t of tools) {
    list.appendChild(
      el("div", { class: "card" }, [
        el("h4", { text: t.name }),
        el("p", { text: t.description }),
        el("div", { class: "badges" }, [
          t.requires_confirmation ? el("span", { class: "badge warn", text: "requires confirmation" }) : el("span", { class: "badge ok", text: "safe" }),
        ]),
      ])
    );
  }
}

// ---------- github explorer ----------
function capabilityBadge(label, ok) {
  return el("span", { class: `badge ${ok ? "ok" : ""}`, text: `${label}: ${ok ? "yes" : "no"}` });
}

async function searchGithub(query) {
  const results = query
    ? await api(`/api/github/search?q=${encodeURIComponent(query)}`)
    : await api("/api/github/search");
  const container = document.getElementById("github-results");
  container.dataset.loaded = "1";
  container.innerHTML = "";
  for (const r of results) {
    const licenseBadge = el("span", {
      class: `badge ${r.license_category === "permissive" ? "ok" : r.license_category === "copyleft" ? "warn" : "danger"}`,
      text: `license: ${r.license}`,
    });
    container.appendChild(
      el("div", { class: "card" }, [
        el("h4", { text: r.name }),
        el("p", { text: `⭐ ${r.stars} · ${r.language || "?"} · updated ${(r.last_update || "").slice(0, 10)}` }),
        el("p", { text: r.description || "" }),
        el("div", { class: "badges" }, [
          licenseBadge,
          capabilityBadge("API", r.api_available),
          capabilityBadge("SDK", r.sdk_available),
          capabilityBadge("MCP", r.mcp_available),
        ]),
        el("div", { class: "card-actions" }, [
          el("a", { href: r.repository, target: "_blank", text: "Open on GitHub" }),
          el("button", { text: "Analyze", onclick: () => analyzeRepo(r.repository) }),
          el("button", { text: "View Integration Plan", onclick: () => analyzeRepo(r.repository) }),
          el("button", { text: "Install Adapter", onclick: () => installAdapter(r.repository) }),
        ]),
      ])
    );
  }
}

function fullNameFromUrl(url) {
  return url.replace("https://github.com/", "");
}

async function analyzeRepo(url) {
  try {
    const plan = await api("/api/github/analyze", {
      method: "POST",
      body: JSON.stringify({ full_name: fullNameFromUrl(url) }),
    });
    showModal(plan);
  } catch (err) {
    showModal(`Analysis failed: ${err.message}`);
  }
}

async function installAdapter(url) {
  try {
    const plan = await api("/api/github/analyze", {
      method: "POST",
      body: JSON.stringify({ full_name: fullNameFromUrl(url) }),
    });
    const note = plan.license_gate.auto_integrable
      ? "License is permissive. No code is executed automatically — copy the adapter stub below into app/integrations/ and wire it up manually."
      : "License requires MANUAL confirmation before any integration. Nothing was installed.";
    showModal(`${note}\n\n${JSON.stringify(plan, null, 2)}`);
  } catch (err) {
    showModal(`Could not build integration plan: ${err.message}`);
  }
}

document.getElementById("github-search-form").addEventListener("submit", (e) => {
  e.preventDefault();
  searchGithub(document.getElementById("github-query").value.trim());
});

// ---------- mcp ----------
async function loadMcp() {
  const servers = await api("/api/mcp");
  const list = document.getElementById("mcp-list");
  list.innerHTML = "";
  for (const s of servers) {
    list.appendChild(
      el("div", { class: "card" }, [
        el("h4", { text: s.name }),
        el("div", { class: "badges" }, [
          el("span", { class: `badge ${s.enabled ? "ok" : "danger"}`, text: s.enabled ? "enabled" : "disabled" }),
          el("span", { class: "badge", text: `transport: ${s.transport}` }),
          el("span", { class: "badge", text: `tools: ${s.tools_discovered}` }),
        ]),
        el("div", { class: "card-actions" }, [
          el("button", { text: "Connect", onclick: async () => { try { const r = await api(`/api/mcp/connect?name=${encodeURIComponent(s.name)}`, { method: "POST" }); showModal(r); loadMcp(); } catch (err) { showModal(err.message); } } }),
          el("button", { text: s.enabled ? "Disable" : "Enable", onclick: async () => { await api(`/api/mcp/${encodeURIComponent(s.name)}/toggle`, { method: "POST", body: JSON.stringify({ enabled: !s.enabled }) }); loadMcp(); } }),
        ]),
      ])
    );
  }
}

document.getElementById("mcp-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const payload = {
    name: fd.get("name"),
    transport: fd.get("transport"),
    command: fd.get("command") || null,
    args: (fd.get("args") || "").split(",").map((s) => s.trim()).filter(Boolean),
    url: fd.get("url") || null,
    timeout: parseFloat(fd.get("timeout") || "30"),
    enabled: true,
  };
  await api("/api/mcp/servers", { method: "POST", body: JSON.stringify(payload) });
  e.target.reset();
  loadMcp();
});

// ---------- memory ----------
async function loadMemory() {
  const memories = await api("/api/memory");
  const list = document.getElementById("memory-list");
  list.innerHTML = "";
  for (const m of memories) {
    list.appendChild(
      el("div", { class: "card" }, [
        el("h4", { text: m.agent_profile }),
        el("p", { text: m.text }),
        el("div", { class: "badges" }, (m.tags ? m.tags.split(",") : []).filter(Boolean).map((t) => el("span", { class: "badge", text: t }))),
        el("div", { class: "card-actions" }, [
          el("button", { text: "Forget", onclick: async () => { await api(`/api/memory/${m.id}`, { method: "DELETE" }); loadMemory(); } }),
        ]),
      ])
    );
  }
}

document.getElementById("memory-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const payload = {
    agent_profile: fd.get("agent_profile"),
    text: fd.get("text"),
    tags: (fd.get("tags") || "").split(",").map((s) => s.trim()).filter(Boolean),
  };
  await api("/api/memory", { method: "POST", body: JSON.stringify(payload) });
  e.target.reset();
  loadMemory();
});

// ---------- settings ----------
async function loadModels() {
  const list = document.getElementById("models-list");
  list.innerHTML = "";
  try {
    const models = await api("/api/models");
    for (const m of models.slice(0, 60)) {
      list.appendChild(el("div", { class: "card" }, [el("h4", { text: m.id }), el("p", { text: m.task || "" })]));
    }
  } catch (err) {
    list.appendChild(el("div", { class: "card" }, [el("p", { text: `Could not load models: ${err.message}` })]));
  }
}

// ---------- init ----------
loadProfileSelect().catch(() => {});
