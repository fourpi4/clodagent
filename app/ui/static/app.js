const state = { sessionId: null, profiles: [] };

// ---------- API key (local-only auth, never hardcoded) ----------
function getApiKey() {
  return localStorage.getItem("agent_api_key") || "";
}
function setApiKey(key) {
  if (key) localStorage.setItem("agent_api_key", key);
  else localStorage.removeItem("agent_api_key");
}

// ---------- helpers ----------
async function api(path, options = {}) {
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  const key = getApiKey();
  if (key) headers["Authorization"] = `Bearer ${key}`;
  const resp = await fetch(path, { ...options, headers });
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
    if (btn.dataset.view === "providers") loadProviders();
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

// ---------- chat (SSE streaming) ----------
const messagesEl = document.getElementById("messages");
const statusLogEl = document.getElementById("status-log");
const confirmationCardEl = document.getElementById("confirmation-card");

function addMessage(role, text) {
  messagesEl.appendChild(el("div", { class: `msg ${role}`, text }));
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function statusLine(text) {
  statusLogEl.appendChild(el("div", { text }));
  statusLogEl.scrollTop = statusLogEl.scrollHeight;
}

function renderEvent(event) {
  switch (event.type) {
    case "planning": statusLine("Planning..."); break;
    case "plan_ready":
      for (const step of event.plan || []) statusLine(`  • ${step}`);
      if (event.replanned) statusLine("  (plan revised after a failed step)");
      break;
    case "provider_selected": statusLine(`Using provider: ${event.provider}`); break;
    case "provider_fallback": statusLine(`Provider '${event.from}' unavailable (${event.reason}) → trying '${event.to || "..."}'`); break;
    case "tool_requested": statusLine(`Agent wants to use: ${event.name}`); break;
    case "tool_started": statusLine(`Using tool: ${event.name}`); break;
    case "tool_finished": statusLine(event.ok ? "Received result." : "Tool error."); break;
    case "generating": statusLine("Generating answer..."); break;
    case "confirmation_required":
      statusLine(`Waiting for approval to use: ${event.tool}`);
      break;
    default: break;
  }
}

function showConfirmationCard(confirmation) {
  confirmationCardEl.classList.remove("hidden");
  confirmationCardEl.innerHTML = "";
  confirmationCardEl.appendChild(el("h4", { text: `Agent wants to use: ${confirmation.tool}` }));
  confirmationCardEl.appendChild(el("p", { text: confirmation.reason || "" }));
  confirmationCardEl.appendChild(el("pre", { text: JSON.stringify(confirmation.arguments, null, 2) }));
  const actions = el("div", { class: "confirmation-actions" }, [
    el("button", {
      class: "btn-approve", text: "Approve",
      onclick: () => resolveConfirmation(confirmation.id, "approve"),
    }),
    el("button", {
      class: "btn-deny", text: "Deny",
      onclick: () => resolveConfirmation(confirmation.id, "deny"),
    }),
  ]);
  confirmationCardEl.appendChild(actions);
}

function hideConfirmationCard() {
  confirmationCardEl.classList.add("hidden");
  confirmationCardEl.innerHTML = "";
}

async function resolveConfirmation(id, decision) {
  hideConfirmationCard();
  statusLine(decision === "approve" ? "Approved. Resuming..." : "Denied. Resuming...");
  try {
    const result = await api(`/api/confirmations/${id}/${decision}`, { method: "POST" });
    handleRunResult(result);
  } catch (err) {
    addMessage("assistant", `Error resuming run: ${err.message}`);
  }
}

function handleRunResult(result) {
  if (result.session_id) state.sessionId = result.session_id;
  if (result.status === "confirmation_required" && result.confirmation) {
    showConfirmationCard(result.confirmation);
    return;
  }
  addMessage("assistant", result.final_answer || `[status: ${result.status}]`);
}

async function streamChat(message, profile) {
  const resp = await fetch("/api/chat/stream", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(getApiKey() ? { Authorization: `Bearer ${getApiKey()}` } : {}),
    },
    body: JSON.stringify({ message, session_id: state.sessionId, profile }),
  });
  if (!resp.ok || !resp.body) {
    addMessage("assistant", `Error: ${resp.status} ${await resp.text()}`);
    return;
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let assistantBubble = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop();
    for (const part of parts) {
      if (!part.startsWith("data:")) continue;
      let event;
      try {
        event = JSON.parse(part.slice(5).trim());
      } catch {
        continue;
      }
      if (event.type === "token") {
        if (!assistantBubble) {
          assistantBubble = el("div", { class: "msg assistant", text: "" });
          messagesEl.appendChild(assistantBubble);
        }
        assistantBubble.textContent += event.text;
        messagesEl.scrollTop = messagesEl.scrollHeight;
      } else if (event.type === "done") {
        handleRunResult(event.result);
      } else if (event.type === "error") {
        addMessage("assistant", `Error: ${event.error}`);
      } else {
        renderEvent(event);
      }
    }
  }
}

document.getElementById("chat-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = document.getElementById("chat-input");
  const message = input.value.trim();
  if (!message) return;
  input.value = "";
  addMessage("user", message);
  statusLogEl.innerHTML = "";
  hideConfirmationCard();

  const profile = document.getElementById("profile-select").value || undefined;
  try {
    await streamChat(message, profile);
  } catch (err) {
    addMessage("assistant", `Error: ${err.message}`);
  }
});

// ---------- agents ----------
async function populateProviderSelect() {
  const select = document.getElementById("agent-provider-select");
  try {
    const providers = await api("/api/providers?check=false");
    select.innerHTML = "";
    select.appendChild(el("option", { value: "", text: "(default)" }));
    for (const p of providers) {
      select.appendChild(el("option", { value: p.name, text: `${p.name}${p.configured ? "" : " (not configured)"}` }));
    }
  } catch {
    // providers endpoint unavailable — leave just the default option
  }
}

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
          el("span", { class: "badge", text: `provider: ${p.provider || "default"}` }),
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
  populateProviderSelect();
}

document.getElementById("agent-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const payload = {
    name: fd.get("name"),
    system_prompt: fd.get("system_prompt") || "You are a helpful, precise AI agent.",
    provider: fd.get("provider") || null,
    model: fd.get("model") || null,
    temperature: parseFloat(fd.get("temperature") || "0.2"),
    tools: (fd.get("tools") || "").split(",").map((s) => s.trim()).filter(Boolean),
    mcp_servers: (fd.get("mcp_servers") || "").split(",").map((s) => s.trim()).filter(Boolean),
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
          el("span", { class: "badge", text: `risk: ${t.risk_level}` }),
          t.side_effects ? el("span", { class: "badge warn", text: "side effects" }) : el("span", { class: "badge ok", text: "no side effects" }),
          t.retry_safe ? el("span", { class: "badge ok", text: "retry-safe" }) : el("span", { class: "badge warn", text: "not retried" }),
          t.requires_confirmation ? el("span", { class: "badge warn", text: "requires confirmation" }) : el("span", { class: "badge ok", text: "auto-runs" }),
        ]),
      ])
    );
  }
}

// ---------- github explorer ----------
function capabilityBadge(label, signal) {
  const info = signal && typeof signal === "object" ? signal : { available: !!signal, confidence: 0 };
  return el("span", {
    class: `badge ${info.available ? "ok" : ""}`,
    text: `${label}: ${info.available ? "yes" : "no"}${info.available ? ` (${Math.round((info.confidence || 0) * 100)}%)` : ""}`,
  });
}

async function searchGithub(query) {
  const results = query
    ? await api(`/api/github/search?q=${encodeURIComponent(query)}`)
    : await api("/api/github/search");
  const container = document.getElementById("github-results");
  container.dataset.loaded = "1";
  container.innerHTML = "";
  for (const r of results) {
    const licenseClass = r.license_category === "permissive" ? "ok" : r.license_category === "weak_copyleft" ? "warn" : "danger";
    const caps = r.capabilities || {};
    container.appendChild(
      el("div", { class: "card" }, [
        el("h4", { text: r.name }),
        el("p", { text: `⭐ ${r.stars} · ${r.language || "?"} · updated ${(r.last_update || "").slice(0, 10)}${r.archived ? " · ARCHIVED" : ""}${r.fork ? " · fork" : ""}` }),
        el("p", { text: r.description || "" }),
        el("div", { class: "badges" }, [
          el("span", { class: `badge ${licenseClass}`, text: `license: ${r.license} (${r.license_category})` }),
          capabilityBadge("API", caps.api),
          capabilityBadge("SDK", caps.sdk),
          capabilityBadge("MCP", caps.mcp),
        ]),
        el("div", { class: "card-actions" }, [
          el("a", { href: r.repository, target: "_blank", text: "Open on GitHub" }),
          el("button", { text: "Generate Integration Plan", onclick: () => analyzeRepo(r.repository) }),
          el("button", { text: "Generate Adapter (stub only)", onclick: () => generateAdapter(r.repository) }),
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

async function generateAdapter(url) {
  try {
    const plan = await api("/api/github/generate-adapter", {
      method: "POST",
      body: JSON.stringify({ full_name: fullNameFromUrl(url) }),
    });
    showModal(`Nothing was activated — a reviewable stub file was written only.\n\n${JSON.stringify(plan, null, 2)}`);
  } catch (err) {
    showModal(`Could not generate adapter: ${err.message}`);
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
    const actions = [
      el("button", { text: "Connect", onclick: async () => { try { const r = await api(`/api/mcp/connect?name=${encodeURIComponent(s.name)}`, { method: "POST" }); showModal(r); loadMcp(); } catch (err) { showModal(err.message); } } }),
      el("button", { text: s.enabled ? "Disable" : "Enable", onclick: async () => { await api(`/api/mcp/${encodeURIComponent(s.name)}/toggle`, { method: "POST", body: JSON.stringify({ enabled: !s.enabled }) }); loadMcp(); } }),
    ];
    if (!s.confirmed) {
      actions.unshift(el("button", { class: "btn-approve", text: "Confirm first-run", onclick: async () => { await api(`/api/mcp/${encodeURIComponent(s.name)}/confirm`, { method: "POST" }); loadMcp(); } }));
    }
    list.appendChild(
      el("div", { class: "card" }, [
        el("h4", { text: s.name }),
        el("div", { class: "badges" }, [
          el("span", { class: `badge ${s.enabled ? "ok" : "danger"}`, text: s.enabled ? "enabled" : "disabled" }),
          el("span", { class: "badge", text: `transport: ${s.transport}` }),
          el("span", { class: `badge ${s.trusted ? "ok" : "warn"}`, text: s.trusted ? "trusted (config file)" : "untrusted (added via API)" }),
          el("span", { class: `badge ${s.confirmed ? "ok" : "warn"}`, text: s.confirmed ? "confirmed" : "needs confirmation" }),
          el("span", { class: "badge", text: `tools: ${s.tools_discovered}` }),
        ]),
        el("div", { class: "card-actions" }, actions),
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
  try {
    const result = await api("/api/mcp/servers", { method: "POST", body: JSON.stringify(payload) });
    if (result.note) showModal(result.note);
    e.target.reset();
    loadMcp();
  } catch (err) {
    showModal(`Could not add server: ${err.message}`);
  }
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

// ---------- providers ----------
async function loadProviders() {
  const list = document.getElementById("providers-list");
  list.innerHTML = "";
  try {
    const providers = await api("/api/providers");
    for (const p of providers) {
      list.appendChild(
        el("div", { class: "card" }, [
          el("h4", { text: p.name }),
          el("div", { class: "badges" }, [
            el("span", { class: `badge ${p.configured ? "ok" : "danger"}`, text: p.configured ? "configured" : "not configured" }),
            el("span", { class: `badge ${p.available ? "ok" : "warn"}`, text: p.available ? "available" : "unavailable" }),
            el("span", { class: "badge", text: `model: ${p.default_model}` }),
            p.supports_tools ? el("span", { class: "badge ok", text: "tools" }) : el("span", { class: "badge warn", text: "no tools" }),
            p.supports_streaming ? el("span", { class: "badge ok", text: "streaming" }) : el("span", { class: "badge warn", text: "no streaming" }),
          ]),
        ])
      );
    }
  } catch (err) {
    list.appendChild(el("div", { class: "card" }, [el("p", { text: `Could not load providers: ${err.message}` })]));
  }
}

// ---------- settings ----------
document.getElementById("apikey-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const value = document.getElementById("apikey-input").value.trim();
  setApiKey(value);
  document.getElementById("apikey-input").value = "";
  showModal(value ? "API key saved locally in this browser." : "API key cleared.");
});

async function loadModels() {
  const list = document.getElementById("models-list");
  list.innerHTML = "";
  try {
    const models = await api("/api/models");
    for (const m of models.slice(0, 60)) {
      list.appendChild(el("div", { class: "card" }, [el("h4", { text: m.id }), el("p", { text: `${m.provider || ""} ${m.task || ""}` })]));
    }
  } catch (err) {
    list.appendChild(el("div", { class: "card" }, [el("p", { text: `Could not load models: ${err.message}` })]));
  }
}

// ---------- init ----------
loadProfileSelect().catch(() => {});
