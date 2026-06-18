const $ = (id) => document.getElementById(id);

const state = {
  chatImage: null,
  visionImage: null,
  memoryCount: 0,
  ollamaReady: false,
  busy: false,
};

const els = {
  apiStatus: $("apiStatus"),
  ollamaStatus: $("ollamaStatus"),
  memoryStatus: $("memoryStatus"),
  modelName: $("modelName"),
  memoryCount: $("memoryCount"),
  conversation: $("conversation"),
  chatForm: $("chatForm"),
  chatInput: $("chatInput"),
  chatImage: $("chatImage"),
  chatPreview: $("chatPreview"),
  chatPreviewImg: $("chatPreviewImg"),
  clearChatImage: $("clearChatImage"),
  useMemory: $("useMemory"),
  storeAnswer: $("storeAnswer"),
  visionForm: $("visionForm"),
  visionQuestion: $("visionQuestion"),
  visionImage: $("visionImage"),
  visionPreview: $("visionPreview"),
  visionPreviewImg: $("visionPreviewImg"),
  clearVisionImage: $("clearVisionImage"),
  cameraButton: $("cameraButton"),
  memoryAddForm: $("memoryAddForm"),
  memoryText: $("memoryText"),
  memorySearchForm: $("memorySearchForm"),
  memoryQuery: $("memoryQuery"),
  memoryResult: $("memoryResult"),
  canvas: $("signalCanvas"),
};

function setPill(el, text, ready) {
  el.textContent = text;
  el.classList.toggle("ready", ready === true);
  el.classList.toggle("warn", ready === false);
}

function appendMessage(role, text) {
  const node = document.createElement("div");
  node.className = `message ${role}`;
  node.textContent = text;
  els.conversation.appendChild(node);
  els.conversation.scrollTop = els.conversation.scrollHeight;
  return node;
}

function setBusy(isBusy) {
  state.busy = isBusy;
  document.querySelectorAll("button").forEach((button) => {
    button.disabled = isBusy;
  });
}

async function loadHealth() {
  try {
    const res = await fetch("/health");
    const health = await res.json();
    state.ollamaReady = Boolean(health.ollama_ready);
    state.memoryCount = Number(health.memory_count || 0);
    setPill(els.apiStatus, "API ready", true);
    setPill(
      els.ollamaStatus,
      state.ollamaReady ? "Ollama ready" : "Ollama offline",
      state.ollamaReady
    );
    setPill(els.memoryStatus, `${state.memoryCount} memories`, true);
    els.modelName.textContent = health.ollama_model || "qwen3-vl:4b";
    els.memoryCount.textContent = String(state.memoryCount);
  } catch (error) {
    setPill(els.apiStatus, "API offline", false);
    setPill(els.ollamaStatus, "Ollama unknown", false);
    setPill(els.memoryStatus, "Memory unknown", false);
  }
}

async function fileToDataUrl(file) {
  if (!file) return null;
  return await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function setPreview(kind, dataUrl) {
  const preview = kind === "chat" ? els.chatPreview : els.visionPreview;
  const image = kind === "chat" ? els.chatPreviewImg : els.visionPreviewImg;
  image.src = dataUrl || "";
  preview.hidden = !dataUrl;
}

async function onImagePick(kind, file) {
  const dataUrl = await fileToDataUrl(file);
  if (kind === "chat") {
    state.chatImage = dataUrl;
  } else {
    state.visionImage = dataUrl;
  }
  setPreview(kind, dataUrl);
}

async function postJson(path, payload) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || data.message || "Request failed");
  }
  return data;
}

async function sendChat(event) {
  event.preventDefault();
  const message = els.chatInput.value.trim();
  if (!message || state.busy) return;

  appendMessage("user", message);
  els.chatInput.value = "";
  setBusy(true);

  const pending = appendMessage("assistant", "Thinking...");
  try {
    const data = await postJson("/chat", {
      message,
      use_memory: els.useMemory.checked,
      store: els.storeAnswer.checked,
      mode: "site",
      image_base64: state.chatImage,
    });
    pending.textContent = data.answer || "No answer returned.";
    if (data.memory_context) {
      appendMessage("system", `Memory context:\n${data.memory_context}`);
    }
    await loadHealth();
  } catch (error) {
    pending.textContent = error.message;
  } finally {
    setBusy(false);
  }
}

async function askVision(event, useCamera = false) {
  event.preventDefault();
  const question = els.visionQuestion.value.trim() || "What can you see?";
  setBusy(true);

  appendMessage("user", `Vision: ${question}`);
  const pending = appendMessage("assistant", "Looking...");
  try {
    const data = await postJson("/vision", {
      question,
      use_memory: els.useMemory.checked,
      store: els.storeAnswer.checked,
      mode: "site-vlm",
      capture_camera: useCamera,
      image_base64: useCamera ? null : state.visionImage,
    });
    pending.textContent = data.answer || "No vision answer returned.";
    await loadHealth();
  } catch (error) {
    pending.textContent = error.message;
  } finally {
    setBusy(false);
  }
}

async function saveMemory(event) {
  event.preventDefault();
  const text = els.memoryText.value.trim();
  if (!text || state.busy) return;

  setBusy(true);
  try {
    const data = await postJson("/memory", {
      text,
      category: "site",
      metadata: { source: "web-ui" },
    });
    els.memoryText.value = "";
    els.memoryResult.textContent = `Stored. Memory count: ${data.memory_count}`;
    await loadHealth();
  } catch (error) {
    els.memoryResult.textContent = error.message;
  } finally {
    setBusy(false);
  }
}

async function searchMemory(event) {
  event.preventDefault();
  const query = els.memoryQuery.value.trim();
  if (!query || state.busy) return;

  setBusy(true);
  try {
    const data = await postJson("/memory/search", { query });
    els.memoryResult.textContent = data.context || "No matching memory.";
  } catch (error) {
    els.memoryResult.textContent = error.message;
  } finally {
    setBusy(false);
  }
}

function drawSignal() {
  const canvas = els.canvas;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  const now = performance.now() / 1000;
  ctx.clearRect(0, 0, w, h);

  ctx.fillStyle = "#151719";
  ctx.fillRect(0, 0, w, h);

  const cols = 9;
  const rows = 5;
  const gapX = w / (cols + 1);
  const gapY = h / (rows + 1);
  const nodes = [];

  for (let y = 1; y <= rows; y += 1) {
    for (let x = 1; x <= cols; x += 1) {
      const wave = Math.sin(now * 1.8 + x * 0.7 + y * 0.9);
      nodes.push({
        x: x * gapX + wave * 4,
        y: y * gapY + Math.cos(now + x) * 3,
        active: (x + y + Math.floor(now)) % 4 === 0,
      });
    }
  }

  ctx.lineWidth = 1;
  nodes.forEach((a, index) => {
    for (let j = index + 1; j < nodes.length; j += 1) {
      const b = nodes[j];
      const dx = a.x - b.x;
      const dy = a.y - b.y;
      const distance = Math.sqrt(dx * dx + dy * dy);
      if (distance < 118) {
        const alpha = 1 - distance / 118;
        ctx.strokeStyle = `rgba(112, 184, 169, ${alpha * 0.22})`;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }
    }
  });

  nodes.forEach((node) => {
    const radius = node.active ? 4 : 2.6;
    ctx.fillStyle = state.ollamaReady
      ? node.active
        ? "#8fd6be"
        : "#d7efe6"
      : node.active
        ? "#e0a074"
        : "#c9cdd3";
    ctx.beginPath();
    ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
    ctx.fill();
  });

  ctx.fillStyle = "rgba(255,255,255,0.82)";
  ctx.font = "700 16px system-ui, sans-serif";
  ctx.fillText(state.ollamaReady ? "VLM online" : "VLM waiting", 22, 34);
  ctx.font = "600 13px system-ui, sans-serif";
  ctx.fillStyle = "rgba(255,255,255,0.62)";
  ctx.fillText(`${state.memoryCount} memory vectors`, 22, 58);

  requestAnimationFrame(drawSignal);
}

els.chatForm.addEventListener("submit", sendChat);
els.visionForm.addEventListener("submit", askVision);
els.cameraButton.addEventListener("click", (event) => askVision(event, true));
els.memoryAddForm.addEventListener("submit", saveMemory);
els.memorySearchForm.addEventListener("submit", searchMemory);
els.chatImage.addEventListener("change", (event) => onImagePick("chat", event.target.files[0]));
els.visionImage.addEventListener("change", (event) => onImagePick("vision", event.target.files[0]));
els.clearChatImage.addEventListener("click", () => {
  state.chatImage = null;
  els.chatImage.value = "";
  setPreview("chat", null);
});
els.clearVisionImage.addEventListener("click", () => {
  state.visionImage = null;
  els.visionImage.value = "";
  setPreview("vision", null);
});

appendMessage("system", "Arju web session ready.");
loadHealth();
setInterval(loadHealth, 8000);
drawSignal();
