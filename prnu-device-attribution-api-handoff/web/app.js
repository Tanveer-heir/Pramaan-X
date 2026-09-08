const state = {
  status: null,
  result: null,
  view: new URLSearchParams(window.location.search).get("view") === "prnu" ? "prnu" : "device",
};

const $ = (id) => document.getElementById(id);

function pct(value) {
  const n = Number(value || 0);
  return `${Math.round(n * 100)}%`;
}

function label(value) {
  return value || "-";
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok || payload.ok === false) {
    throw new Error(payload.error || `Request failed: ${response.status}`);
  }
  return payload;
}

async function refreshStatus() {
  const payload = await fetchJson("/api/status");
  state.status = payload;
  renderStatus(payload);
}

function renderStatus(payload) {
  const summary = payload.dataset_summary || {};
  const devices = summary.devices || {};
  $("modelBadge").textContent = payload.model_trained ? "Model Ready" : "Not Trained";
  $("datasetSummary").innerHTML = [
    summaryItem("Images", summary.total_images || 0),
    summaryItem("Devices", Object.keys(devices).length),
    summaryItem("Dataset", payload.dataset || "-"),
  ].join("");

  const warnings = [...(payload.model?.warnings || [])];
  const trainedRoot = String(payload.model?.dataset_root || payload.dataset || "");
  if (trainedRoot.toLowerCase().includes("sample_dataset")) {
    warnings.unshift("Demo model is trained on synthetic sample_dataset. Real phone names appear only after you train on your actual dataset folder.");
  }
  if ((summary.total_images || 0) === 0) {
    warnings.unshift("The active dataset folder has no real images yet.");
  }
  $("warnings").textContent = warnings.length ? warnings.slice(0, 4).join(" ") : "";
}

function summaryItem(name, value) {
  return `<div class="summary-item"><p>${name}</p><strong>${value}</strong></div>`;
}

async function trainModel() {
  const btn = $("trainBtn");
  btn.disabled = true;
  btn.textContent = "Training...";
  try {
    await fetchJson("/api/train", { method: "POST" });
    await refreshStatus();
  } catch (error) {
    alert(error.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Train";
  }
}

async function analyze(event) {
  event.preventDefault();
  const file = $("imageInput").files?.[0];
  if (!file) {
    alert("Choose at least one image first.");
    return;
  }
  $("resultBadge").textContent = "Running";
  const data = new FormData();
  data.append("image", file);
  try {
    const payload = await fetchJson("/api/analyze?mode=device", { method: "POST", body: data });
    state.result = payload.result;
    renderResult(payload.result);
    $("resultBadge").textContent = "Complete";
  } catch (error) {
    $("resultBadge").textContent = "Error";
    alert(error.message);
  }
}

async function matchPrnu(event) {
  event.preventDefault();
  const references = Array.from($("referenceInput").files || []);
  const query = $("queryInput").files?.[0];
  if (!references.length || !query) {
    alert("Choose reference images and one unseen query image first.");
    return;
  }

  const button = $("matchPrnuButton");
  button.disabled = true;
  button.textContent = "Matching...";
  $("prnuResultBadge").textContent = "Running";
  const data = new FormData();
  references.forEach((file) => data.append("reference", file));
  data.append("query", query);
  const folderLabel = references[0]?.webkitRelativePath?.split("/")[0] || "";
  const endpoint = folderLabel
    ? `/api/prnu-match?reference_label=${encodeURIComponent(folderLabel)}`
    : "/api/prnu-match";
  try {
    const payload = await fetchJson(endpoint, { method: "POST", body: data });
    renderPrnuResult(payload.result);
    $("prnuResultBadge").textContent = "Complete";
  } catch (error) {
    $("prnuResultBadge").textContent = "Error";
    alert(error.message);
  } finally {
    button.disabled = false;
    button.textContent = "Build Fingerprint & Match";
  }
}

function renderResult(result) {
  const device = result.channels?.device || result.device_attribution || {};
  $("devicePrediction").textContent = label(device.display_name || device.prediction);
  $("deviceMeter").style.width = pct(device.confidence);
  const deviceProfile = device.metadata_profile || {};
  const deviceHint = deviceProfile.model || deviceProfile.make || "";
  $("deviceMethod").textContent = `${pct(device.confidence)} | ${label(device.primary_method)}${deviceHint ? ` | ${deviceHint}` : ""}`;
  $("knownDeviceLabel").textContent = label(device.prediction);
  $("cameraModelHint").textContent = label(deviceProfile.model || deviceProfile.make || result.metadata?.model || result.metadata?.make);
  $("deviceTemplate").textContent = label(result.rankings?.prnu?.[0]?.best_template);
  const deviceEvidence = [
    ...(device.evidence || []),
    ...(result.warnings || []).filter((w) => /PRNU|device/i.test(w)).map((w) => `Warning: ${w}`),
  ];
  $("deviceEvidenceList").innerHTML = deviceEvidence.length
    ? deviceEvidence.map((item) => `<li>${escapeHtml(item)}</li>`).join("")
    : "<li>No device evidence available.</li>";

  const meta = result.metadata || {};
  const compression = meta.compression || {};
  const rows = {
    File: meta.file_name,
    SHA256: meta.sha256,
    Format: meta.format,
    Size: meta.width && meta.height ? `${meta.width} x ${meta.height}` : "-",
    Make: meta.make,
    Model: meta.model,
    Software: meta.software,
    EXIF: meta.exif_present ? `${meta.exif_tag_count} tags` : "absent",
    "JPEG Q Hash": compression.quant_hash,
    "Bytes/MP": compression.bytes_per_megapixel,
  };
  $("metadataList").innerHTML = Object.entries(rows)
    .map(([key, value]) => `<dt>${key}</dt><dd>${escapeHtml(label(value))}</dd>`)
    .join("");
}

function renderPrnuResult(result) {
  const status = result.status || "inconclusive";
  const decision = label(result.device_name || result.decision);
  $("prnuDecision").textContent = decision;
  $("prnuDecision").className = `decision-${status}`;
  $("prnuMeter").style.width = pct(result.confidence);
  $("prnuSummary").textContent = `${pct(result.confidence)} confidence | ${label(result.method)} | ${result.usable_reference_count || 0} usable reference image(s)`;
  $("prnuCorrelation").textContent = label(result.correlation);
  $("prnuConfidence").textContent = pct(result.confidence);
  $("prnuReferenceCount").textContent = `${result.usable_reference_count || 0} / ${result.reference_count || 0} usable`;
  const meta = result.query_metadata || {};
  $("prnuQueryCamera").textContent = label(meta.model || meta.make);
  const evidence = result.evidence || [];
  if (result.reference_errors?.length) {
    evidence.push(...result.reference_errors.map((item) => `Reference warning: ${item}`));
  }
  $("prnuEvidenceList").innerHTML = evidence.length
    ? evidence.map((item) => `<li>${escapeHtml(item)}</li>`).join("")
    : "<li>No PRNU evidence available.</li>";
}

function setView(view) {
  state.view = view === "prnu" ? "prnu" : "device";
  $("deviceView").hidden = state.view !== "device";
  $("prnuView").hidden = state.view !== "prnu";
  document.querySelectorAll(".view-tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === state.view);
  });
  const url = new URL(window.location.href);
  if (state.view === "prnu") url.searchParams.set("view", "prnu");
  else url.searchParams.delete("view");
  window.history.replaceState({}, "", url);
}

function renderReferenceFiles() {
  const files = Array.from($("referenceInput").files || []);
  $("referenceBadge").textContent = files.length ? `${files.length} files` : "No files";
  $("referenceFileList").innerHTML = files.length
    ? files.map((file) => `<li>${escapeHtml(file.name)}</li>`).join("")
    : "";
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

$("refreshBtn").addEventListener("click", refreshStatus);
$("trainBtn").addEventListener("click", trainModel);
$("analyzeForm").addEventListener("submit", analyze);
$("prnuForm").addEventListener("submit", matchPrnu);
document.querySelectorAll(".view-tab").forEach((button) => {
  button.addEventListener("click", () => setView(button.dataset.view));
});
$("imageInput").addEventListener("change", () => {
  const file = $("imageInput").files[0];
  if (!file) return;
  $("preview").src = URL.createObjectURL(file);
  $("previewWrap").classList.remove("hidden");
});

$("referenceInput").addEventListener("change", renderReferenceFiles);
$("queryInput").addEventListener("change", () => {
  const file = $("queryInput").files[0];
  $("queryBadge").textContent = file ? file.name : "No file";
  if (!file) return;
  $("queryPreview").src = URL.createObjectURL(file);
  $("queryPreviewWrap").classList.remove("hidden");
});

setView(state.view);
refreshStatus().catch((error) => {
  $("modelBadge").textContent = "Offline";
  $("warnings").textContent = error.message;
});
