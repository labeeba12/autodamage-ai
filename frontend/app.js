const API_URL = "http://localhost:8000";
 
let selectedFile = null;
 
// FILE INPUT
document.getElementById("file-input").addEventListener("change", function (e) {
  const file = e.target.files[0];
  if (file) handleFileSelected(file);
});
 
// Drag and drop
const uploadZone = document.getElementById("upload-zone");
 
uploadZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadZone.classList.add("dragover");
});
 
uploadZone.addEventListener("dragleave", () => {
  uploadZone.classList.remove("dragover");
});
 
uploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadZone.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file) handleFileSelected(file);
});
 
// HANDLE FILE SELECTED
function handleFileSelected(file) {
  selectedFile = file;
 
  const reader = new FileReader();
  reader.onload = (e) => {
    document.getElementById("preview-img").src = e.target.result;
    document.getElementById("preview-filename").textContent = file.name;
    document.getElementById("upload-zone").style.display = "none";
    document.getElementById("preview-container").style.display = "block";
  };
  reader.readAsDataURL(file);
 
  resetResults();
}
 
// ANALYZE IMAGE
async function analyzeImage() {
  if (!selectedFile) return;
 
  document.getElementById("preview-container").style.display = "none";
  document.getElementById("loading-state").style.display = "flex";
  resetResults();
 
  try {
    const formData = new FormData();
    formData.append("file", selectedFile);
 
    const response = await fetch(`${API_URL}/estimate`, {
      method: "POST",
      body: formData,
    });
 
    const data = await response.json();
 
    document.getElementById("loading-state").style.display = "none";
    document.getElementById("preview-container").style.display = "block";
 
    if (!response.ok) {
      showError(data.detail || "Analysis failed. Please try again.");
      return;
    }
 
    renderResults(data);
 
  } catch (err) {
    document.getElementById("loading-state").style.display = "none";
    document.getElementById("preview-container").style.display = "block";
    showError("Cannot connect to API. Make sure the server is running on port 8000.");
  }
}
 
// RENDER RESULTS
function renderResults(data) {
  window._lastReportData = data;
  hideAll();
 
  // No damage
  if (data.total_damages === 0) {
    document.getElementById("no-damage").style.display = "flex";
    return;
  }
 
  // Cost summary
  document.getElementById("cost-summary").style.display = "block";
  document.getElementById("cost-amount").innerHTML =
    `<span class="cur">PKR</span> ${formatNumber(data.total_min)} – ${formatNumber(data.total_max)}`;
  document.getElementById("cost-range").textContent = "Estimated repair cost range";
  document.getElementById("severity-badge").innerHTML =
    `<span class="sev-chip sev-${data.overall_severity}">${data.overall_severity.toUpperCase()} DAMAGE</span>`;
 
  // Damage list
  const damageList = document.getElementById("damage-list");
  damageList.style.display = "flex";
  damageList.innerHTML = "";
 
  data.damages.forEach((dmg, index) => {
    const item = document.createElement("div");
    item.className = "damage-item";
    item.style.animationDelay = `${index * 0.07}s`;
 
    const actionClass = dmg.action === "replace" ? "t-replace" : "t-repair";
    const sevClass    = `t-sev-${dmg.severity_label}`;
    const confPct     = Math.round(dmg.confidence * 100);
 
    item.innerHTML = `
      <div>
        <div class="di-type">${dmg.damage_type}</div>
        <div class="di-part">on ${dmg.part_name}</div>
        <div class="di-tags">
          <span class="tag ${actionClass}">${dmg.action.toUpperCase()}</span>
          <span class="tag ${sevClass}">${dmg.severity_label}</span>
          <span class="tag t-conf">${confPct}% conf</span>
        </div>
      </div>
      <div class="di-cost">
        <div class="di-cost-range">${formatNumber(dmg.min_cost)}–${formatNumber(dmg.max_cost)}</div>
        <div class="di-cost-cur">PKR</div>
      </div>
    `;
    damageList.appendChild(item);
  });
 
  // Heatmap
   if (data.heatmap_path) {
    const filename = data.heatmap_path.split(/[\\/]/).pop();
    document.getElementById("heatmap-img").src = `${API_URL}/heatmap/${filename}`;
    document.getElementById("heatmap-container").style.display = "flex";
    document.getElementById("images-grid").style.display = "grid";
    document.getElementById("images-section").style.display = "block";
  }

  if (data.result_image_path) {
    const filename = data.result_image_path.split(/[\\/]/).pop();
    document.getElementById("before-after-img").src = `${API_URL}/result-image/${filename}`;
    document.getElementById("before-after-container").style.display = "flex";
    document.getElementById("images-grid").style.display = "grid";
    document.getElementById("images-section").style.display = "block";
  }

  // Paint
  if (data.paint_color) {
    document.getElementById("paint-dot").style.background = data.paint_hex || "#ccc";
    document.getElementById("paint-name").textContent = `${data.paint_color} (${data.paint_code})`;
    document.getElementById("paint-cost").textContent =
      `Paint cost: PKR ${data.paint_cost_min?.toLocaleString()} – ${data.paint_cost_max?.toLocaleString()}`;
    document.getElementById("paint-container").style.display = "flex";
  }
 
  // Car model — show with color-coded confidence
  if (data.car_model) {
    const name = data.car_model.replace(/_/g, " ");
    const conf = Math.round(data.car_model_confidence * 100);
    const confColor = conf >= 80 ? "#34d399" : conf >= 50 ? "#e3b341" : "#f85149";
    document.getElementById("car-model-name").textContent = name;
    document.getElementById("car-model-conf").textContent = `${conf}% confidence`;
    document.getElementById("car-model-conf").style.color = confColor;
    document.getElementById("car-model-container").style.display = "flex";
  }
 
  // Stats
  document.getElementById("stats-bar").style.display = "block";
  document.getElementById("stat-damages").textContent = data.total_damages;
  const uniqueParts = [...new Set(data.damages.map(d => d.part_name))].length;
  document.getElementById("stat-parts").textContent = uniqueParts;
  const replaceCount = data.damages.filter(d => d.action === "replace").length;
  document.getElementById("stat-action").textContent =
    replaceCount > data.total_damages / 2 ? "REPLACE" : "REPAIR";
 
  // Action footer
  document.getElementById("action-footer").style.display = "block";
}
 
// SHOW ERROR
function showError(message) {
  hideAll();
  document.getElementById("error-state").style.display = "block";
  document.getElementById("error-msg").textContent = message;
}
 
// HIDE ALL
function hideAll() {
  document.getElementById("empty-state").style.display         = "none";
  document.getElementById("error-state").style.display         = "none";
  document.getElementById("no-damage").style.display           = "none";
  document.getElementById("cost-summary").style.display        = "none";
  document.getElementById("damage-list").style.display         = "none";
  document.getElementById("images-section").style.display      = "none";
  document.getElementById("images-grid").style.display         = "none";
  document.getElementById("heatmap-container").style.display   = "none";
  document.getElementById("before-after-container").style.display = "none";
  document.getElementById("paint-container").style.display     = "none";
  document.getElementById("car-model-container").style.display = "none";
  document.getElementById("stats-bar").style.display           = "none";
  document.getElementById("action-footer").style.display       = "none";
}
 
// RESET
function resetResults() {
  hideAll();
  document.getElementById("empty-state").style.display = "flex";
}
 
// START NEW
function startNew() {
  selectedFile = null;
  document.getElementById("preview-container").style.display = "none";
  document.getElementById("upload-zone").style.display = "flex";
  document.getElementById("file-input").value = "";
  resetResults();
}
 
// HELPERS
function formatNumber(num) {
  return num.toLocaleString("en-PK");
}

// PRINT REPORT
function printReport() {
  if (!window._lastReportData) return;
  const data = window._lastReportData;

  const now = new Date();
  const dateStr = now.toLocaleDateString("en-PK", {
    day: "numeric", month: "long", year: "numeric"
  });
  const timeStr = now.toLocaleTimeString("en-PK", {
    hour: "2-digit", minute: "2-digit"
  });

  const damagesHTML = data.damages.map((dmg, i) => `
    <tr>
      <td>${i + 1}</td>
      <td style="text-transform:capitalize;">${dmg.damage_type}</td>
      <td>${dmg.part_name}</td>
      <td>${dmg.severity_label}</td>
      <td>${dmg.action.toUpperCase()}</td>
      <td>${Math.round(dmg.confidence * 100)}%</td>
      <td>PKR ${dmg.min_cost.toLocaleString()} – ${dmg.max_cost.toLocaleString()}</td>
    </tr>
  `).join("");

  const reportHTML = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8"/>
  <title>AutoDamage AI — Damage Report</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Segoe UI', Arial, sans-serif;
      color: #1a1a2e;
      background: #fff;
      padding: 2.5rem;
      font-size: 13px;
    }
    .header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      padding-bottom: 1.5rem;
      border-bottom: 2px solid #1a1a2e;
      margin-bottom: 1.5rem;
    }
    .logo {
      display: flex;
      align-items: center;
      gap: 0.6rem;
    }
    .logo-mark {
      width: 36px; height: 36px;
      background: #2563eb;
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: white;
      font-weight: 800;
      font-size: 0.8rem;
    }
    .logo-text {
      font-size: 1.1rem;
      font-weight: 700;
      color: #1a1a2e;
    }
    .logo-text span { color: #2563eb; }
    .report-meta {
      text-align: right;
      font-size: 0.75rem;
      color: #666;
      line-height: 1.6;
    }
    .report-meta strong { color: #1a1a2e; }
    .section {
      margin-bottom: 1.75rem;
    }
    .section-title {
      font-size: 0.65rem;
      font-weight: 600;
      color: #666;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      margin-bottom: 0.75rem;
      padding-bottom: 0.4rem;
      border-bottom: 1px solid #e5e7eb;
    }
    .cost-box {
      background: #f0f7ff;
      border: 1px solid #bfdbfe;
      border-radius: 8px;
      padding: 1.25rem 1.5rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .cost-amount {
      font-size: 1.8rem;
      font-weight: 800;
      color: #1a1a2e;
      letter-spacing: -0.03em;
    }
    .cost-amount .cur { font-size: 1rem; color: #666; font-weight: 500; }
    .cost-sub { font-size: 0.75rem; color: #666; margin-top: 0.2rem; }
    .sev-badge {
      padding: 0.3rem 0.8rem;
      border-radius: 20px;
      font-size: 0.7rem;
      font-weight: 600;
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }
    .sev-minor    { background: #dcfce7; color: #166534; }
    .sev-moderate { background: #fef9c3; color: #854d0e; }
    .sev-severe   { background: #fee2e2; color: #991b1b; }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.82rem;
    }
    th {
      background: #f8fafc;
      padding: 0.6rem 0.75rem;
      text-align: left;
      font-size: 0.65rem;
      font-weight: 600;
      color: #666;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      border-bottom: 1px solid #e5e7eb;
    }
    td {
      padding: 0.7rem 0.75rem;
      border-bottom: 1px solid #f1f5f9;
      color: #334155;
    }
    tr:last-child td { border-bottom: none; }
    .info-grid {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 0.75rem;
    }
    .info-card {
      background: #f8fafc;
      border: 1px solid #e5e7eb;
      border-radius: 6px;
      padding: 0.75rem 1rem;
    }
    .info-card-label {
      font-size: 0.62rem;
      color: #666;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 0.25rem;
    }
    .info-card-value {
      font-size: 0.875rem;
      font-weight: 600;
      color: #1a1a2e;
    }
    .info-card-sub {
      font-size: 0.72rem;
      color: #2563eb;
      margin-top: 0.15rem;
    }
    .stats-row {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 0.75rem;
    }
    .stat-box {
      background: #f8fafc;
      border: 1px solid #e5e7eb;
      border-radius: 6px;
      padding: 0.75rem 1rem;
      text-align: center;
    }
    .stat-box-num {
      font-size: 1.5rem;
      font-weight: 800;
      color: #1a1a2e;
      letter-spacing: -0.03em;
    }
    .stat-box-label {
      font-size: 0.62rem;
      color: #666;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      margin-top: 0.2rem;
    }
    .footer {
      margin-top: 2rem;
      padding-top: 1rem;
      border-top: 1px solid #e5e7eb;
      display: flex;
      justify-content: space-between;
      font-size: 0.7rem;
      color: #999;
    }
    @media print {
      body { padding: 1.5rem; }
      @page { margin: 1cm; }
    }
  </style>
</head>
<body>

  <div class="header">
    <div class="logo">
      <div class="logo-mark">AD</div>
      <div>
        <div class="logo-text">Auto<span>Damage</span> AI</div>
        <div style="font-size:0.7rem; color:#666; margin-top:0.1rem;">Intelligent Car Damage Assessment</div>
      </div>
    </div>
    <div class="report-meta">
      <div><strong>Report Generated</strong></div>
      <div>${dateStr} at ${timeStr}</div>
      <div style="margin-top:0.3rem;">AutoDamage AI v1.0</div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Cost Summary</div>
    <div class="cost-box">
      <div>
        <div class="cost-amount"><span class="cur">PKR </span>${data.total_min.toLocaleString()} – ${data.total_max.toLocaleString()}</div>
        <div class="cost-sub">Estimated total repair cost range</div>
      </div>
      <span class="sev-badge sev-${data.overall_severity}">${data.overall_severity} damage</span>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Damage Breakdown</div>
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>Damage Type</th>
          <th>Part</th>
          <th>Severity</th>
          <th>Action</th>
          <th>Confidence</th>
          <th>Cost Range</th>
        </tr>
      </thead>
      <tbody>${damagesHTML}</tbody>
    </table>
  </div>

  <div class="section">
    <div class="section-title">Vehicle Information</div>
    <div class="info-grid">
      ${data.car_model ? `
      <div class="info-card">
        <div class="info-card-label">Vehicle Model</div>
        <div class="info-card-value">${data.car_model.replace(/_/g, " ")}</div>
        <div class="info-card-sub">${Math.round(data.car_model_confidence * 100)}% confidence</div>
      </div>` : ""}
      ${data.paint_color ? `
      <div class="info-card">
        <div class="info-card-label">Paint Color</div>
        <div class="info-card-value">${data.paint_color} (${data.paint_code})</div>
        <div class="info-card-sub">PKR ${data.paint_cost_min?.toLocaleString()} – ${data.paint_cost_max?.toLocaleString()}</div>
      </div>` : ""}
      <div class="info-card">
        <div class="info-card-label">Overall Severity</div>
        <div class="info-card-value" style="text-transform:capitalize;">${data.overall_severity}</div>
        <div class="info-card-sub">${data.total_damages} damage(s) detected</div>
      </div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Summary Statistics</div>
    <div class="stats-row">
      <div class="stat-box">
        <div class="stat-box-num">${data.total_damages}</div>
        <div class="stat-box-label">Damages Found</div>
      </div>
      <div class="stat-box">
        <div class="stat-box-num">${[...new Set(data.damages.map(d => d.part_name))].length}</div>
        <div class="stat-box-label">Parts Affected</div>
      </div>
      <div class="stat-box">
        <div class="stat-box-num">${data.damages.filter(d => d.action === "replace").length}</div>
        <div class="stat-box-label">Parts to Replace</div>
      </div>
    </div>
  </div>

  <div class="footer">
    <span>Generated by AutoDamage AI · Powered by YOLOv11</span>
    <span>This report is an AI estimate. Consult a professional for final assessment.</span>
  </div>

  <div class="footer">
    <span>Generated by AutoDamage AI · Powered by YOLOv11</span>
    <span>This report is an AI estimate. Consult a professional for final assessment.</span>
  </div>

  <script>window.onload = () => window.print();</script>
</body>
</html>`;

  const win = window.open("", "_blank");
  win.document.write(reportHTML);
  win.document.close();
}
 