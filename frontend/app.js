const API_URL = "http://localhost:8000";

let selectedFile = null;

// File input handling
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

// Handle file selected
function handleFileSelected(file) {
  selectedFile = file;

  // Show preview
  const reader = new FileReader();
  reader.onload = (e) => {
    document.getElementById("preview-img").src = e.target.result;
    document.getElementById("preview-filename").textContent = file.name;
    document.getElementById("upload-zone").style.display = "none";
    document.getElementById("preview-container").style.display = "block";
  };
  reader.readAsDataURL(file);

  // Reset results
  resetResults();
}

// Analyze Image — calls FastAPI
async function analyzeImage() {
  if (!selectedFile) return;

  // Show loading
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

    // Hide loading, show preview again
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

// Render results
function renderResults(data) {
  hideAll();

  // No damage found
  if (data.total_damages === 0) {
    document.getElementById("no-damage").style.display = "flex";
    return;
  }

  // Cost summary
  const costSummary = document.getElementById("cost-summary");
  costSummary.style.display = "block";

  document.getElementById("cost-amount").textContent =
    `PKR ${formatNumber(data.total_min)} – ${formatNumber(data.total_max)}`;

  document.getElementById("cost-range").textContent =
    `Estimated repair cost range`;

  const severityBadge = document.getElementById("severity-badge");
  severityBadge.innerHTML = `
    <span class="severity-badge severity-${data.overall_severity}">
      ${data.overall_severity.toUpperCase()} DAMAGE
    </span>
  `;

  // Damage list
  const damageList = document.getElementById("damage-list");
  damageList.style.display = "flex";
  damageList.innerHTML = "";

  data.damages.forEach((dmg, index) => {
    const item = document.createElement("div");
    item.className = "damage-item";
    item.style.animationDelay = `${index * 0.08}s`;

    const actionClass = dmg.action === "replace" ? "tag-replace" : "tag-repair";
    const confPct     = Math.round(dmg.confidence * 100);

    item.innerHTML = `
      <div>
        <div class="damage-type">${dmg.damage_type}</div>
        <div class="damage-part">on ${dmg.part_name}</div>
        <div class="damage-meta">
          <span class="meta-tag ${actionClass}">${dmg.action.toUpperCase()}</span>
          <span class="meta-tag severity-${dmg.severity_label}">${dmg.severity_label}</span>
          <span class="meta-tag tag-conf">${confPct}% conf</span>
        </div>
      </div>
      <div class="damage-cost">
        <div class="cost-min-max">
          ${formatNumber(dmg.min_cost)}–${formatNumber(dmg.max_cost)}
        </div>
        <div class="cost-pkr">PKR</div>
      </div>
    `;
    damageList.appendChild(item);
  });

  // Stats bar
  const statsBar = document.getElementById("stats-bar");
  statsBar.style.display = "flex";

  document.getElementById("stat-damages").textContent = data.total_damages;

  const uniqueParts = [...new Set(data.damages.map(d => d.part_name))].length;
  document.getElementById("stat-parts").textContent = uniqueParts;

  const replaceCount = data.damages.filter(d => d.action === "replace").length;
  document.getElementById("stat-action").textContent =
    replaceCount > data.total_damages / 2 ? "REPLACE" : "REPAIR";
}

// Show ERROR
function showError(message) {
  hideAll();
  const errorState = document.getElementById("error-state");
  errorState.style.display = "block";
  document.getElementById("error-msg").textContent = message;
}

// Helpers
function hideAll() {
  document.getElementById("empty-state").style.display    = "none";
  document.getElementById("error-state").style.display    = "none";
  document.getElementById("no-damage").style.display      = "none";
  document.getElementById("cost-summary").style.display   = "none";
  document.getElementById("damage-list").style.display    = "none";
  document.getElementById("stats-bar").style.display      = "none";
}

function resetResults() {
  hideAll();
  document.getElementById("empty-state").style.display = "flex";
}

function formatNumber(num) {
  return num.toLocaleString("en-PK");
}