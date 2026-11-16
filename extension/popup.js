// Popup Script for PhishBP Extension

document.addEventListener("DOMContentLoaded", async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab || !tab.id) return;

  chrome.storage.local.get([tab.id.toString()], (result) => {
    const data = result[tab.id.toString()];
    if (!data) {
      document.getElementById("domain-val").textContent = tab.url ? new URL(tab.url).hostname : "No active tab";
      return;
    }

    renderPopup(data);
  });
});

function renderPopup(data) {
  document.getElementById("domain-val").textContent = data.domain;

  const pct = Math.round(data.phishing_score * 100);
  document.getElementById("score-val").textContent = `${pct}%`;
  
  const fill = document.getElementById("score-fill");
  fill.style.width = `${pct}%`;

  const badge = document.getElementById("verdict-badge");
  badge.textContent = data.verdict;
  badge.className = `badge badge-${data.badge_color}`;

  if (data.badge_color === "red") {
    fill.style.background = "#c0392b";
  } else if (data.badge_color === "yellow") {
    fill.style.background = "#f39c12";
  } else {
    fill.style.background = "#27ae60";
  }

  document.getElementById("dns-status").textContent = data.dns_resolved ? "✓ Live" : "✗ Unresolved / NXDOMAIN";
  document.getElementById("ip-val").textContent = data.ips.length > 0 ? data.ips.slice(0, 3).join(", ") : "None";

  // Risks
  if (data.top_risk_factors && data.top_risk_factors.length > 0) {
    const riskCard = document.getElementById("risk-card");
    const riskList = document.getElementById("risk-list");
    riskList.innerHTML = "";
    data.top_risk_factors.forEach(r => {
      const li = document.createElement("li");
      li.textContent = r;
      riskList.appendChild(li);
    });
    riskCard.style.display = "block";
  }
}
