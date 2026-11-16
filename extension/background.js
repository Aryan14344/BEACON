// PhishBP Chrome Extension - Background Service Worker

const API_ENDPOINT = "http://127.0.0.1:8000/predict";

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === 'loading' && tab.url && tab.url.startsWith("http")) {
    inspectUrl(tab.url, tabId);
  }
});

async function inspectUrl(url, tabId) {
  try {
    const response = await fetch(API_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: url })
    });

    if (!response.ok) return;

    const data = await response.json();

    // Cache latest status for popup
    chrome.storage.local.set({ [tabId]: data });

    // Update Extension Badge
    if (data.verdict === "PHISHING") {
      chrome.action.setBadgeText({ text: "!", tabId: tabId });
      chrome.action.setBadgeBackgroundColor({ color: "#e74c3c", tabId: tabId });
    } else if (data.verdict === "SUSPICIOUS") {
      chrome.action.setBadgeText({ text: "?", tabId: tabId });
      chrome.action.setBadgeBackgroundColor({ color: "#f39c12", tabId: tabId });
    } else {
      chrome.action.setBadgeText({ text: "✓", tabId: tabId });
      chrome.action.setBadgeBackgroundColor({ color: "#2ecc71", tabId: tabId });
    }
  } catch (err) {
    console.debug("PhishBP Server offline or query skipped:", err);
  }
}
