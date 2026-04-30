const API_BASE = "/api";

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || res.statusText);
  }
  return res.json();
}

export function fetchPatients(filters = {}) {
  const params = new URLSearchParams();
  if (filters.site) params.set("site", filters.site);
  if (filters.riskMin != null) params.set("riskMin", filters.riskMin);
  if (filters.riskMax != null) params.set("riskMax", filters.riskMax);
  const qs = params.toString();
  return request(`/patients${qs ? `?${qs}` : ""}`);
}

export function fetchPatient(id) {
  return request(`/patients/${id}`);
}

export function fetchStats() {
  return request("/stats");
}

export function fetchGlobalShap() {
  return request("/shap/global");
}

export function fetchPatientShap(id) {
  return request(`/shap/patient/${id}`);
}

export function predictWhatIf(features) {
  return request("/predict", {
    method: "POST",
    body: JSON.stringify({ features }),
  });
}

export function fetchNarrative(patientId) {
  return request("/narrative", {
    method: "POST",
    body: JSON.stringify({ patientId }),
  });
}

export function sendChat(message, patientId = null, history = [], language = 'en') {
  return request("/chat", {
    method: "POST",
    body: JSON.stringify({ message, patientId, history, language }),
  });
}
