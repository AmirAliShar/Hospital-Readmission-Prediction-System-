const API_BASE = "http://localhost:8000/api";

export const api = {
  async getHealth() {
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (!res.ok) throw new Error("Health check failed");
      return await res.json();
    } catch (err) {
      console.warn("Backend offline or unreachable:", err);
      return { status: "offline", model_loaded: false };
    }
  },

  async getSchema() {
    try {
      const res = await fetch(`${API_BASE}/predict/schema`);
      if (!res.ok) throw new Error("Failed to fetch schema");
      return await res.json();
    } catch (err) {
      console.error(err);
      return null;
    }
  },

  async predict(patientData) {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patientData),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Prediction request failed");
    }
    return await res.json();
  },

  async batchPredict(patients) {
    const res = await fetch(`${API_BASE}/predict/batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ patients }),
    });
    if (!res.ok) throw new Error("Batch prediction failed");
    return await res.json();
  },

  async getEDASummary() {
    const res = await fetch(`${API_BASE}/eda/summary`);
    if (!res.ok) throw new Error("Failed to fetch EDA summary");
    return await res.json();
  },

  async getCharts() {
    const res = await fetch(`${API_BASE}/eda/charts`);
    if (!res.ok) throw new Error("Failed to fetch charts list");
    return await res.json();
  },

  async getModelInfo() {
    const res = await fetch(`${API_BASE}/model/info`);
    if (!res.ok) throw new Error("Failed to fetch model info");
    return await res.json();
  },

  async getModelMetrics() {
    const res = await fetch(`${API_BASE}/model/metrics`);
    if (!res.ok) throw new Error("Failed to fetch model metrics");
    return await res.json();
  },

  async getModelComparison() {
    const res = await fetch(`${API_BASE}/model/comparison`);
    if (!res.ok) throw new Error("Failed to fetch model comparison");
    return await res.json();
  },

  async getFeatureImportance() {
    const res = await fetch(`${API_BASE}/model/features`);
    if (!res.ok) throw new Error("Failed to fetch feature importance");
    return await res.json();
  }
};
