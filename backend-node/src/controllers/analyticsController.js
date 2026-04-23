import path from "path";
import fs from "fs/promises";
import { pool } from "../config/db.js";
import { validateDatasetAccess, getDatasetPaths } from "../utils/accessUtils.js";

// ─────────────────────────────────────────────────────────────────────────────
// Lightweight CSV parser (header row + rows)
// ─────────────────────────────────────────────────────────────────────────────
const quickParseCSV = (text, maxRows = 500) => {
  const lines = text.trim().split("\n").filter(Boolean);
  if (lines.length < 2) return { headers: [], rows: [] };
  const headers = lines[0].split(",").map((h) => h.trim().replace(/^"|"$/g, ""));
  const rows = lines.slice(1, maxRows + 1).map((line) => {
    const cells = line.split(",").map((c) => c.trim().replace(/^"|"$/g, ""));
    const obj = {};
    headers.forEach((h, i) => { obj[h] = cells[i] ?? ""; });
    return obj;
  });
  return { headers, rows };
};

// ─────────────────────────────────────────────────────────────────────────────
// Detect column type from a sample of values
// ─────────────────────────────────────────────────────────────────────────────
const detectType = (values) => {
  const sample = values.filter(Boolean).slice(0, 100);
  if (sample.length === 0) return "text";
  const numericCount = sample.filter((v) => !isNaN(parseFloat(v)) && isFinite(v)).length;
  if (numericCount > sample.length * 0.7) return "numeric";
  const dateCount = sample.filter((v) => !isNaN(Date.parse(v)) && v.length > 6).length;
  if (dateCount > sample.length * 0.5) return "datetime";
  return "categorical";
};

// ─────────────────────────────────────────────────────────────────────────────
// Helper: get file path to use for analysis (prefer cleaned → working → raw)
// ─────────────────────────────────────────────────────────────────────────────
const resolveCleanedPath = async (datasetId, fileName) => {
  const paths = getDatasetPaths(datasetId, fileName);
  for (const candidate of [paths.cleaned, paths.working, paths.raw]) {
    try { await fs.access(candidate); return candidate; } catch {}
  }
  return null;
};

// ─────────────────────────────────────────────────────────────────────────────
// GET /api/analytics?datasetId=... — summary insights
// Previously read from ml_engine/data/users/.../insights.json — dead path.
// Now computes dynamically from the actual cleaned CSV + schema_json stats.
// ─────────────────────────────────────────────────────────────────────────────
export const analyticsSummary = async (req, res) => {
  const datasetId = req.query.datasetId || req.body.datasetId;
  if (!datasetId) return res.status(400).json({ error: "datasetId is required" });

  const userEmail = req.user?.email;
  if (!userEmail) return res.status(401).json({ success: false, message: "Authentication required" });

  try {
    // Access control
    const userResult = await pool.query("SELECT user_id, role FROM users WHERE email = $1", [userEmail]);
    const user = userResult.rows[0];
    if (!user) return res.status(401).json({ success: false, message: "User not found" });
    if (!(await validateDatasetAccess(user.user_id, datasetId, user.role))) {
      return res.status(403).json({ success: false, message: "Access denied" });
    }

    // Fetch dataset metadata from DB
    const dsResult = await pool.query(
      "SELECT dataset_name, file_name, schema_json FROM datasets WHERE dataset_id = $1",
      [datasetId]
    );
    if (dsResult.rows.length === 0) return res.status(404).json({ error: "Dataset not found" });
    const { dataset_name, file_name, schema_json } = dsResult.rows[0];

    // Prefer cached raw stats from schema_json (written by getRawStats helper during cleaning)
    const rawStats = schema_json?.rawStats || schema_json?.raw_stats || null;

    // Load cleaned file to compute column-level stats
    const filePath = await resolveCleanedPath(datasetId, file_name);
    let insights = [];

    if (filePath) {
      const csvText = await fs.readFile(filePath, "utf-8");
      const { headers, rows } = quickParseCSV(csvText, 1000);

      headers.forEach((col) => {
        const values = rows.map((r) => r[col]).filter(Boolean);
        const type = detectType(values);
        if (type === "numeric") {
          const nums = values.map(parseFloat).filter((n) => !isNaN(n));
          if (nums.length > 0) {
            const sum = nums.reduce((a, b) => a + b, 0);
            const avg = sum / nums.length;
            const max = Math.max(...nums);
            const min = Math.min(...nums);
            insights.push({
              key: col,
              title: col,
              value: avg.toLocaleString(undefined, { maximumFractionDigits: 2 }),
              description: `Avg across ${nums.length} records. Range: ${min.toLocaleString()} – ${max.toLocaleString()}`,
              type: "numeric",
              sum: Math.round(sum * 100) / 100,
              avg: Math.round(avg * 100) / 100,
              max, min, count: nums.length
            });
          }
        } else if (type === "categorical") {
          const uniqueVals = [...new Set(values)];
          insights.push({
            key: col,
            title: col,
            value: uniqueVals.length,
            description: `${uniqueVals.length} unique categories`,
            type: "categorical",
            topValues: uniqueVals.slice(0, 5)
          });
        }
      });
    }

    res.json({
      success: true,
      dataset_name,
      insights,
      rawStats: rawStats || null
    });
  } catch (err) {
    console.error("[analyticsController.analyticsSummary]", err);
    res.status(500).json({ error: "Failed to generate analytics summary", details: err.message });
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// POST /api/analytics/chart — forecast/chart data
// Previously read from ml_engine/data/users/.../forecast.json — dead path.
// Now computes a simple grouped time-series from cleaned data if possible.
// ─────────────────────────────────────────────────────────────────────────────
export const analyticsChart = async (req, res) => {
  const { datasetId } = req.body;
  if (!datasetId) return res.status(400).json({ error: "datasetId is required" });

  const userEmail = req.user?.email;
  if (!userEmail) return res.status(401).json({ success: false, message: "Authentication required" });

  try {
    const userResult = await pool.query("SELECT user_id, role FROM users WHERE email = $1", [userEmail]);
    const user = userResult.rows[0];
    if (!user) return res.status(401).json({ success: false, message: "User not found" });
    if (!(await validateDatasetAccess(user.user_id, datasetId, user.role))) {
      return res.status(403).json({ success: false, message: "Access denied" });
    }

    const dsResult = await pool.query("SELECT file_name, dataset_name FROM datasets WHERE dataset_id = $1", [datasetId]);
    if (dsResult.rows.length === 0) return res.status(404).json({ error: "Dataset not found" });
    const { file_name, dataset_name } = dsResult.rows[0];

    const filePath = await resolveCleanedPath(datasetId, file_name);
    if (!filePath) return res.status(404).json({ error: "Cleaned data not found" });

    const csvText = await fs.readFile(filePath, "utf-8");
    const { headers, rows } = quickParseCSV(csvText, 2000);

    // Find first numeric column for charting
    const numericCol = headers.find((h) => {
      const vals = rows.map((r) => r[h]).filter(Boolean);
      const numCount = vals.filter((v) => !isNaN(parseFloat(v))).length;
      return numCount > vals.length * 0.7;
    });

    if (!numericCol) return res.json({ success: true, chart_data: [] });

    // Sample every Nth row to generate a trend chart (max 100 points)
    const step = Math.max(1, Math.floor(rows.length / 100));
    const chartData = rows.filter((_, i) => i % step === 0).map((row, i) => ({
      index: i + 1,
      value: parseFloat(row[numericCol]) || 0
    }));

    res.json({ success: true, column: numericCol, dataset_name, chart_data: chartData });
  } catch (err) {
    console.error("[analyticsController.analyticsChart]", err);
    res.status(500).json({ error: "Failed to generate chart data", details: err.message });
  }
};
