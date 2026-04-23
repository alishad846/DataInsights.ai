import path from "path";
import fs from "fs/promises";
import { createReadStream } from "fs";
import { pool } from "../config/db.js";
import { validateDatasetAccess, getDatasetPaths } from "../utils/accessUtils.js";

// ─────────────────────────────────────────────
// CSV Parser (handles quoted commas properly)
// ─────────────────────────────────────────────
const parseCSV = (csvText) => {
  const lines = csvText.trim().split("\n");
  if (lines.length < 2) return { headers: [], rows: [] };

  const headers = lines[0].split(",").map((h) => h.trim().replace(/^"|"$/g, ""));
  const rows = lines.slice(1).map((line) => {
    const values = [];
    let current = "";
    let inQuotes = false;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (ch === '"') {
        inQuotes = !inQuotes;
      } else if (ch === "," && !inQuotes) {
        values.push(current.trim());
        current = "";
      } else {
        current += ch;
      }
    }
    values.push(current.trim());
    const obj = {};
    headers.forEach((h, idx) => {
      obj[h] = values[idx] !== undefined ? values[idx] : "";
    });
    return obj;
  });
  return { headers, rows };
};

// ─────────────────────────────────────────────
// Column Type + Stats Detection
// ─────────────────────────────────────────────
const detectColumnType = (rows, col) => {
  const sample = Math.min(rows.length, 200);
  let numCount = 0;
  for (let i = 0; i < sample; i++) {
    const val = rows[i][col];
    if (val === "" || val == null) continue;
    if (!isNaN(parseFloat(val)) && isFinite(val)) numCount++;
  }
  return numCount > sample * 0.6 ? "numeric" : "categorical";
};

const buildColumnStats = (rows, headers, columnTypes) => {
  const columnStats = {};
  headers.forEach((col) => {
    const type = columnTypes[col];
    const values = rows.map((r) => r[col]).filter((v) => v !== "" && v !== null && v !== undefined);
    if (type === "numeric") {
      const vals = values.map((v) => parseFloat(v)).filter((v) => !isNaN(v));
      vals.sort((a, b) => a - b);
      columnStats[col] = {
        min: vals.length ? vals[0] : 0,
        max: vals.length ? vals[vals.length - 1] : 0,
        mean: vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0,
        count: vals.length,
        totalCount: values.length,
      };
    } else {
      const counts = new Map();
      values.forEach((value) => {
        const key = String(value);
        counts.set(key, (counts.get(key) || 0) + 1);
      });
      const topValues = [...counts.entries()]
        .sort((a, b) => b[1] - a[1])
        .slice(0, 50)
        .map(([value, count]) => ({
          value,
          count,
          pct: values.length ? (count / values.length) * 100 : 0,
        }));
      columnStats[col] = {
        uniqueCount: counts.size,
        values: [...counts.keys()].slice(0, 50),
        topValues,
        totalCount: values.length,
      };
    }
  });
  return columnStats;
};

// ─────────────────────────────────────────────
// Apply Filters + Search to Rows
// ─────────────────────────────────────────────
const applyFiltersAndSearch = (rows, headers, columnTypes, filters, search) => {
  let filtered = rows;

  Object.entries(filters).forEach(([col, filterVal]) => {
    if (!headers.includes(col)) return;
    const type = columnTypes[col];
    if (type === "categorical") {
      if (Array.isArray(filterVal) && filterVal.length > 0) {
        const set = new Set(filterVal.map(String));
        filtered = filtered.filter((r) => set.has(String(r[col])));
      }
    } else if (type === "numeric") {
      if (filterVal.min !== undefined)
        filtered = filtered.filter((r) => parseFloat(r[col]) >= parseFloat(filterVal.min));
      if (filterVal.max !== undefined)
        filtered = filtered.filter((r) => parseFloat(r[col]) <= parseFloat(filterVal.max));
    }
  });

  if (search) {
    const s = search.toLowerCase();
    filtered = filtered.filter((r) => headers.some((h) => String(r[h]).toLowerCase().includes(s)));
  }

  return filtered;
};

const parsePaginationLimit = (rawLimit, defaultLimit = 500) => {
  if (rawLimit === undefined || rawLimit === null || rawLimit === "") {
    return { mode: "paged", limit: defaultLimit };
  }

  if (String(rawLimit).toLowerCase() === "all") {
    return { mode: "all", limit: null };
  }

  const parsed = parseInt(rawLimit, 10);
  if (Number.isNaN(parsed)) {
    return { mode: "paged", limit: defaultLimit };
  }

  if (parsed <= 0) {
    return { mode: "all", limit: null };
  }

  return { mode: "paged", limit: parsed };
};

// ─────────────────────────────────────────────
// Helper: Resolve which file to serve
// ─────────────────────────────────────────────
const resolveDatasetFile = async (datasetId, userEmail) => {
  // Look up the file_name from DB
  const dsResult = await pool.query(
    "SELECT file_name, upload_status FROM datasets WHERE dataset_id = $1",
    [datasetId]
  );
  if (dsResult.rows.length === 0) return null;
  const { file_name, upload_status } = dsResult.rows[0];
  
  const userResult = await pool.query("SELECT full_name FROM users WHERE email = $1", [userEmail]);
  const fullName = userResult.rows[0]?.full_name;
  const paths = getDatasetPaths(datasetId, file_name, fullName);

  const fs = (await import("fs/promises")).default;
  const path = await import("path");
  const tempDir = path.resolve(process.cwd(), "..", "uploads", "temp");

  // If the dataset is already finalized, the cleaned file is canonical.
  if (upload_status === "cleaned" || upload_status === "completed") {
    try {
      await fs.access(paths.cleaned);
      return paths.cleaned;
    } catch {}
  }

  // Priority during active cleaning: newest temp for this dataset/user > cleaned > raw
  try {
    const files = await fs.readdir(tempDir).catch(() => []);
    const dsTempFiles = files.filter(f => f.startsWith(datasetId) && f.endsWith(".csv"));
    if (dsTempFiles.length > 0) {
      const userSafeName = fullName ? fullName.replace(/[^a-zA-Z0-9]/g, "_").toLowerCase() : "unknown";
      const tempCandidates = await Promise.all(
        dsTempFiles.map(async (file) => {
          const fullPath = path.join(tempDir, file);
          try {
            const stat = await fs.stat(fullPath);
            return { file, fullPath, mtimeMs: stat.mtimeMs };
          } catch {
            return null;
          }
        })
      );

      const availableCandidates = tempCandidates.filter(Boolean);
      const userMatches = availableCandidates.filter((entry) => entry.file.includes(userSafeName));
      const preferredCandidates = (userMatches.length > 0 ? userMatches : availableCandidates)
        .sort((a, b) => b.mtimeMs - a.mtimeMs);
      if (preferredCandidates[0]?.fullPath) {
        return preferredCandidates[0].fullPath;
      }
    }

    // Check Cleaned
    try {
      await fs.access(paths.cleaned);
      return paths.cleaned;
    } catch {
      // Check Raw
      try {
        await fs.access(paths.raw);
        return paths.raw;
      } catch {}
    }
  } catch {}

  return null;
};

// ─────────────────────────────────────────────
// GET /api/cleaned-data/:id
// ─────────────────────────────────────────────
export const getCleanedData = async (req, res) => {
  try {
    const datasetId = req.params.id;
    const userEmail = req.user?.email;

    if (!userEmail) return res.status(401).json({ success: false, message: "Authentication required" });
    if (!datasetId) return res.status(400).json({ success: false, message: "Dataset ID required" });

    // Access control
    const userResult = await pool.query("SELECT user_id, role FROM users WHERE email = $1", [userEmail]);
    const user = userResult.rows[0];
    if (!user) return res.status(401).json({ success: false, message: "User not found" });
    if (!(await validateDatasetAccess(user.user_id, datasetId, user.role))) {
      return res.status(403).json({ success: false, message: "Access denied" });
    }

    // Resolve file path
    const filePath = await resolveDatasetFile(datasetId, userEmail);
    if (!filePath) {
      return res.status(404).json({
        success: false,
        message: "Cleaned data not found. Please complete the cleaning process first.",
      });
    }

    console.log(`[CLEANED-DATA] Serving: ${filePath}`);
    const csvText = await fs.readFile(filePath, "utf-8");
    const { headers, rows } = parseCSV(csvText);

    if (headers.length === 0) {
      return res.status(400).json({ success: false, message: "Empty or invalid CSV file" });
    }

    // Build type + stats on full data
    const columnTypes = {};
    headers.forEach((col) => { columnTypes[col] = detectColumnType(rows, col); });
    const columnStats = buildColumnStats(rows, headers, columnTypes);

    // Filters + search
    const filters = req.query.filters ? JSON.parse(req.query.filters) : {};
    const search = req.query.search || "";
    const filteredRows = applyFiltersAndSearch(rows, headers, columnTypes, filters, search);

    // Pagination
    const totalRows = filteredRows.length;
    const { mode, limit } = parsePaginationLimit(req.query.limit, 500);
    const page = mode === "all" ? 1 : (parseInt(req.query.page, 10) || 1);
    const paginatedRows = mode === "all"
      ? filteredRows
      : filteredRows.slice((page - 1) * limit, page * limit);
    const responseLimit = mode === "all" ? totalRows : limit;
    const totalPages = mode === "all" ? 1 : Math.ceil(totalRows / limit);

    return res.json({
      success: true,
      headers: headers.filter((h) => h !== "Unnamed: 0.1" && h !== "Unnamed: 0"),
      columnTypes,
      columnStats,
      rows: paginatedRows,
      totalRows,
      page,
      limit: responseLimit,
      totalPages,
    });
  } catch (err) {
    console.error("[CLEANED-DATA ERROR]", err);
    return res.status(500).json({ success: false, message: "Internal server error" });
  }
};

// ─────────────────────────────────────────────
// GET /api/original-data/:id
// ─────────────────────────────────────────────
export const getOriginalData = async (req, res) => {
  try {
    const datasetId = req.params.id;
    const userEmail = req.user?.email;

    if (!userEmail) return res.status(401).json({ success: false, message: "Authentication required" });
    if (!datasetId) return res.status(400).json({ success: false, message: "Dataset ID required" });

    const userResult = await pool.query("SELECT user_id, role FROM users WHERE email = $1", [userEmail]);
    const user = userResult.rows[0];
    if (!user) return res.status(401).json({ success: false, message: "User not found" });
    if (!(await validateDatasetAccess(user.user_id, datasetId, user.role))) {
      return res.status(403).json({ success: false, message: "Access denied" });
    }

    const dsResult = await pool.query(
      "SELECT file_name FROM datasets WHERE dataset_id = $1",
      [datasetId]
    );
    if (dsResult.rows.length === 0) return res.status(404).json({ success: false, message: "Dataset not found" });

    const paths = getDatasetPaths(datasetId, dsResult.rows[0].file_name);

    let filePath = null;
    for (const candidate of [paths.raw]) {
      try { await fs.access(candidate); filePath = candidate; break; } catch {}
    }

    if (!filePath) return res.status(404).json({ success: false, message: "Original data not found" });

    const csvText = await fs.readFile(filePath, "utf-8");
    const { headers, rows } = parseCSV(csvText);

    if (headers.length === 0) return res.status(400).json({ success: false, message: "Empty CSV file" });

    const columnTypes = {};
    headers.forEach((col) => { columnTypes[col] = detectColumnType(rows, col); });
    const columnStats = buildColumnStats(rows, headers, columnTypes);

    const filters = req.query.filters ? JSON.parse(req.query.filters) : {};
    const search = req.query.search || "";
    const filteredRows = applyFiltersAndSearch(rows, headers, columnTypes, filters, search);

    const totalRows = filteredRows.length;
    const { mode, limit } = parsePaginationLimit(req.query.limit, 500);
    const page = mode === "all" ? 1 : (parseInt(req.query.page, 10) || 1);
    const paginatedRows = mode === "all"
      ? filteredRows
      : filteredRows.slice((page - 1) * limit, page * limit);
    const responseLimit = mode === "all" ? totalRows : limit;
    const totalPages = mode === "all" ? 1 : Math.ceil(totalRows / limit);

    return res.json({
      success: true,
      headers,
      columnTypes,
      columnStats,
      rows: paginatedRows,
      totalRows,
      page,
      limit: responseLimit,
      totalPages,
    });
  } catch (err) {
    console.error("[ORIGINAL-DATA ERROR]", err);
    return res.status(500).json({ success: false, message: "Internal server error" });
  }
};
