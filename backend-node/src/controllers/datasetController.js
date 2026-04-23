import path from "path";
import fs from "fs/promises";
import { createReadStream } from "fs";
import { parse } from "csv-parse";
import XLSX from "xlsx";
import { pool } from "../config/db.js";
import { validateDatasetAccess, getDatasetPaths } from "../utils/accessUtils.js";
import { logCleaningActivity } from "./activityController.js";


/** Map schema.txt columns to fields the React app already expects */
export function mapDatasetRow(row) {
  if (!row) return row;
  const schema = row.schema_json && typeof row.schema_json === "object" ? row.schema_json : {};
  return {
    ...row,
    name: row.dataset_name,
    filename: row.file_name,
    status: row.upload_status,
    size: row.file_size,
    uploaded_by: row.uploaded_by_name || row.uploaded_by_email || row.uploaded_by,
    rows_count: schema.rows_count ?? schema.total_rows ?? null,
    columns_count: schema.columns_count ?? schema.total_columns ?? null,
    version: 1,
    has_access: row.has_access ?? true, // Default to true if not specified (e.g. for admins)
  };
}

// Removed checkAccess local function, using validateDatasetAccess from utils

export const getAllDatasets = async (req, res) => {
  const userId = req.user?.email;
  if (!userId) {
    return res.status(401).json({ success: false, message: "Authentication required" });
  }

  try {
    const userRes = await pool.query("SELECT user_id, role, company_id FROM users WHERE email = $1", [userId]);
    if (userRes.rows.length === 0) {
      return res.status(404).json({ success: false, message: "User not found" });
    }

    const user = userRes.rows[0];
    const companyId = user.company_id;
    const isEmployee = user.role === 'employee';

    let query = `
      SELECT d.*, u.full_name AS uploaded_by_name, u.email AS uploaded_by_email
    `;
    const params = [companyId];

    if (isEmployee) {
      query += `, TRUE AS has_access `;
      query += ` FROM datasets d
                 LEFT JOIN users u ON d.uploaded_by = u.user_id
                 INNER JOIN permissions p ON d.dataset_id = p.dataset_id
                 WHERE d.company_id = $1 AND p.user_id = $2 AND p.can_view = TRUE AND d.upload_status != 'failed' `;
      params.push(user.user_id);

    } else {
      query += ` FROM datasets d
                 LEFT JOIN users u ON d.uploaded_by = u.user_id
                 WHERE d.company_id = $1 `;
    }

    query += ` ORDER BY d.created_at DESC `;

    const result = await pool.query(query, params);

    return res.json({
      success: true,
      count: result.rows.length,
      data: result.rows.map(row => mapDatasetRow({
        ...row,
        has_access: isEmployee ? row.has_access : true
      })),
    });
  } catch (err) {
    console.error("getAllDatasets error:", err);
    return res.status(500).json({ success: false, message: "Server error" });
  }
};

export const getAllDatasetsAdmin = async (req, res) => {
  try {
    const result = await pool.query(
      `SELECT d.*, u.full_name AS uploaded_by_name, u.email AS uploaded_by_email, c.company_name
       FROM datasets d
       LEFT JOIN users u ON d.uploaded_by = u.user_id
       LEFT JOIN companies c ON d.company_id = c.company_id
       WHERE d.upload_status != 'failed'
       ORDER BY d.created_at DESC`
    );
    return res.json({
      success: true,
      count: result.rows.length,
      data: result.rows.map(mapDatasetRow),
    });
  } catch (err) {
    console.error("getAllDatasetsAdmin error:", err);
    return res.status(500).json({ success: false, message: "Server error" });
  }
};

export const getDatasetById = async (req, res) => {
  try {
    const result = await pool.query(
      `SELECT d.*, u.full_name AS uploaded_by_name, u.email AS uploaded_by_email
       FROM datasets d
       LEFT JOIN users u ON d.uploaded_by = u.user_id
       WHERE d.dataset_id = $1`,
      [req.params.id]
    );
    if (result.rows.length === 0) return res.status(404).json({ success: false, message: "Dataset not found" });

    // Check access
    const userRole = req.user?.role;
    const userId = (await pool.query("SELECT user_id FROM users WHERE email = $1", [req.user?.email])).rows[0]?.user_id;

    const hasAccess = await validateDatasetAccess(userId, req.params.id, userRole);
    if (!hasAccess) {
      return res.status(403).json({ success: false, message: "Unauthorized: Access restricted by administrator" });
    }

    return res.json({ success: true, data: mapDatasetRow({ ...result.rows[0], has_access: true }) });
  } catch (err) {
    console.error("getDatasetById error:", err);
    return res.status(500).json({ message: "Server error" });
  }
};

export const getDatasetStatus = async (req, res) => {
  const datasetId = req.params.id;
  const userEmail = req.user?.email;

  try {
    const userRes = await pool.query("SELECT user_id, role, full_name FROM users WHERE email = $1", [userEmail]);
    const user = userRes.rows[0];
    if (!user) return res.status(404).json({ success: false, message: "User not found" });

    const datasetRes = await pool.query("SELECT * FROM datasets WHERE dataset_id = $1", [datasetId]);
    if (datasetRes.rows.length === 0) return res.status(404).json({ success: false, message: "Dataset not found" });

    const dataset = datasetRes.rows[0];
    if (!(await validateDatasetAccess(user.user_id, datasetId, user.role))) {
      return res.status(403).json({ success: false, message: "Access denied" });
    }

    return res.json({ success: true, status: dataset.upload_status });
  } catch (err) {
    console.error("getDatasetStatus error:", err);
    return res.status(500).json({ message: "Server error" });
  }
};

export const assignDataset = async (req, res) => {
  const datasetId = req.params.id;
  const { userIds } = req.body; // Array of UUIDs
  const adminEmail = req.user?.email;

  if (!Array.isArray(userIds) || userIds.length === 0) {
    return res.status(400).json({ success: false, message: "userIds array is required" });
  }

  try {
    const adminResult = await pool.query("SELECT user_id, company_id FROM users WHERE email = $1", [adminEmail]);
    const admin = adminResult.rows[0];
    if (!admin) {
      return res.status(401).json({ success: false, message: "Administrator not found" });
    }

    // Bulk assign
    for (const uid of userIds) {
      await pool.query(
        `INSERT INTO permissions (company_id, user_id, dataset_id, can_view, granted_by)
         VALUES ($1, $2, $3, TRUE, $4)
         ON CONFLICT (user_id, dataset_id) 
         DO UPDATE SET can_view = TRUE, updated_at = NOW()`,
        [admin.company_id, uid, datasetId, admin.user_id]
      );
    }

    res.json({ success: true, message: `Successfully assigned ${userIds.length} users` });
  } catch (err) {
    console.error("assignDataset error:", err);
    res.status(500).json({ success: false, message: "Server error" });
  }
};

export const unassignDataset = async (req, res) => {
  const datasetId = req.params.id;
  const { userId } = req.body;

  if (!userId) return res.status(400).json({ success: false, message: "userId is required" });

  try {
    // Requirements: set can_view = false instead of deleting
    await pool.query(
      "UPDATE permissions SET can_view = FALSE, updated_at = NOW() WHERE user_id = $1 AND dataset_id = $2",
      [userId, datasetId]
    );
    res.json({ success: true, message: "User access revoked successfully" });
  } catch (err) {
    console.error("unassignDataset error:", err);
    res.status(500).json({ success: false, message: "Server error" });
  }
};

/**
 * Efficiently count lines in a CSV file (total rows)
 */
const countLines = async (filePath) => {
  const { createReadStream } = await import("fs");
  return new Promise((resolve) => {
    let count = 0;
    createReadStream(filePath)
      .on('data', (chunk) => {
        for (let i = 0; i < chunk.length; ++i) if (chunk[i] === 10) ++count;
      })
      .on('end', () => resolve(count))
      .on('error', () => resolve(0));
  });
};

const pathExists = async (filePath) => {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
};

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const waitForPath = async (filePath, timeoutMs = 2500) => {
  const startedAt = Date.now();
  while (Date.now() - startedAt < timeoutMs) {
    if (await pathExists(filePath)) {
      return true;
    }
    await sleep(150);
  }
  return pathExists(filePath);
};

const copyFileWithFallback = async (sourcePath, destPath) => {
  let lastError;

  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      await fs.copyFile(sourcePath, destPath);
      return;
    } catch (error) {
      lastError = error;

      // If another request is already materializing the temp file, reuse it.
      if ((error.code === "EPERM" || error.code === "EBUSY") && await waitForPath(destPath, 1000)) {
        return;
      }

      try {
        const buffer = await fs.readFile(sourcePath);
        await fs.writeFile(destPath, buffer);
        return;
      } catch (fallbackError) {
        lastError = fallbackError;
      }

      if (attempt < 2 && (lastError.code === "EPERM" || lastError.code === "EBUSY")) {
        await sleep(250 * (attempt + 1));
        continue;
      }

      throw lastError;
    }
  }

  throw lastError;
};

const resolveDatasetFilePath = async (datasetId, fileName, fullName, options = {}) => {
  const { preferCleaned = false } = options;
  const paths = getDatasetPaths(datasetId, fileName, fullName);
  const tempDir = path.resolve(process.cwd(), "..", "uploads", "temp");
  const safeFullName = fullName ? fullName.replace(/[^a-zA-Z0-9]/g, "_").toLowerCase() : "";

  if (preferCleaned && await pathExists(paths.cleaned)) {
    return { paths, targetPath: paths.cleaned };
  }

  try {
    const files = await fs.readdir(tempDir);
    const dsTempFiles = files.filter((file) => file.startsWith(datasetId) && file.endsWith(".csv"));
    if (dsTempFiles.length > 0) {
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
      const userMatches = safeFullName
        ? availableCandidates.filter((entry) => entry.file.includes(safeFullName))
        : availableCandidates;
      const preferredCandidates = userMatches.length > 0 ? userMatches : availableCandidates;

      preferredCandidates.sort((a, b) => b.mtimeMs - a.mtimeMs);
      const preferredTempPath = preferredCandidates[0]?.fullPath;
      if (preferredTempPath && await pathExists(preferredTempPath)) {
        return { paths, targetPath: preferredTempPath };
      }
    }
  } catch {}

  if (await pathExists(paths.cleaned)) {
    return { paths, targetPath: paths.cleaned };
  }

  if (await pathExists(paths.raw)) {
    return { paths, targetPath: paths.raw };
  }

  return { paths, targetPath: null };
};

/**
 * Get statistics for the raw dataset by calling the Python transformer
 */
const getRawStats = async (rawPath) => {
  const { spawn } = await import("child_process");
  const path = await import("path");
  const transformerScript = path.resolve(process.cwd(), "..", "ml_engine", "pipeline", "transformer.py");

  return new Promise((resolve) => {
    const pythonProcess = spawn("python", [
      transformerScript,
      "--input", rawPath,
      "--output", rawPath + ".stats", // Dummy output for stats mode
      "--config", JSON.stringify({ type: "get_stats" })
    ]);

    let stdout = "";
    pythonProcess.stdout.on("data", (data) => (stdout += data.toString()));
    pythonProcess.on("close", (code) => {
      if (code === 0) {
        try {
          const result = JSON.parse(stdout.trim());
          resolve(result);
        } catch { resolve(null); }
      } else { resolve(null); }
    });
  });
};

/**
 * Get the first line (headers) of a CSV file
 */
const getCSVHeaders = async (filePath) => {
  const { createReadStream } = await import("fs");
  const { parse } = await import("csv-parse");
  return new Promise((resolve) => {
    let settled = false;
    const finish = (headers = []) => {
      if (!settled) {
        settled = true;
        resolve(headers);
      }
    };

    const stream = createReadStream(filePath);
    stream.on("error", () => finish([]));

    const parser = stream.pipe(parse({ to_line: 1 }));
    parser.on("error", () => finish([]));

    (async () => {
      try {
        for await (const row of parser) {
          finish(row);
          return;
        }
        finish([]);
      } catch {
        finish([]);
      }
    })();
  });
};

/**
 * Preview rows of a dataset (Paginated)
 * GET /api/datasets/:id/preview?page=1&pageSize=50
 */
export const getDatasetPreview = async (req, res) => {
  const datasetId = req.params.id;
  const userEmail = req.user?.email;
  const page = parseInt(req.query.page) || 1;
  const pageSize = parseInt(req.query.pageSize) || 50;

  try {
    const userResult = await pool.query("SELECT user_id, role FROM users WHERE email = $1", [userEmail]);
    const user = userResult.rows[0];
    if (!user) return res.status(401).json({ success: false, message: "User not found" });

    if (!(await validateDatasetAccess(user.user_id, datasetId, user.role))) {
      return res.status(403).json({ success: false, message: "Unauthorized" });
    }

    const dsResult = await pool.query("SELECT file_name, upload_status, schema_json FROM datasets WHERE dataset_id = $1", [datasetId]);
    if (dsResult.rows.length === 0) {
      return res.status(404).json({ success: false, message: "Dataset not found" });
    }
    const { file_name, upload_status, schema_json } = dsResult.rows[0];

    const fullName = await pool.query("SELECT full_name FROM users WHERE email = $1", [userEmail]).then(r => r.rows[0]?.full_name);
    const { paths, targetPath } = await resolveDatasetFilePath(datasetId, file_name, fullName, {
      preferCleaned: upload_status === "cleaned" || upload_status === "completed",
    });

    if (!targetPath) {
      return res.status(404).json({
        success: false,
        message: "Dataset file not found on disk. Re-upload or reprocess this dataset.",
      });
    }

    const totalRows = await countLines(targetPath);
    const headers = await getCSVHeaders(targetPath);
    if (headers.length === 0) {
      return res.status(404).json({
        success: false,
        message: "Dataset file is unavailable or unreadable.",
      });
    }
    
    const { createReadStream } = await import("fs");
    const { parse } = await import("csv-parse");
    const rows = [];
    
    // from_line 1 is the header. We want to start data from line 2.
    // Page 1: from_line = 2, to_line = 51
    // Page 2: from_line = 52, to_line = 101
    const fromLine = (page - 1) * pageSize + 2;
    const toLine = page * pageSize + 1;

    const parser = createReadStream(targetPath).pipe(
      parse({ 
        columns: headers, 
        trim: true, 
        skip_empty_lines: true, 
        from_line: fromLine, 
        to_line: toLine 
      })
    );

    for await (const record of parser) {
      rows.push(record);
    }

    // Handle Raw Stats (Baseline for right panel)
    let rawStats = (schema_json && schema_json.raw_stats) ? schema_json.raw_stats : null;
    if (!rawStats && await pathExists(paths.raw)) {
      const statsResult = await getRawStats(paths.raw);
      if (statsResult && statsResult.status === "success") {
        rawStats = {
          totalRows: statsResult.total_rows,
          totalNulls: statsResult.total_nulls,
          totalDuplicates: statsResult.total_duplicates,
          columnNulls: statsResult.column_nulls
        };
        // Cache in DB
        const newSchema = { ...(schema_json || {}), raw_stats: rawStats };
        await pool.query("UPDATE datasets SET schema_json = $1 WHERE dataset_id = $2", [newSchema, datasetId]);
      }
    }

    // Current working stats reflect whichever file the preview is showing now.
    let currentStats = null;
    if (targetPath) {
      const statsResult = await getRawStats(targetPath);
      if (statsResult && statsResult.status === "success") {
        currentStats = {
          totalRows: statsResult.total_rows,
          totalNulls: statsResult.total_nulls,
          totalDuplicates: statsResult.total_duplicates,
          columnNulls: statsResult.column_nulls
        };
      }
    }

    return res.json({ 
      success: true, 
      data: rows, 
      totalRows: totalRows,
      currentPage: page,
      pageSize: pageSize,
      totalRowsPreviewed: rows.length,
      rawStats: rawStats,
      currentStats: currentStats
    });
  } catch (err) {
    console.error("getDatasetPreview error:", err);
    return res.status(500).json({ success: false, message: "Failed to load preview" });
  }
};

/**
 * Download the full dataset
 * GET /api/datasets/:id/download
 */
export const downloadDataset = async (req, res) => {
  const datasetId = req.params.id;
  const userEmail = req.user?.email;

  try {
    const userResult = await pool.query("SELECT user_id, role FROM users WHERE email = $1", [userEmail]);
    const user = userResult.rows[0];

    if (!user) return res.status(401).json({ success: false, message: "User not found" });

    // 1. Check Access
    const hasAccess = await validateDatasetAccess(user.user_id, datasetId, user.role);
    if (!hasAccess) {
      return res.status(403).json({ success: false, message: "Unauthorized: Access denied" });
    }

    // 2. Fetch record
    const dsResult = await pool.query("SELECT file_name, dataset_name FROM datasets WHERE dataset_id = $1", [datasetId]);
    if (dsResult.rows.length === 0) return res.status(404).json({ success: false, message: "Dataset not found" });

    const fileName = dsResult.rows[0].file_name;
    const originalName = dsResult.rows[0].dataset_name;
    const fullName = await pool.query("SELECT full_name FROM users WHERE email = $1", [userEmail]).then(r => r.rows[0]?.full_name);
    const statusResult = await pool.query("SELECT upload_status FROM datasets WHERE dataset_id = $1", [datasetId]);
    const currentStatus = statusResult.rows[0]?.upload_status;
    const { targetPath: filePath } = await resolveDatasetFilePath(datasetId, fileName, fullName, {
      preferCleaned: currentStatus === "cleaned" || currentStatus === "completed",
    });

    if (!filePath) {
      return res.status(404).json({
        success: false,
        message: "Dataset file not found on disk. Re-upload or reprocess this dataset.",
      });
    }

    // 3. Send file
    res.download(filePath, originalName, (err) => {
      if (err) {
        console.error("Download error:", err);
        if (!res.headersSent) {
          res.status(500).json({ success: false, message: "Failed to download file" });
        }
      }
    });

  } catch (err) {
    console.error("downloadDataset error:", err);
    res.status(500).json({ success: false, message: "Server error" });
  }
};

export const getDatasetAssignments = async (req, res) => {
  const datasetId = req.params.id;
  try {
    const result = await pool.query(
      `SELECT u.user_id, u.full_name, u.email, u.department, u.designation 
       FROM users u
       JOIN permissions p ON u.user_id = p.user_id
       WHERE p.dataset_id = $1 AND p.can_view = TRUE`,
      [datasetId]
    );
    res.json({ success: true, users: result.rows });
  } catch (err) {
    console.error("getDatasetAssignments error:", err);
    res.status(500).json({ success: false, message: "Server error" });
  }
};

export const updateDatasetStatus = async (req, res) => {
  const { status } = req.body;
  const allowedStatus = ["not_cleaned", "cleaning", "cleaned", "failed", "processing", "completed", "trained"]; // Keep legacy aliases during migration

  if (!allowedStatus.includes(status)) {
    return res.status(400).json({ success: false, message: `Invalid status value: ${status}` });
  }

  try {
    const userRole = req.user?.role;
    const userEmail = req.user?.email;
    const dbUserId = (await pool.query("SELECT user_id FROM users WHERE email = $1", [userEmail])).rows[0]?.user_id;

    if (!await validateDatasetAccess(dbUserId, req.params.id, userRole)) {
      return res.status(403).json({ success: false, message: "Access denied" });
    }

    const uploadStatus = status === "trained" ? "completed" : status;
    const result = await pool.query(
      "UPDATE datasets SET upload_status = $1, updated_at = NOW() WHERE dataset_id = $2 RETURNING *",
      [uploadStatus, req.params.id]
    );
    if (result.rows.length === 0) return res.status(404).json({ success: false, message: "Dataset not found" });
    return res.json({ success: true, message: "Dataset status updated" });
  } catch (err) {
    console.error("updateDatasetStatus error:", err);
    return res.status(500).json({ message: "Server error" });
  }
};

export const transformDataset = async (req, res) => {
  const datasetId = req.params.id;
  const { type, params } = req.body;
  const userEmail = req.user?.email;

  try {
    const userRes = await pool.query("SELECT user_id, role, full_name FROM users WHERE email = $1", [userEmail]);
    const user = userRes.rows[0];
    if (!(await validateDatasetAccess(user.user_id, datasetId, user.role))) {
      return res.status(403).json({ success: false, message: "Access denied" });
    }

    const dsRes = await pool.query("SELECT file_name FROM datasets WHERE dataset_id = $1", [datasetId]);
    const { file_name } = dsRes.rows[0];
    const fullName = user.full_name || "unknown_user";
    const paths = getDatasetPaths(datasetId, file_name, fullName);

    const fs = (await import("fs/promises")).default;
    const tempDir = path.resolve(process.cwd(), "..", "uploads", "temp");
    await fs.mkdir(tempDir, { recursive: true });

    const safeFullName = fullName.replace(/[^a-zA-Z0-9]/g, "_").toLowerCase();
    const outputStamp = Date.now();
    const tempOutputFile = path.join(tempDir, `${datasetId}_${safeFullName}_${outputStamp}.csv`);

    // Ensure temp file exists (Resume logic)
    let sourcePath = null;
    try {
      const resolved = await resolveDatasetFilePath(datasetId, file_name, fullName);
      sourcePath = resolved.targetPath;
    } catch {
      // If no temp file, start from either Cleaned (if re-cleaning) or Raw
      sourcePath = null;
    }

    if (!sourcePath) {
      sourcePath = paths.raw;
      try { await fs.access(paths.cleaned); sourcePath = paths.cleaned; } catch {}
      if (!await pathExists(sourcePath)) {
        return res.status(404).json({
          success: false,
          message: "Dataset file not found on disk. Re-upload or reprocess this dataset.",
        });
      }
    }

    // Set status to cleaning
    const dsStatusResult = await pool.query("SELECT upload_status, dataset_name FROM datasets WHERE dataset_id = $1", [datasetId]);
    const oldStatus = dsStatusResult.rows[0]?.upload_status;
    const dsName = dsStatusResult.rows[0]?.dataset_name;

    await pool.query("UPDATE datasets SET upload_status = 'cleaning', updated_at = NOW() WHERE dataset_id = $1", [datasetId]);
    
    // Log CLEAN_START if this is the beginning of a cleaning session
    if (oldStatus !== 'cleaning') {
      await logCleaningActivity(user.user_id, user.full_name, userEmail, datasetId, dsName, 'CLEAN_START', "ok", "Cleaning session started");
    }



    // Call Python transformer
    const { spawn } = await import("child_process");
    const pathMod = await import("path");
    const transformerScript = pathMod.resolve(process.cwd(), "..", "ml_engine", "pipeline", "transformer.py");
    
    const pythonProcess = spawn("python", [
      transformerScript,
      "--input", sourcePath,
      "--output", tempOutputFile,
      "--config", JSON.stringify({ type, params })
    ]);

    let stdout = "";
    pythonProcess.stdout.on("data", (data) => (stdout += data.toString()));

    pythonProcess.on("close", async (code) => {
      if (code === 0) {
        try {
          const result = JSON.parse(stdout.trim());
          if (result.status === "success") {
            return res.json({ success: true, message: "Transformation applied", stats: result });
          } else {
            throw new Error(result.message);
          }
        } catch (err) {
          await fs.unlink(tempOutputFile).catch(() => {});
          await pool.query("UPDATE datasets SET upload_status = 'not_cleaned', updated_at = NOW() WHERE dataset_id = $1", [datasetId]);
          return res.status(500).json({ success: false, message: err.message });
        }
      } else {
        await fs.unlink(tempOutputFile).catch(() => {});
        await pool.query("UPDATE datasets SET upload_status = 'not_cleaned', updated_at = NOW() WHERE dataset_id = $1", [datasetId]);
        return res.status(500).json({ success: false, message: "Python transformation failed (System Error)" });
      }
    });

  } catch (err) {
    console.error("transformDataset error:", err);
    return res.status(500).json({ success: false, message: "Server error" });
  }
};

export const finalizeDataset = async (req, res) => {
  const datasetId = req.params.id;
  const userEmail = req.user?.email;

  try {
    const userRes = await pool.query("SELECT user_id, role, full_name FROM users WHERE email = $1", [userEmail]);
    const user = userRes.rows[0];
    if (!(await validateDatasetAccess(user.user_id, datasetId, user.role))) {
      return res.status(403).json({ success: false, message: "Access denied" });
    }

    const dsRes = await pool.query("SELECT file_name FROM datasets WHERE dataset_id = $1", [datasetId]);
    const { file_name } = dsRes.rows[0];
    const fullName = user.full_name || "unknown_user";
    const paths = getDatasetPaths(datasetId, file_name, fullName);

    const fs = (await import("fs/promises")).default;
    await fs.mkdir(path.dirname(paths.cleaned), { recursive: true });

    const resolved = await resolveDatasetFilePath(datasetId, file_name, fullName);
    const workingPath = resolved.targetPath;
    if (!workingPath) {
      return res.status(404).json({
        success: false,
        message: "Temporary cleaned dataset not found. Start cleaning again before finalizing.",
      });
    }

    // Copy the latest working file to cleaned, then remove the working copy if it lives in temp.
    await fs.copyFile(workingPath, paths.cleaned);
    if (workingPath.includes(path.resolve(process.cwd(), "..", "uploads", "temp"))) {
      await fs.unlink(workingPath).catch(() => {});
    }

    await pool.query("UPDATE datasets SET upload_status = 'cleaned', updated_at = NOW() WHERE dataset_id = $1", [datasetId]);
    
    // Log Activity (CLEAN_DONE)
    await logCleaningActivity(user.user_id, user.full_name, userEmail, datasetId, file_name, 'CLEAN_DONE', "ok", "Cleaning completed and finalized");



    return res.json({ success: true, message: "Dataset finalized and ready for visualization" });
  } catch (err) {
    console.error("finalizeDataset error:", err);
    return res.status(500).json({ success: false, message: "Failed to finalize dataset" });
  }
};

export const trainDataset = async (req, res) => {
  return res.json({
    success: true,
    message: "Model training is automatically handled by the background AI pipeline.",
  });
};

export const getAnalysis = async (req, res) => {
  const datasetId = req.params.id;
  const userId = req.user?.email;
  if (!userId) {
    return res.status(401).json({ success: false, message: "Authentication required" });
  }

  const userRole = req.user?.role;
  const dbUserId = (await pool.query("SELECT user_id FROM users WHERE email = $1", [userId])).rows[0]?.user_id;
  if (!await validateDatasetAccess(dbUserId, datasetId, userRole)) {
    return res.status(403).json({ success: false, message: "Access denied" });
  }

  // Resolve the correct file to use for analysis (cleaned > working > raw)
  const dsRowResult = await pool.query("SELECT file_name FROM datasets WHERE dataset_id = $1", [datasetId]);
  const fileName = dsRowResult.rows[0]?.file_name;
  if (!fileName) {
    return res.json({ success: true, dataset_name: datasetId, row_count: 0, column_count: 0, quality_score: null, total_nulls: 0, duplicate_rows: 0, cleaning_report: [], columns: [] });
  }

  const paths = getDatasetPaths(datasetId, fileName);
  let cleanedDataPath = null;

  // Prefer cleaned, then working, then raw
  if (await fs.access(paths.cleaned).then(() => true).catch(() => false)) {
    cleanedDataPath = paths.cleaned;
  } else if (await fs.access(paths.working).then(() => true).catch(() => false)) {
    cleanedDataPath = paths.working;
  } else if (await fs.access(paths.raw).then(() => true).catch(() => false)) {
    cleanedDataPath = paths.raw;
  }

  if (!cleanedDataPath) {
    return res.json({ success: true, dataset_name: datasetId, row_count: 0, column_count: 0, quality_score: null, total_nulls: 0, duplicate_rows: 0, cleaning_report: [], columns: [] });
  }

  try {
    // Try to read profile report
    const profilePath = path.join(paths.artifacts, "profile_report.json");
    let profile = {};
    try {
      profile = JSON.parse(await fs.readFile(profilePath, "utf-8"));
    } catch { }

    // Read cleaned data to get column analysis
    let columns = [];
    let cleaningReport = [];

    try {
      const csv = await fs.readFile(cleanedDataPath, 'utf-8');
      const lines = csv.split('\n').filter(line => line.trim());
      if (lines.length > 1) {
        const headers = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''));

        // Read a sample to analyze columns
        const sampleLines = lines.slice(1, Math.min(101, lines.length));

        columns = headers.map((colName, idx) => {
          const colValues = sampleLines.map(line => {
            const cells = line.split(',').map(c => c.trim().replace(/^"|"$/g, ''));
            return cells[idx];
          }).filter(v => v !== '' && v !== undefined);

          const nullCount = sampleLines.length - colValues.length;
          const nullPct = (nullCount / sampleLines.length) * 100;

          // Determine type from sample
          const numericCount = colValues.filter(v => !isNaN(parseFloat(v)) && isFinite(v)).length;
          const isNumeric = numericCount > colValues.length * 0.7;

          // Check if datetime
          const dateCount = colValues.filter(v => !isNaN(Date.parse(v)) && v.length > 6).length;
          const isDateTime = dateCount > colValues.length * 0.5;

          const uniqueCount = new Set(colValues).size;

          return {
            name: colName,
            type: isNumeric ? 'float64' : isDateTime ? 'datetime' : 'string',
            null_count: nullCount,
            null_pct: parseFloat(nullPct.toFixed(2)),
            nunique: uniqueCount,
            sample: colValues.slice(0, 3),
            inferred_type: isNumeric ? 'numeric' : isDateTime ? 'datetime' : uniqueCount < 10 ? 'categorical' : 'text'
          };
        });
      }
    } catch (err) {
      console.warn('Could not read cleaned data:', err.message);
    }

    // Read metadata for cleaning report
    const metadataPath = path.join(paths.artifacts, "dataset_metadata.json");
    let qualityScore = null;
    let totalNulls = 0;
    let duplicateRows = 0;
    let cleaningStats = {
      missing_values_handled: 0,
      duplicates_removed: 0,
      outliers_removed: 0,
      data_type_fixes: 0
    };

    try {
      const metadata = JSON.parse(await fs.readFile(metadataPath, 'utf-8'));
      qualityScore = metadata.data_quality_score;
      cleaningStats.missing_values_handled = metadata.missing_values_handled || 0;
      cleaningStats.duplicates_removed = metadata.duplicates_removed || 0;
      cleaningStats.outliers_removed = metadata.outliers_removed || 0;
      cleaningStats.data_type_fixes = metadata.data_type_fixes || 0;
      duplicateRows = metadata.duplicates_removed || 0;
    } catch { }

    // Calculate totals
    totalNulls = columns.reduce((sum, c) => sum + (c.null_count || 0), 0);

    // Generate cleaning report
    if (cleaningStats.missing_values_handled > 0 || cleaningStats.duplicates_removed > 0 || cleaningStats.outliers_removed > 0) {
      cleaningReport = [
        { category: 'Missing Values', count: cleaningStats.missing_values_handled, action: 'Filled using MEAN/MEDIAN/MODE', reason: 'Columns with missing values were imputed' },
        { category: 'Duplicates', count: cleaningStats.duplicates_removed, action: 'Removed duplicate rows', reason: 'Identical rows detected in dataset' },
        { category: 'Outliers', count: cleaningStats.outliers_removed, action: 'Removed using IQR method', reason: 'Values outside 1.5*IQR range detected' },
        { category: 'Data Types', count: cleaningStats.data_type_fixes, action: 'Converted to standard format', reason: 'Date columns converted to datetime' }
      ].filter(item => item.count > 0);
    }

    return res.json({
      success: true,
      dataset_name: profile.dataset_name || datasetId,
      row_count: profile.row_count || 0,
      column_count: profile.column_count || columns.length,
      quality_score: qualityScore,
      total_nulls: totalNulls,
      duplicate_rows: duplicateRows,
      cleaning_report: cleaningReport,
      columns: columns
    });
  } catch (err) {
    console.error('getAnalysis error:', err);
    return res.status(500).json({ success: false, message: err.message });
  }
};

export const getMetrics = async (req, res) => {
  const datasetId = req.params.id;
  const userEmail = req.user?.email;
  if (!userEmail) return res.status(401).json({ success: false, message: "Authentication required" });

  try {
    const userResult = await pool.query("SELECT user_id, role FROM users WHERE email = $1", [userEmail]);
    const user = userResult.rows[0];
    if (!user) return res.status(401).json({ success: false, message: "User not found" });
    if (!(await validateDatasetAccess(user.user_id, datasetId, user.role))) {
      return res.status(403).json({ success: false, message: "Access denied" });
    }

    // Read metrics from DB schema_json (written by cleaning pipeline)
    const dsResult = await pool.query(
      "SELECT schema_json, dataset_name FROM datasets WHERE dataset_id = $1",
      [datasetId]
    );
    if (dsResult.rows.length === 0) return res.status(404).json({ success: false, message: "Dataset not found" });
    const { schema_json, dataset_name } = dsResult.rows[0];

    return res.json({
      success: true,
      dataset_name,
      rawStats: schema_json?.rawStats || schema_json?.raw_stats || null,
      qualityScore: schema_json?.qualityScore || schema_json?.data_quality_score || null,
      cleaningSteps: schema_json?.cleaningSteps || schema_json?.cleaning_steps || []
    });
  } catch (err) {
    console.error("[getMetrics]", err);
    return res.status(500).json({ success: false, message: "Internal server error" });
  }
};

export const getDashboardConfig = async (req, res) => {
  const datasetId = req.params.id;
  const userEmail = req.user?.email;

  try {
    const userResult = await pool.query("SELECT user_id, role FROM users WHERE email = $1", [userEmail]);
    const user = userResult.rows[0];
    if (!user) return res.status(401).json({ success: false, message: "User not found" });

    if (!(await validateDatasetAccess(user.user_id, datasetId, user.role))) {
      return res.status(403).json({ success: false, message: "Access denied" });
    }

    // Check if there is a cached dashboard config in schema_json
    const dsResult = await pool.query("SELECT schema_json FROM datasets WHERE dataset_id = $1", [datasetId]);
    const schemaJson = dsResult.rows[0]?.schema_json || {};

    if (schemaJson.dashboard_config) {
      return res.json({ success: true, ...schemaJson.dashboard_config });
    }

    // No config available yet — return empty gracefully
    return res.json({ success: true, charts: [], insights: [], executive_summary: "" });
  } catch (err) {
    console.error("[getDashboardConfig] Error:", err);
    return res.status(500).json({ success: false, message: "Internal server error" });
  }
};

// Delete dataset
export const deleteDataset = async (req, res) => {
  const datasetId = req.params.id;
  const userId = req.user?.email;

  if (!userId) {
    return res.status(401).json({ success: false, message: "Authentication required" });
  }

  try {
    // Get user's company
    const userResult = await pool.query("SELECT company_id FROM users WHERE email = $1", [userId]);
    if (userResult.rows.length === 0) {
      return res.status(401).json({ success: false, message: "User not found" });
    }
    const companyId = userResult.rows[0].company_id;

    // Check if dataset belongs to user's company
    const datasetCheck = await pool.query(
      "SELECT dataset_id, dataset_name FROM datasets WHERE dataset_id = $1 AND company_id = $2",
      [datasetId, companyId]
    );

    if (datasetCheck.rows.length === 0) {
      return res.status(404).json({ success: false, message: "Dataset not found or access denied" });
    }

    // Delete from database
    await pool.query("DELETE FROM datasets WHERE dataset_id = $1", [datasetId]);

    // Delete files from ml_engine directory
    const datasetDir = path.resolve(process.cwd(), "..", "ml_engine", "data", "users", userId, datasetId);
    try {
      await fs.rm(datasetDir, { recursive: true, force: true });
    } catch (fsErr) {
      console.warn("Could not delete dataset files:", fsErr.message);
    }

    return res.json({ success: true, message: "Dataset deleted successfully" });
  } catch (err) {
    console.error("deleteDataset error:", err);
    return res.status(500).json({ success: false, message: "Server error" });
  }
};

export const pauseCleaning = async (req, res) => {
  const datasetId = req.params.id;
  const userEmail = req.user?.email;

  try {
    const userRes = await pool.query("SELECT user_id, full_name, role FROM users WHERE email = $1", [userEmail]);
    const user = userRes.rows[0];
    if (!user) {
      return res.status(401).json({ success: false, message: "User not found" });
    }
    
    const dsRes = await pool.query("SELECT dataset_name, upload_status FROM datasets WHERE dataset_id = $1", [datasetId]);
    const dsName = dsRes.rows[0]?.dataset_name;
    const status = dsRes.rows[0]?.upload_status;

    if (status === 'cleaning') {
      await logCleaningActivity(user.user_id, user.full_name, userEmail, datasetId, dsName, 'CLEAN_PAUSE', "ok", "Cleaning session paused (User left page)");
      // Optional: Reset status to not_cleaned if no other user is cleaning? 
      // For now, just log the pause.
    }


    res.json({ success: true });
  } catch (err) {
    res.status(500).json({ success: false });
  }
};

export const getAvailableDatasetsToRequest = async (req, res) => {
  const userEmail = req.user?.email;
  try {
    const userRes = await pool.query("SELECT user_id, company_id FROM users WHERE email = $1", [userEmail]);
    const user = userRes.rows[0];
    if (!user) return res.status(404).json({ success: false, message: "User not found" });

    // Fetch datasets in company NOT assigned to user OR can_view is false
    const query = `
      SELECT d.*, u.full_name AS uploaded_by_name
      FROM datasets d
      LEFT JOIN users u ON d.uploaded_by = u.user_id
      WHERE d.company_id = $1
      AND d.upload_status != 'failed'
      AND d.dataset_id NOT IN (
        SELECT dataset_id FROM permissions WHERE user_id = $2 AND can_view = TRUE
      )
      ORDER BY d.created_at DESC

    `;
    const result = await pool.query(query, [user.company_id, user.user_id]);

    res.json({
      success: true,
      data: result.rows.map(mapDatasetRow)
    });
  } catch (err) {
    console.error("getAvailableDatasetsToRequest error:", err);
    res.status(500).json({ success: false, message: "Server error" });
  }
};

