import path from "path";
import fs from "fs/promises";
import { pool } from "../config/db.js";
import { validateDatasetAccess, getDatasetPaths } from "../utils/accessUtils.js";

// ─────────────────────────────────────────────────────────────────────────────
// GET /api/visualization/:id
// Returns dashboard config from the DB, or a graceful empty response.
// Previously searched for a dead ml_engine/data/users/... path.
// ─────────────────────────────────────────────────────────────────────────────
export const getVisualization = async (req, res) => {
  try {
    const datasetId = req.params.id;
    if (!datasetId) return res.status(400).json({ success: false, message: "Dataset ID required" });

    const userEmail = req.user?.email;
    if (!userEmail) return res.status(401).json({ success: false, message: "Authentication required" });

    // Access control via DB
    const userResult = await pool.query("SELECT user_id, role FROM users WHERE email = $1", [userEmail]);
    const user = userResult.rows[0];
    if (!user) return res.status(401).json({ success: false, message: "User not found" });

    if (!(await validateDatasetAccess(user.user_id, datasetId, user.role))) {
      return res.status(403).json({ success: false, message: "Access denied" });
    }

    // Read dashboard config from DB schema_json (set by cleaning/finalization pipeline)
    const dsResult = await pool.query(
      "SELECT schema_json, dataset_name FROM datasets WHERE dataset_id = $1",
      [datasetId]
    );
    const row = dsResult.rows[0];
    if (!row) return res.status(404).json({ success: false, message: "Dataset not found" });

    const schemaJson = row.schema_json || {};

    if (schemaJson.dashboard_config) {
      return res.json({ success: true, dataset_name: row.dataset_name, ...schemaJson.dashboard_config });
    }

    // No config yet — return empty gracefully (avoids frontend crash)
    return res.json({ success: true, dataset_name: row.dataset_name, charts: [], insights: [], executive_summary: "" });

  } catch (err) {
    console.error("[VISUALIZATION CONTROLLER]", err);
    return res.status(500).json({ success: false, message: "Internal server error" });
  }
};
