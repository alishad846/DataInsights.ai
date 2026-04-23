import { pool } from "../config/db.js";
import path from "path";
import fs from "fs/promises";

const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

/**
 * Validates if a user has access to a specific dataset.
 * @param {string} userId - UUID of the user.
 * @param {string} datasetId - UUID of the dataset.
 * @param {string} role - User role ('admin' or 'employee').
 * @returns {Promise<boolean>}
 */
export async function validateDatasetAccess(userId, datasetId, role) {
  if (role === 'admin') return true;
  if (!UUID_PATTERN.test(userId || "") || !UUID_PATTERN.test(datasetId || "")) {
    return false;
  }

  try {
    const result = await pool.query(
      "SELECT can_view FROM permissions WHERE user_id = $1 AND dataset_id = $2 AND can_view = TRUE",
      [userId, datasetId]
    );
    return result.rows.length > 0;
  } catch (err) {
    console.error("[AccessUtils] Error checking permissions:", err.message);
    return false;
  }
}

/**
 * Returns standardized paths for dataset files.
 * @param {string} datasetId - UUID of the dataset.
 * @param {string} fileName - Original sanitized filename or generated unique name.
 * @param {string} fullName - Full name of the user (used for temp files).
 */
export function getDatasetPaths(datasetId, fileName, fullName = null) {
  const rootDir = process.cwd();
  const uploadsDir = path.resolve(rootDir, "..", "uploads");
  
  const sanitizedFullName = fullName ? fullName.replace(/[^a-zA-Z0-9]/g, '_').toLowerCase() : 'unknown';

  return {
    raw: path.join(uploadsDir, "raw", fileName),
    temp: fullName ? path.join(uploadsDir, "temp", `${datasetId}_${sanitizedFullName}.csv`) : null,
    working: path.join(uploadsDir, "raw", `working_${datasetId}.csv`), // Legacy working path (deprecating)
    cleaned: path.join(uploadsDir, "cleaned", `cleaned_${datasetId}.csv`),
    artifacts: path.resolve(rootDir, "..", "ml_engine", "artifacts", datasetId)
  };
}

/**
 * Ensures a directory exists for artifacts.
 */
export async function ensureArtifactsDir(datasetId) {
  const { artifacts } = getDatasetPaths(datasetId);
  await fs.mkdir(artifacts, { recursive: true });
  return artifacts;
}
