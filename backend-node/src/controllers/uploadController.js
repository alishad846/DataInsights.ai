import path from "path";
import crypto from "crypto";
import fs from "fs/promises";
import { pool } from "../config/db.js";
import { sanitizeFilename } from "../utils/fileUtils.js";

export const uploadDataset = async (req, res) => {
  try {
    if (req.user?.role !== "admin") {
      return res.status(403).json({ success: false, message: "Only administrators can upload datasets." });
    }

    if (!req.file) {
      return res.status(400).json({ success: false, message: "No file uploaded. Use key 'dataset'." });
    }

    const userEmail = req.user?.email || "default_user";
    const datasetId = crypto.randomUUID();
    const sanitizedName = sanitizeFilename(req.file.originalname);
    const finalFileName = `${datasetId}_${sanitizedName}`;

    const centralStorageDir = path.resolve(process.cwd(), "..", "uploads", "raw");
    const finalDestPath = path.join(centralStorageDir, finalFileName);

    console.log(`[UPLOAD] userEmail: ${userEmail}, datasetId: ${datasetId}, finalName: ${finalFileName}`);

    try {
      await fs.mkdir(centralStorageDir, { recursive: true });
      await fs.rename(req.file.path, finalDestPath);
      console.log(`[FILE] Moved ${req.file.path} -> ${finalDestPath}`);
    } catch (moveErr) {
      console.error("[FILE-MOVE] Error:", moveErr.message);
      try {
        await fs.copyFile(req.file.path, finalDestPath);
        await fs.unlink(req.file.path);
        console.log(`[FILE-FALLBACK] Copied and unlinked ${req.file.path} -> ${finalDestPath}`);
      } catch (copyErr) {
        throw new Error(`Failed to store uploaded file: ${copyErr.message}`);
      }
    }

    try {
      const userResult = await pool.query("SELECT user_id, company_id FROM users WHERE email = $1", [userEmail]);
      const uploadedBy = userResult.rows[0]?.user_id || null;
      let companyId = userResult.rows[0]?.company_id || null;

      if (!companyId) {
        const companyResult = await pool.query("SELECT company_id FROM companies ORDER BY created_at ASC LIMIT 1");
        companyId = companyResult.rows[0]?.company_id;
      }

      if (!companyId) throw new Error("No company in database environment");

      await pool.query(
        `INSERT INTO datasets (dataset_id, company_id, uploaded_by, dataset_name, file_name, file_size, upload_status)
         VALUES ($1, $2, $3, $4, $5, $6, $7)`,
        [
          datasetId,
          companyId,
          uploadedBy,
          req.file.originalname,
          finalFileName,
          req.file.size || null,
          "not_cleaned",
        ]
      );

      console.log(`[DB] Dataset record created with ID: ${datasetId}`);
    } catch (dbErr) {
      console.error("[DB-INSERT] Error:", dbErr.message);
      return res.status(500).json({ success: false, message: "Database failure during upload registration" });
    }

    // Return response immediately - cleaning is triggered on demand later
    return res.status(200).json({
      success: true,
      datasetId,
      originalName: req.file.originalname,
      fileName: finalFileName,
      size: req.file.size,
      message: "Dataset upload successful. Ready for cleaning.",
      metrics: req.metrics,
    });
  } catch (error) {
    console.error("[UPLOAD ERROR]", error);
    return res.status(500).json({ success: false, message: error.message || "Upload failed" });
  }
};
