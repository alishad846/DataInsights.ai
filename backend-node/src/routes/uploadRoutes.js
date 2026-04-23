import express from "express";
import multer from "multer";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { dirname } from "path";
import { uploadDataset } from "../controllers/uploadController.js";
import { uploadLimiter } from "../middleware/rateLimiter.js";
import { protect } from "../middleware/protect.js";
import { logActivity } from "../controllers/activityController.js";
import { pool } from "../config/db.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const router = express.Router();

const uploadDir = path.join(__dirname, "../../uploads");
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

const storage = multer.diskStorage({
  destination: uploadDir,
  filename: (req, file, cb) => {
    cb(null, `${Date.now()}-${file.originalname}`);
  },
});

const upload = multer({ 
  storage,
  fileFilter: (req, file, cb) => {
    const allowedMimes = [
      "text/csv",
      "application/vnd.ms-excel",
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
      "application/json",
    ];
    
    const allowedExtensions = [".csv", ".xlsx", ".xls", ".json"];
    const fileExt = path.extname(file.originalname).toLowerCase();
    
    if (
      allowedMimes.includes(file.mimetype) ||
      allowedExtensions.includes(fileExt)
    ) {
      cb(null, true);
    } else {
      cb(new Error("Invalid file type. Only CSV, Excel, or JSON files are allowed."));
    }
  },
  limits: {
    fileSize: 100 * 1024 * 1024, // 100MB limit
  },
});

router.post("/upload", protect, uploadLimiter, (req, res, next) => {
  req.uploadStartTime = Date.now();
  req.metrics = [];
  const startMsg = `[UPLOAD-START] dataset upload initiated`;
  console.log(startMsg);
  req.metrics.push(startMsg);

  upload.single("dataset")(req, res, async (err) => {
    const saveTime = Date.now() - req.uploadStartTime;
    const saveMsg = `[FILE-SAVED] file saved in ${saveTime}ms`;
    console.log(saveMsg);
    req.metrics.push(saveMsg);
    req.lastStepTime = Date.now();

    if (err) {
      if (err instanceof multer.MulterError) {
        if (err.code === "LIMIT_FILE_SIZE") {
          return res.status(400).json({
            success: false,
            message: "File too large. Maximum size is 100MB.",
          });
        }
        return res.status(400).json({
          success: false,
          message: err.message || "File upload error",
        });
      }
      return res.status(400).json({
        success: false,
        message: err.message || "Invalid file type",
      });
    }

    if (req.file) {
      const userId = req.user?.id || req.user?.userId || req.user?.user_id || null;
      const userEmail = req.user?.email;
      const userName = req.user?.full_name || req.user?.name || userEmail?.split('@')[0] || 'Unknown';
      const datasetName = req.file.originalname;
      const fileSize = req.file.size;

      try {
        await logActivity({
          userId,
          userName,
          userEmail,
          eventType: "UPLOAD",
          eventDescription: `Uploaded dataset ${datasetName}`,
          datasetName,
          detail: `Uploaded · ${(fileSize / 1024 / 1024 > 1 ? (fileSize / 1024 / 1024).toFixed(1) + 'MB' : (fileSize / 1024).toFixed(1) + 'KB')}`,
          status: "pending"
        });
      } catch (logErr) {
        console.error("Activity logging error:", logErr);
      }
    }

    next();
  });
}, uploadDataset);

export default router;
