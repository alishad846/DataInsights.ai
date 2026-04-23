import express from "express";
import multer from "multer";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { dirname } from "path";
import {
  ragUpload,
  ragChat,
  ragStatus,
  ragModels,
  ragClear,
} from "../controllers/ragChatController.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const router = express.Router();

// Upload directory for RAG files
const ragUploadDir = path.join(__dirname, "../../../ml_engine/rag_data/uploads");
if (!fs.existsSync(ragUploadDir)) {
  fs.mkdirSync(ragUploadDir, { recursive: true });
}

const storage = multer.diskStorage({
  destination: ragUploadDir,
  filename: (req, file, cb) => {
    cb(null, `${Date.now()}-${file.originalname}`);
  },
});

const upload = multer({
  storage,
  fileFilter: (req, file, cb) => {
    const allowedExts = [".csv", ".xlsx", ".xls", ".json", ".pdf", ".txt"];
    const ext = path.extname(file.originalname).toLowerCase();
    if (allowedExts.includes(ext)) {
      cb(null, true);
    } else {
      cb(new Error(`Unsupported file type: ${ext}. Allowed: ${allowedExts.join(", ")}`));
    }
  },
  limits: { fileSize: 50 * 1024 * 1024 }, // 50MB
});

router.post("/rag/upload", upload.single("file"), ragUpload);
router.post("/rag/chat", ragChat);
router.get("/rag/status", ragStatus);
router.get("/rag/models", ragModels);
router.post("/rag/clear", ragClear);

export default router;
