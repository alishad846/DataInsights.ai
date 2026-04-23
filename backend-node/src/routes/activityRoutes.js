import express from "express";
import { getActivityLogs, getActivityStats, getQueryVolume } from "../controllers/activityController.js";
import { protect } from "../middleware/protect.js";

const router = express.Router();

router.get("/activity-logs", protect, getActivityLogs);
router.get("/activity-stats", protect, getActivityStats);
router.get("/query-logs/volume", protect, getQueryVolume);

export default router;