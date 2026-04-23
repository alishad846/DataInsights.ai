import express from "express";
import {
  analyticsSummary,
  analyticsChart
} from "../controllers/analyticsController.js";
import { protect } from "../middleware/protect.js";
import { logViewSummaryActivity } from "../controllers/activityController.js";
import { pool } from "../config/db.js";

const router = express.Router();

router.get("/", protect, analyticsSummary);
router.post("/summary", protect, async (req, res, next) => {
  const datasetId = req.query.datasetId || req.body?.datasetId;
  
  try {
    const userId = req.user?.id || req.user?.userId || req.user?.user_id || null;
    const userEmail = req.user?.email;
    
      if (userId && datasetId) {
        let userName = req.user?.full_name || req.user?.name || userEmail?.split('@')[0] || 'Unknown';
        let datasetName = datasetId;
        
        try {
          const dsResult = await pool.query(
            `SELECT d.dataset_name, COALESCE(u.email, $2) as uploaded_by
             FROM datasets d
             LEFT JOIN users u ON d.uploaded_by = u.user_id
             WHERE d.dataset_id = $1`,
            [datasetId, userEmail]
          );
          if (dsResult.rows.length > 0) {
            datasetName = dsResult.rows[0].dataset_name || datasetName;
            userName = dsResult.rows[0].uploaded_by || userName;
          }
        } catch (e) {}
      
      await logViewSummaryActivity(userId, userName, userEmail, datasetId, datasetName, 'ok');
    }
  } catch (err) {
    console.error("Activity logging error:", err);
  }
  
  next();
}, analyticsSummary);
router.post("/chart", protect, analyticsChart);

export default router;
