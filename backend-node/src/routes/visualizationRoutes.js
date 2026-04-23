import express from "express";
import { getVisualization } from "../controllers/visualizationController.js";
import { protect } from "../middleware/protect.js";
import { logVisualizationActivity } from "../controllers/activityController.js";
import { pool } from "../config/db.js";

const router = express.Router();

router.get(
  "/dashboard/:id",
  protect,
  async (req, res, next) => {
    const startTime = Date.now();
    const datasetId = req.params.id;
    
    try {
      const userId = req.user?.id || req.user?.userId || req.user?.user_id || null;
      const userEmail = req.user?.email;
      
      let userName = req.user?.full_name || req.user?.name || userEmail?.split('@')[0] || 'Unknown';
      let datasetName = datasetId;
      
      if (userId && datasetId) {
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
          }
        } catch (e) {}
        
        await logVisualizationActivity(userId, userName, userEmail, datasetId, datasetName, 'ok', 'Visualization accessed');
      }
      
      req.activityStartTime = startTime;
    } catch (err) {
      console.error("Activity logging error:", err);
    }
    
    next();
  },
  getVisualization
);

export default router;
