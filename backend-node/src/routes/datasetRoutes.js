import express from "express";
import {
  getAllDatasets,
  getDatasetById,
  getDatasetStatus,
  updateDatasetStatus,
  transformDataset,
  finalizeDataset,
  trainDataset,
  getAnalysis,
  getMetrics,
  getDashboardConfig,
  deleteDataset,
  getAllDatasetsAdmin,
  assignDataset,
  unassignDataset,
  getDatasetAssignments,
  getDatasetPreview,
  downloadDataset,
  pauseCleaning,
  getAvailableDatasetsToRequest,

} from "../controllers/datasetController.js";

import { protect } from "../middleware/protect.js";
import { logCleaningActivity } from "../controllers/activityController.js";
import { pool } from "../config/db.js";

const router = express.Router();

/* =====================================================
   DATASET MANAGEMENT
   All routes require authentication
 ===================================================== */

// Get all datasets
router.get("/datasets", protect, getAllDatasets);

// Discover datasets to request access
router.get("/datasets/available-to-request", protect, getAvailableDatasetsToRequest);


// Get all datasets (admin - no company filter)
router.get("/datasets-admin", protect, getAllDatasetsAdmin);

// Get dataset by ID
router.get("/datasets/:id", protect, getDatasetById);

// Get dataset status
router.get("/dataset-status/:id", protect, getDatasetStatus);

// Update dataset status (admin / debug)
router.patch("/datasets/:id/status", protect, updateDatasetStatus);

// Delete dataset
router.delete("/datasets/:id", protect, deleteDataset);

// Dataset Assignment
router.post("/datasets/:id/assign", protect, assignDataset);
router.delete("/datasets/:id/unassign", protect, unassignDataset);
router.get("/datasets/:id/assignments", protect, getDatasetAssignments);
router.get("/datasets/:id/preview", protect, getDatasetPreview);
router.get("/datasets/:id/download", protect, downloadDataset);

/* =====================================================
   ML PIPELINE AUTOMATION
===================================================== */

// Dataset Transformations (Workspace Model)
router.post("/datasets/:id/transform", protect, transformDataset);
router.post("/datasets/:id/finalize", protect, finalizeDataset);
router.post("/datasets/:id/pause", protect, pauseCleaning);


// Train ML model (Python script)
router.post("/datasets/:id/train", protect, trainDataset);

// Get data analysis report
router.get("/datasets/:id/analysis", protect, getAnalysis);

// Get trained model metrics
router.get("/datasets/:id/metrics", protect, getMetrics);

// Get dashboard configuration (charts, insights, KPIs)
router.get("/dashboard/:id", getDashboardConfig);

export default router;
