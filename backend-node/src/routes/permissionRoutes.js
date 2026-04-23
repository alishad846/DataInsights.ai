import express from "express";
import { protect } from "../middleware/protect.js";
import { isAdmin } from "../middleware/isAdmin.js";
import {
  submitPermissionRequest,
  getPendingRequests,
  resolvePermissionRequest,
  getPermissionBadgeCounts
} from "../controllers/permissionController.js";

const router = express.Router();

// Employee routes
router.post("/request", protect, submitPermissionRequest);

// Admin routes
router.get("/pending", protect, isAdmin, getPendingRequests);
router.post("/resolve", protect, isAdmin, resolvePermissionRequest);
router.get("/badge-counts", protect, isAdmin, getPermissionBadgeCounts);


export default router;
