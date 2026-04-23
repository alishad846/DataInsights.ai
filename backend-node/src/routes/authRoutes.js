import express from "express";
import {
  login, 
  createUser, getAllUsers, updateUserRole, updateUserStatus, deleteUser, getUserStats,
  getMe, getPendingUsers, approveUser,
  forgotPassword, validateResetToken, resetPassword
} from "../controllers/authController.js";
import { logEmployeeLogout } from "../controllers/activityController.js";
import { protect } from "../middleware/protect.js";
import { isAdmin } from "../middleware/isAdmin.js";

const router = express.Router();

router.post("/logout", protect, async (req, res) => {
  try {
    if (req.user && req.user.id && req.user.email) {
      await logEmployeeLogout(req.user.id, req.user.email.split('@')[0], req.user.email, 0);
    }
  } catch (err) {
    console.error("Logout logging error:", err.message);
  }
  res.json({ success: true, message: "Logged out" });
});

// Auth
router.post("/login", login);
router.post("/forgot-password", forgotPassword);
router.get("/validate-reset-token", validateResetToken);
router.post("/reset-password", resetPassword);
router.get("/me", protect, getMe);

// User management (admin)
router.post("/users", protect, isAdmin, createUser);
router.get("/users", protect, isAdmin, getAllUsers);
router.get("/users/stats", protect, isAdmin, getUserStats);
router.get("/users/pending", protect, isAdmin, getPendingUsers);
router.put("/users/:email/approve", protect, isAdmin, approveUser);
router.put("/users/:email/role", protect, isAdmin, updateUserRole);
router.put("/users/:email/status", protect, isAdmin, updateUserStatus);
router.delete("/users/:email", protect, isAdmin, deleteUser);

export default router;
