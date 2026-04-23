import express from "express";
import cors from "cors";
import dotenv from "dotenv";

import connectDB, { pool } from "./config/db.js";
import errorHandler from "./middleware/errorMiddleware.js";
import analyticsRoutes from "./routes/analyticsRoutes.js";
import visualizationRoutes from "./routes/visualizationRoutes.js";
import uploadRoutes from "./routes/uploadRoutes.js";
import datasetRoutes from "./routes/datasetRoutes.js";
import chatRoutes from "./routes/chatRoutes.js";
import ragChatRoutes from "./routes/ragChatRoutes.js";
import authRoutes from "./routes/authRoutes.js";
import cleanedDataRoutes from "./routes/cleanedDataRoutes.js";
import activityRoutes from "./routes/activityRoutes.js";
import permissionRoutes from "./routes/permissionRoutes.js";

dotenv.config();
await connectDB();

const app = express();

/* ========== MIDDLEWARE ========== */
app.use(cors());
app.use(express.json());

/* ========== ROUTES ========== */
app.use("/api", chatRoutes);
app.use("/api", ragChatRoutes);
app.use("/api", visualizationRoutes);
app.use("/api", uploadRoutes);
app.use("/api", datasetRoutes);
app.use("/api/auth", authRoutes);
app.use("/api/analytics", analyticsRoutes);
app.use("/api", cleanedDataRoutes);
app.use("/api", activityRoutes);
app.use("/api/permissions", permissionRoutes);

/* ========== HEALTH CHECK ========== */
app.get("/", (req, res) => {
  res.send("Backend is running");
});

/* ========== DIAGNOSTIC ENDPOINT ========== */
app.get("/api/health", async (req, res) => {
  let pgStatus = "disconnected";
  try {
    const result = await pool.query("SELECT 1");
    pgStatus = result.rows ? "connected" : "error";
  } catch {
    pgStatus = "error";
  }
  res.json({
    server: "running",
    postgresql: pgStatus,
    port: process.env.PORT || 5000,
  });
});

/* ========== ERROR HANDLER ========== */
app.use(errorHandler);

/* ========== SERVER ========== */
const PORT = Number(process.env.PORT) || 5000;

// Keep the server process alive for local development sessions.
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});

setInterval(() => {}, 1000 * 60 * 60);

export default app;
