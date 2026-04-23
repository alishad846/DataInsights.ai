import path from "path";
import { spawn } from "child_process";
import { pool } from "../config/db.js";
import { validateDatasetAccess, getDatasetPaths } from "../utils/accessUtils.js";
import fs from "fs/promises";
import { logActivity, logChatActivity, logPermissionActivity } from "./activityController.js";


export const askQuestion = async (req, res) => {
  const { message, question, datasetId } = req.body;
  const queryText = (message || question || "").trim();
  const userEmail = req.user?.email;

  if (!queryText || !datasetId) {
    return res.status(400).json({ success: false, message: "Message and datasetId are required." });
  }

  try {
    const userRes = await pool.query("SELECT user_id, role, company_id FROM users WHERE email = $1", [userEmail]);
    const user = userRes.rows[0];
    if (!user) return res.status(401).json({ success: false, message: "User not found" });

    // 1. Permission Check
    // Get granular permissions from database
    let granularPerms = { can_view: false, can_edit: false, can_query: false };
    
    if (user.role === 'admin') {
      granularPerms = { can_view: true, can_edit: true, can_query: true };
    } else {
      const permRes = await pool.query(
        "SELECT can_view, can_edit, can_query FROM permissions WHERE user_id = $1 AND dataset_id = $2",
        [user.user_id, datasetId]
      );
      if (permRes.rows.length > 0) {
        granularPerms = permRes.rows[0];
      }
    }

    if (!granularPerms.can_view && !granularPerms.can_query) {
      return res.status(403).json({ success: false, message: "Forbidden: You do not have access to this dataset" });
    }

    const dsRes = await pool.query("SELECT file_name FROM datasets WHERE dataset_id = $1", [datasetId]);
    if (!dsRes.rows[0]) return res.status(404).json({ success: false, message: "Dataset not found" });

    const { file_name } = dsRes.rows[0];
    const paths = getDatasetPaths(datasetId, file_name);

    // 2. Resolve the best available CSV file (cleaned > working > raw)
    let csvFilePath = null;
    for (const candidate of [paths.cleaned, paths.working, paths.raw]) {
      try { await fs.access(candidate); csvFilePath = candidate; break; } catch {}
    }

    if (!csvFilePath) {
      return res.json({
        success: true,
        source: "fallback",
        answer: "⚠️ This dataset has not been finalized yet. Please complete the cleaning process first."
      });
    }

    // 3. Spawn Python Cognitive Engine — pass explicit csv_file and granular permissions
    const pythonScript = path.resolve(process.cwd(), "..", "ml_engine", "pipeline", "cognitive_engine.py");

    const pyProcess = spawn("python", [
      pythonScript,
      "--user_id",    userEmail,
      "--dataset_id", datasetId,
      "--question",   queryText,
      "--dataset_dir", path.dirname(csvFilePath),
      "--csv_file",   csvFilePath,
      "--permissions", JSON.stringify(granularPerms)
    ]);
    
    const startTimeMs = Date.now();

    let stdout = "";
    let stderr = "";
    pyProcess.stdout.on("data", (data) => (stdout += data.toString()));
    pyProcess.stderr.on("data", (data) => (stderr += data.toString()));

    pyProcess.on("close", async (code) => {
      if (stderr) console.warn(`[query_engine stderr]: ${stderr}`);
      if (code !== 0) {
        console.error(`[query_engine] exited with code ${code}`);
        return res.json({ success: true, source: "fallback", answer: "The query engine encountered an error. Please try rephrasing your question." });
      }
      try {
        const result = JSON.parse(stdout.trim());
        const duration = Date.now() - startTimeMs;
        
        // Log deep AI query metrics into query_logs
        try {
          await pool.query(
            "INSERT INTO query_logs (company_id, user_id, dataset_id, query_text, query_type, execution_time_ms, status, generated_code, error_msg) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
            [
              user.company_id || null, 
              user.user_id, 
              datasetId, 
              queryText, 
              result.intent || 'unknown',
              duration,
              result.success === false ? "failed" : "success",
              result.generated_code || null,
              result.error || null
            ]
          );
        } catch (dbErr) {
          console.error("Could not log deep AI query to query_logs:", dbErr);
        }
        
        // Handle permission requests dynamically returned by python LLM
        if (result.success === false && result.require_permission) {
          // Log Permission Denied
          const dsNameRes = await pool.query("SELECT dataset_name FROM datasets WHERE dataset_id = $1", [datasetId]);
          const dsName = dsNameRes.rows[0]?.dataset_name || "Unknown Dataset";
          
          await logPermissionActivity(
            user.user_id,
            user.full_name,
            userEmail,
            datasetId,
            dsName,
            "PERM_DENIED",
            `User denied ${result.require_permission} on ${dsName} (Intent: ${result.intent})`
          );

          return res.json({
            success: false,
            require_permission: result.require_permission,
            answer: result.answer,
            intent: result.intent
          });
        }


        return res.json({
          success: true,
          source: result.fallback_used ? "ml-engine-fallback" : "ml-engine",
          answer: result.answer || "I couldn't find an answer.",
          intent: result.intent,
          confidence: result.confidence,
          suggested_questions: result.suggested_questions || []
        });
      } catch {
        // Raw text response (shouldn't happen, but handle gracefully)
        return res.json({ success: true, source: "ml-engine-raw", answer: stdout.trim() });
      }
    });

  } catch (err) {
    console.error("askQuestion error:", err);
    return res.status(500).json({ success: false, message: "Internal server error" });
  }
};

export const startChatSession = async (req, res) => {
  try {
    const { datasetId } = req.body;
    const userId = req.user.user_id;
    const email = req.user.email;
    
    const dsRes = await pool.query("SELECT dataset_name FROM datasets WHERE dataset_id = $1", [datasetId]);
    const dsName = dsRes.rows[0]?.dataset_name || "Unknown Dataset";
    
    await logChatActivity(userId, null, email, datasetId, dsName, "CHAT_START");
    res.json({ success: true });
  } catch (err) {
    res.status(500).json({ success: false });
  }
};

export const endChatSession = async (req, res) => {
  try {
    const { datasetId, reason } = req.body; // reason: 'closed', 'cleared'
    const userId = req.user.user_id;
    const email = req.user.email;
    
    const dsRes = await pool.query("SELECT dataset_name FROM datasets WHERE dataset_id = $1", [datasetId]);
    const dsName = dsRes.rows[0]?.dataset_name || "Unknown Dataset";
    
    await logChatActivity(userId, null, email, datasetId, dsName, "CHAT_END", `Session ended: ${reason || 'User finished'}`);
    res.json({ success: true });
  } catch (err) {
    res.status(500).json({ success: false });
  }
};
