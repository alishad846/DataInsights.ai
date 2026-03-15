import path from "path";
import { exec } from "child_process";
import { fileURLToPath } from "url";
import { dirname } from "path";

const __filename = fileURLToPath(import.meta.url);
const __dirname  = dirname(__filename);

// ML engine base path (relative to backend-node/src/controllers)
const ML_ENGINE_DIR = path.resolve(__dirname, "../../../ml_engine");
const QUERY_SCRIPT  = path.join(ML_ENGINE_DIR, "pipeline", "query_engine.py");
const DATA_DIR      = path.join(ML_ENGINE_DIR, "data", "users");

const QUERY_TIMEOUT_MS = 30_000; // 30 second hard limit

export const askQuestion = async (req, res) => {
  const { message, question, datasetId } = req.body;
  const queryText = (message || question || "").trim();

  if (!queryText || !datasetId) {
    return res.status(400).json({
      success: false,
      message: "Both 'message' (or 'question') and 'datasetId' are required.",
    });
  }

  const userId      = req.user?.id || "default_user";
  const datasetDir  = path.join(DATA_DIR, userId, datasetId);

  // Escape args to prevent shell injection
  const safeQuestion   = queryText.replace(/"/g, '\\"');
  const safeUserId     = userId.replace(/"/g, '\\"');
  const safeDatasetId  = datasetId.replace(/"/g, '\\"');
  const safeDatasetDir = datasetDir.replace(/"/g, '\\"');

  const cmd = `python "${QUERY_SCRIPT}" --user_id "${safeUserId}" --dataset_id "${safeDatasetId}" --question "${safeQuestion}" --dataset_dir "${safeDatasetDir}"`;

  let responded = false;
  const timer = setTimeout(() => {
    if (!responded) {
      responded = true;
      console.warn("[ChatController] Python query engine timed out.");
      return safeFallback("The query engine took too long to respond.", res);
    }
  }, QUERY_TIMEOUT_MS);

  try {
    exec(cmd, { timeout: QUERY_TIMEOUT_MS }, (err, stdout, stderr) => {
      if (responded) return;
      clearTimeout(timer);
      responded = true;

      if (err) {
        console.error("[ChatController] Python execution error:", err.message);
        if (stderr) console.error("[ChatController] STDERR:", stderr.slice(0, 500));
        return safeFallback("The query engine encountered an error. Please try again.", res);
      }

      if (!stdout || !stdout.trim()) {
        console.warn("[ChatController] Python returned empty output. STDERR:", stderr?.slice(0, 300));
        return safeFallback("The query engine returned no output.", res);
      }

      try {
        const payload = JSON.parse(stdout.trim());
        return res.json({
          success:    true,
          source:     "ml-engine",
          answer:     payload.answer  || "I could not find an answer to your question.",
          intent:     payload.intent  || "unknown",
          confidence: payload.confidence ?? null,
        });
      } catch (parseErr) {
        console.warn("[ChatController] Failed to parse JSON from Python:", stdout.slice(0, 200));
        // Return raw text as answer (handles non-JSON edge cases)
        return res.json({
          success: true,
          source:  "ml-engine-raw",
          answer:  stdout.trim(),
          intent:  "raw",
        });
      }
    });
  } catch (error) {
    clearTimeout(timer);
    console.error("[ChatController] Unexpected error:", error);
    return safeFallback("An unexpected server error occurred.", res);
  }
};

const safeFallback = (reason, res) => {
  return res.json({
    success:    true,
    source:     "fallback",
    answer:     `⚠️ ${reason} Please verify your dataset was processed successfully, or rephrase your question.`,
    intent:     "error",
    confidence: 0,
  });
};
