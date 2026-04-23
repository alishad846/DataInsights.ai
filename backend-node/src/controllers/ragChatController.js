/**
 * [DEPRECATED] This controller is being phased out in favor of chatController.js.
 * All RAG functionality is being consolidated into a single stabilized chatbot engine.
 */
import { askQuestion } from "./chatController.js";

export const ragUpload = async (req, res) => {
  return res.status(410).json({ error: "Endpoint deprecated. Use direct upload instead." });
};

export const ragChat = async (req, res) => {
  console.warn("[DEPRECATED] ragChat called. Redirecting to chatController.askQuestion");
  return askQuestion(req, res);
};

export const ragStatus = async (req, res) => {
  return res.json({ server_running: true, consolidated: true });
};

export const ragModels = async (req, res) => {
  return res.json({ models: ["consolidated-engine"] });
};

export const ragClear = async (req, res) => {
  return res.json({ status: "success" });
};
