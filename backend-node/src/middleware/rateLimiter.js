import rateLimit from "express-rate-limit";

// Limit to 50 requests per minute per IP for queries
export const queryLimiter = rateLimit({
  windowMs: 1 * 60 * 1000, // 1 minute
  max: 50,
  message: { success: false, message: "Too many query requests from this IP, please try again after a minute" }
});

// Limit to 50 dataset uploads per hour per IP (increased for testing)
export const uploadLimiter = rateLimit({
  windowMs: 60 * 60 * 1000, // 1 hour
  max: 50,
  message: { success: false, message: "Too many uploads from this IP, please try again after an hour" }
});
