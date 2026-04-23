import jwt from "jsonwebtoken";
import { pool } from "../config/db.js";

export const protect = async (req, res, next) => {
  try {
    const token = req.headers.authorization?.replace("Bearer ", "");

    if (!token) {
      return res.status(401).json({ message: "Not authorized, no token" });
    }

    const decoded = jwt.verify(token, process.env.JWT_SECRET || "secret");
    const userResult = await pool.query(
      "SELECT user_id, email, full_name, role, company_id, is_active FROM users WHERE email = $1",
      [decoded.email]
    );

    if (userResult.rows.length === 0) {
      return res.status(401).json({ message: "Not authorized, user not found" });
    }

    const u = userResult.rows[0];

    if (!u.is_active) {
      return res.status(401).json({ message: "Not authorized, account is inactive" });
    }

    req.user = {
      id: u.user_id,
      userId: u.user_id,
      user_id: u.user_id,
      email: u.email,
      name: u.full_name,
      full_name: u.full_name,
      role: decoded.role || "employee",
      company_id: u.company_id,
    };

    next();
  } catch (err) {
    console.error("Protect middleware error:", err.message);
    res.status(401).json({ message: "Not authorized, invalid token" });
  }
};
