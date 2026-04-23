import jwt from "jsonwebtoken";
import { pool } from "../config/db.js";
import { logEmployeeLogin } from "./activityController.js";
import { v4 as uuidv4 } from "uuid";
import { sendResetEmail } from "../utils/mailer.js";

import bcrypt from "bcryptjs";

const SALT_ROUNDS = 12;

function generateInitials(fullName) {
  if (!fullName) return "??";
  const parts = fullName.trim().split(/\s+/).filter(Boolean);
  if (parts.length >= 2) return (parts[0][0] + parts[1][0]).toUpperCase();
  return fullName.slice(0, 2).toUpperCase();
}

const AVATAR_COLORS = ["#c84b2f", "#1d4ed8", "#2d6a4f", "#b45309", "#7c3aed", "#0891b2", "#dc2626", "#059669"];

function hashCode(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) - hash) + str.charCodeAt(i);
    hash |= 0;
  }
  return hash;
}

async function ensurePasswordResetTokensTable() {
  await pool.query('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"');
  await pool.query(`
    CREATE TABLE IF NOT EXISTS password_reset_tokens (
      id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
      user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
      token TEXT NOT NULL,
      expires_at TIMESTAMP NOT NULL,
      is_active BOOLEAN DEFAULT TRUE,
      created_at TIMESTAMP DEFAULT NOW()
    )
  `);
}



// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  LOGIN
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
export const login = async (req, res) => {
  const { email, password, role } = req.body;
  if (!email) return res.status(400).json({ message: "Email is required" });
  if (!password) return res.status(400).json({ message: "Password is required" });

  try {
    const result = await pool.query(
      `SELECT user_id, company_id, full_name, email, password_hash, role, is_active, failed_attempts, lock_until
       FROM users
       WHERE email = $1`,
      [email]
    );

    if (result.rows.length === 0) {
      return res.status(401).json({ 
        message: "Invalid email or password. 3 attempts left.", 
        attemptsLeft: 3 
      });
    }

    const user = result.rows[0];

    // Check lockout
    if (user.lock_until && new Date(user.lock_until) > new Date()) {
      const remainingMs = new Date(user.lock_until).getTime() - Date.now();
      const remainingMins = Math.ceil(remainingMs / 60000);
      return res.status(401).json({ 
        message: `Too many failed attempts. Try again after ${remainingMins} minute${remainingMins !== 1 ? 's' : ''}.`,
        locked: true,
        attemptsLeft: 0
      });
    }

    // Verify password
    let isMatch = false;
    if (user.password_hash) {
      isMatch = await bcrypt.compare(password, user.password_hash);
    }

    if (!isMatch) {
      const newAttempts = (user.failed_attempts || 0) + 1;
      let updateQuery = "UPDATE users SET failed_attempts = $1";
      const params = [newAttempts, user.user_id];
      
      let lockoutMessage = "";
      let attemptsLeft = 0;

      if (newAttempts >= 3) {
        updateQuery += ", lock_until = NOW() + INTERVAL '5 minutes'";
        lockoutMessage = "Too many failed attempts. Try again after 5 minutes.";
        attemptsLeft = 0;
      } else {
        attemptsLeft = 3 - newAttempts;
        lockoutMessage = `Invalid email or password. ${attemptsLeft} attempt${attemptsLeft !== 1 ? 's' : ''} left.`;
      }
      
      updateQuery += " WHERE user_id = $2";
      await pool.query(updateQuery, params);
      
      return res.status(401).json({ 
        message: lockoutMessage, 
        attemptsLeft: attemptsLeft 
      });
    }

    // Check active status
    if (!user.is_active) {
      return res.status(403).json({
        message: "Account is pending approval or inactive. Please contact an administrator.",
        pending: true,
      });
    }

    // If admin login requested but user is not admin
    if (role === "admin" && user.role !== "admin") {
      return res.status(403).json({ message: "This account is not authorized for admin access" });
    }

    // SUCCESS - Reset attempts and update last_login only after all access checks pass
    await pool.query(
      "UPDATE users SET failed_attempts = 0, lock_until = NULL, last_login = NOW() WHERE user_id = $1", 
      [user.user_id]
    );

    const token = jwt.sign(
      { email, role: user.role, userId: user.user_id, companyId: user.company_id },
      process.env.JWT_SECRET || "secret",
      { expiresIn: "1d" }
    );

    logEmployeeLogin(user.user_id, user.full_name, user.email).catch((logErr) => {
      console.error("Login activity log error:", logErr);
    });

    res.json({
      token,
      role: user.role,
      name: user.full_name,
      email,
      userId: user.user_id,
      companyId: user.company_id,
    });
  } catch (err) {
    console.error("Login error:", err);
    res.status(500).json({ message: "Server error" });
  }
};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  GET ALL USERS (Admin)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
export const getAllUsers = async (req, res) => {
  const { role: roleFilter } = req.query;
  try {
    let query = `
      SELECT 
        u.user_id, u.company_id, u.full_name, u.email, u.role, u.department, u.designation, u.is_active, u.created_at,
        COALESCE(count_data.count, 0)::int AS datasets_count
      FROM users u
      LEFT JOIN (
        SELECT p.user_id, COUNT(d.dataset_id) as count
        FROM permissions p
        JOIN datasets d ON p.dataset_id = d.dataset_id
        WHERE p.can_view = TRUE AND d.upload_status != 'failed'
        GROUP BY p.user_id
      ) count_data ON u.user_id = count_data.user_id
    `;
    const params = [];

    if (roleFilter && roleFilter !== "all") {
      query += ` WHERE u.role = $1`;
      params.push(roleFilter);
    }

    query += ` ORDER BY u.created_at DESC`;

    const result = await pool.query(query, params);

    res.json({ success: true, users: formatUsers(result.rows), count: result.rows.length });
  } catch (err) {
    console.error("getAllUsers error:", err);
    res.status(500).json({ message: "Server error" });
  }
};

function formatUsers(rows) {
  return rows.map(u => ({
    ...u,
    initials: generateInitials(u.full_name || u.email),
    color: AVATAR_COLORS[Math.abs(hashCode(u.email)) % AVATAR_COLORS.length],
  }));
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  CREATE USER (Admin)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
export const createUser = async (req, res) => {
  const {
    full_name,
    email,
    password,
    role = "employee",
    department = "",
    designation = "",
    is_active = true,
  } = req.body;

  const normalizedRole = String(role || "").toLowerCase();
  const validRoles = ["admin", "employee"];

  if (!full_name || !email || !password) {
    return res.status(400).json({ success: false, message: "full_name, email, and password are required" });
  }

  if (!validRoles.includes(normalizedRole)) {
    return res.status(400).json({ success: false, message: `Invalid role. Must be one of: ${validRoles.join(", ")}` });
  }

  if (String(password).length < 6) {
    return res.status(400).json({ success: false, message: "Password must be at least 6 characters long" });
  }

  try {
    const existing = await pool.query("SELECT user_id FROM users WHERE email = $1", [email]);
    if (existing.rows.length > 0) {
      return res.status(409).json({ success: false, message: "A user with this email already exists" });
    }

    const passwordHash = await bcrypt.hash(password, SALT_ROUNDS);
    const companyId = req.user?.company_id;

    if (!companyId) {
      return res.status(400).json({ success: false, message: "Unable to determine company for the current admin" });
    }

    const result = await pool.query(
      `INSERT INTO users (company_id, full_name, email, password_hash, role, department, designation, is_active)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
       RETURNING user_id, company_id, full_name, email, role, department, designation, is_active, created_at`,
      [
        companyId,
        full_name.trim(),
        email.trim().toLowerCase(),
        passwordHash,
        normalizedRole,
        department?.trim() || null,
        designation?.trim() || null,
        Boolean(is_active),
      ]
    );

    return res.status(201).json({
      success: true,
      message: "Employee created successfully",
      user: formatUsers(result.rows)[0],
    });
  } catch (err) {
    console.error("createUser error:", err);
    return res.status(500).json({ success: false, message: "Server error" });
  }
};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  UPDATE USER ROLE (Admin)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
export const updateUserRole = async (req, res) => {
  const { email } = req.params;
  const { role } = req.body;

  const validRoles = ["admin", "employee"];
  if (!validRoles.includes(role)) {
    return res.status(400).json({ success: false, message: `Invalid role. Must be one of: ${validRoles.join(", ")}` });
  }

  try {
    const result = await pool.query(
      "UPDATE users SET role = $1 WHERE email = $2 RETURNING user_id",
      [role, email]
    );
    if (result.rows.length === 0) return res.status(404).json({ success: false, message: "User not found" });

    res.json({ success: true, message: `Role updated to ${role}` });
  } catch (err) {
    console.error("updateUserRole error:", err);
    res.status(500).json({ message: "Server error" });
  }
};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  UPDATE USER STATUS (Admin)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
export const updateUserStatus = async (req, res) => {
  const { email } = req.params;
  const { status } = req.body;

  const validStatuses = ["active", "inactive"];
  if (!validStatuses.includes(status)) {
    return res.status(400).json({ success: false, message: `Invalid status. Must be one of: ${validStatuses.join(", ")}` });
  }

  try {
    const result = await pool.query(
      "UPDATE users SET is_active = $1 WHERE email = $2 RETURNING user_id",
      [status === "active", email]
    );
    if (result.rows.length === 0) return res.status(404).json({ success: false, message: "User not found" });

    res.json({ success: true, message: `Status updated to ${status}` });
  } catch (err) {
    console.error("updateUserStatus error:", err);
    res.status(500).json({ message: "Server error" });
  }
};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  DELETE USER (Admin)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
export const deleteUser = async (req, res) => {
  const { email } = req.params;
  try {
    const result = await pool.query("DELETE FROM users WHERE email = $1 RETURNING user_id", [email]);
    if (result.rows.length === 0) return res.status(404).json({ success: false, message: "User not found" });
    res.json({ success: true, message: "User deleted" });
  } catch (err) {
    console.error("deleteUser error:", err);
    res.status(500).json({ message: "Server error" });
  }
};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  USER STATS (Admin Dashboard)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
export const getUserStats = async (req, res) => {
  try {
    const total    = await pool.query("SELECT COUNT(*)::int AS count FROM users");
    const active   = await pool.query("SELECT COUNT(*)::int AS count FROM users WHERE is_active = true");
    const inactive = await pool.query("SELECT COUNT(*)::int AS count FROM users WHERE is_active = false");
    const byRole   = await pool.query("SELECT role, COUNT(*)::int AS count FROM users GROUP BY role");

    res.json({
      success: true,
      stats: {
        total:    total.rows[0].count,
        active:   active.rows[0].count,
        inactive: inactive.rows[0].count,
        byRole:   Object.fromEntries(byRole.rows.map(r => [r.role, r.count])),
      },
    });
  } catch (err) {
    console.error("getUserStats error:", err);
    res.status(500).json({ message: "Server error" });
  }
};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  GET ME (Token-based identity)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
export const getMe = async (req, res) => {
  const token = req.headers.authorization?.replace("Bearer ", "");
  if (!token) return res.status(401).json({ message: "No token provided" });

  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET || "secret");
    const result = await pool.query(
      `SELECT user_id, company_id, full_name, email, role, department, designation, is_active, created_at
       FROM users WHERE email = $1`,
      [decoded.email]
    );
    if (result.rows.length === 0) return res.status(404).json({ message: "User not found" });

    const user = result.rows[0];
    res.json({
      success: true,
      user: {
        ...user,
        initials: generateInitials(user.full_name || user.email),
      },
    });
  } catch (err) {
    res.status(401).json({ message: "Invalid token" });
  }
};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  GET PENDING USERS (Admin — approval queue)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
export const getPendingUsers = async (req, res) => {
  try {
    const result = await pool.query(`
      SELECT user_id, company_id, full_name, email, role, department, designation, is_active, created_at
      FROM users
      WHERE is_active = false
      ORDER BY created_at DESC
    `);
    res.json({
      success: true,
      users: formatUsers(result.rows).map(u => ({ ...u, status: "pending" })),
      count: result.rows.length,
    });
  } catch (err) {
    console.error("getPendingUsers error:", err);
    res.status(500).json({ message: "Server error" });
  }
};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  APPROVE USER (Admin — activate or reject)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
export const approveUser = async (req, res) => {
  const { email } = req.params;
  const { approved } = req.body;

  try {
    const user = await pool.query("SELECT user_id FROM users WHERE email = $1", [email]);
    if (user.rows.length === 0) {
      return res.status(404).json({ success: false, message: "User not found" });
    }

    if (approved) {
      await pool.query("UPDATE users SET is_active = true WHERE email = $1", [email]);
      res.json({ success: true, message: "User approved and activated" });
    } else {
      await pool.query("DELETE FROM users WHERE email = $1", [email]);
      res.json({ success: true, message: "User registration rejected and removed" });
    }
  } catch (err) {
    console.error("approveUser error:", err);
    res.status(500).json({ message: "Server error" });
  }
};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  FORGOT PASSWORD
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
export const forgotPassword = async (req, res) => {
  const { email } = req.body;
  if (!email) {
    return res.status(400).json({ message: "Email is required" });
  }

  try {
    await ensurePasswordResetTokensTable();

    const userResult = await pool.query("SELECT user_id, lock_until FROM users WHERE email = $1", [email]);
    if (userResult.rows.length > 0) {
      const user = userResult.rows[0];

      // Check lockout
      if (user.lock_until && new Date(user.lock_until) > new Date()) {
        return res.status(401).json({ message: "Please wait before requesting password reset." });
      }

      const resetToken = uuidv4();
      
      // Invalidate existing active tokens for this user
      await pool.query("UPDATE password_reset_tokens SET is_active = false WHERE user_id = $1", [user.user_id]);

      // Insert new token with 15 mins expiry
      const expiresAt = new Date(Date.now() + 15 * 60 * 1000);
      await pool.query(
        "INSERT INTO password_reset_tokens (user_id, token, expires_at) VALUES ($1, $2, $3)",
        [user.user_id, resetToken, expiresAt]
      );

      // Send email
      await sendResetEmail(email, resetToken);
    }
    
    // Always return generic response to prevent email enumeration
    res.json({ message: "If an account exists, a reset link has been sent to your email." });
  } catch (err) {
    console.error("forgotPassword error:", err);
    res.status(500).json({ message: "Server error" });
  }
};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  VALIDATE RESET TOKEN
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
export const validateResetToken = async (req, res) => {
  const { token } = req.query;
  if (!token) return res.status(400).json({ message: "Token is required" });

  try {
    await ensurePasswordResetTokensTable();

    const tokenResult = await pool.query(
      "SELECT * FROM password_reset_tokens WHERE token = $1 AND is_active = true AND expires_at > NOW()",
      [token]
    );

    if (tokenResult.rows.length === 0) {
      return res.status(400).json({ valid: false, message: "Invalid or expired reset token" });
    }

    res.json({ valid: true });
  } catch (err) {
    console.error("validateResetToken error:", err);
    res.status(500).json({ message: "Server error" });
  }
};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  RESET PASSWORD
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
export const resetPassword = async (req, res) => {
  const { token, newPassword } = req.body;
  if (!token || !newPassword) {
    return res.status(400).json({ message: "Token and new password are required" });
  }

  try {
    await ensurePasswordResetTokensTable();

    const tokenResult = await pool.query(
      "SELECT user_id, id FROM password_reset_tokens WHERE token = $1 AND is_active = true AND expires_at > NOW()",
      [token]
    );

    if (tokenResult.rows.length === 0) {
      return res.status(400).json({ message: "Invalid or expired reset token" });
    }

    const userId = tokenResult.rows[0].user_id;
    const passwordHash = await bcrypt.hash(newPassword, SALT_ROUNDS);

    await pool.query("UPDATE users SET password_hash = $1 WHERE user_id = $2", [passwordHash, userId]);
    
    // Invalidate the token natively
    await pool.query("UPDATE password_reset_tokens SET is_active = false WHERE token = $1", [token]);
    
    res.json({ message: "Password updated successfully" });
  } catch (err) {
    console.error("resetPassword error:", err);
    res.status(500).json({ message: "Server error" });
  }
};
