import { pool } from "../config/db.js";

const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

async function resolveUserId(userId, userEmail) {
  if (userId && UUID_PATTERN.test(userId)) return userId;
  if (userEmail) {
    const byEmail = await pool.query("SELECT user_id FROM users WHERE email = $1", [userEmail]);
    return byEmail.rows[0]?.user_id || null;
  }
  if (userId && typeof userId === "string" && userId.includes("@")) {
    const byEmailLikeId = await pool.query("SELECT user_id FROM users WHERE email = $1", [userId]);
    return byEmailLikeId.rows[0]?.user_id || null;
  }
  return null;
}

async function resolveCompanyId(userId) {
  if (!userId) return null;
  const r = await pool.query("SELECT company_id FROM users WHERE user_id = $1", [userId]);
  return r.rows[0]?.company_id || null;
}

export const logActivity = async ({
  userId,
  userName,
  userEmail,
  eventType,
  eventDescription,
  datasetId,
  datasetName,
  detail,
  status = "ok",
  moduleName = "APP",
}) => {
  try {
    const resolvedUserId = await resolveUserId(userId, userEmail);
    const companyId = await resolveCompanyId(resolvedUserId);
    const description = detail || eventDescription || "";
    await pool.query(
      `INSERT INTO activity_logs
        (company_id, user_id, dataset_id, activity_type, activity_description, module_name, status)
       VALUES ($1, $2, $3, $4, $5, $6, $7)`,
      [companyId, resolvedUserId, datasetId || null, eventType, description, moduleName, status]
    );
    return { success: true };
  } catch (error) {
    console.error("Error logging activity:", error.message);
    return { success: false, error: error.message };
  }
};

export const getActivityLogs = async (req, res) => {
  try {
    const { employee, event, status, startDate, endDate, limit = 100, includeSessions, dataset } = req.query;

    let query = `
      SELECT
        al.log_id,
        al.timestamp AS created_at,
        al.activity_type AS event_type,
        al.activity_description AS event_description,
        al.status,
        al.module_name,
        u.full_name AS user_name,
        u.email AS user_email,
        d.dataset_name,
        al.activity_description AS detail
      FROM activity_logs al
      LEFT JOIN users u ON al.user_id = u.user_id
      LEFT JOIN datasets d ON al.dataset_id = d.dataset_id
      WHERE 1=1
    `;
    const params = [];
    let paramIndex = 1;

    if (employee && employee !== "All") {
      query += ` AND (u.full_name = $${paramIndex} OR u.email = $${paramIndex})`;
      params.push(employee);
      paramIndex++;
    }

    if (event && event !== "all") {
      query += ` AND al.activity_type = $${paramIndex}`;
      params.push(event.toUpperCase());
      paramIndex++;
    }

    if (status && status !== "all") {
      query += ` AND al.status = $${paramIndex}`;
      params.push(status);
      paramIndex++;
    }

    if (dataset && dataset !== "all") {
      query += ` AND d.dataset_name = $${paramIndex}`;
      params.push(dataset);
      paramIndex++;
    }

    if (startDate) {
      query += ` AND al.timestamp >= $${paramIndex}`;
      params.push(startDate);
      paramIndex++;
    }

    if (endDate) {
      query += ` AND al.timestamp <= $${paramIndex}`;
      params.push(endDate);
      paramIndex++;
    }

    query += ` ORDER BY al.timestamp DESC LIMIT $${paramIndex}`;
    params.push(Number(limit));

    const result = await pool.query(query, params);
    let logs = result.rows;

    if (includeSessions === "true") {
      try {
        const sessionResult = await pool.query(`
          SELECT user_email, login_time, logout_time, total_duration_seconds
          FROM user_sessions
          ORDER BY login_time DESC
          LIMIT 50
        `);
        const sessions = sessionResult.rows.map((s) => {
          const duration = s.logout_time
            ? s.total_duration_seconds || 0
            : Math.floor((Date.now() - new Date(s.login_time).getTime()) / 1000);
          return {
            log_id: `session-${s.user_email}-${s.login_time}`,
            user_email: s.user_email,
            user_name: s.user_email?.split("@")[0] || "Unknown",
            event_type: "SESSION",
            event_description: s.logout_time ? "Session completed" : "Currently active",
            detail: s.logout_time
              ? `Logged in at ${new Date(s.login_time).toLocaleTimeString()}. Logged out at ${new Date(s.logout_time).toLocaleTimeString()}`
              : `Logged in at ${new Date(s.login_time).toLocaleTimeString()}. Still active`,
            status: s.logout_time ? "completed" : "active",
            duration_seconds: duration,
            created_at: s.login_time,
            dataset_name: null,
          };
        });
        logs = [...logs, ...sessions];
        logs.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
      } catch {
        /* user_sessions optional — schema.txt does not define it */
      }
    }

    res.json({
      success: true,
      logs,
      count: logs.length,
    });
  } catch (error) {
    console.error("Error fetching activity logs:", error.message);
    res.status(500).json({ success: false, message: "Failed to fetch activity logs" });
  }
};

export const getActivityStats = async (req, res) => {
  try {
    const stats = await pool.query(`
      SELECT activity_type AS event_type, status, COUNT(*)::int as count
      FROM activity_logs
      GROUP BY activity_type, status
    `);

    const users = await pool.query(`
      SELECT DISTINCT u.full_name AS user_name, u.email AS user_email
      FROM activity_logs al
      JOIN users u ON al.user_id = u.user_id
      WHERE u.full_name IS NOT NULL
      ORDER BY u.full_name
    `);

    const uniqueEvents = await pool.query(`
      SELECT DISTINCT activity_type AS event_type FROM activity_logs ORDER BY activity_type
    `);

    res.json({
      success: true,
      stats: stats.rows,
      users: users.rows,
      events: uniqueEvents.rows.map((r) => r.event_type),
    });
  } catch (error) {
    console.error("Error fetching activity stats:", error.message);
    res.status(500).json({ success: false, message: "Failed to fetch activity stats" });
  }
};

/** Query volume from query_logs (schema.txt) for admin charts */
export const getQueryVolume = async (req, res) => {
  try {
    const days = Math.min(30, Math.max(1, parseInt(req.query.days || "7", 10) || 7));
    const result = await pool.query(
      `
      SELECT (timestamp AT TIME ZONE 'UTC')::date AS d, COUNT(*)::int AS c
      FROM query_logs
      WHERE (timestamp AT TIME ZONE 'UTC')::date >= (NOW() AT TIME ZONE 'UTC')::date - ($1::int - 1)
      GROUP BY 1
      ORDER BY 1 ASC
    `,
      [days]
    );

    const countsByDate = Object.fromEntries(
      result.rows.map((r) => {
        const d = r.d;
        const key = d instanceof Date ? d.toISOString().slice(0, 10) : String(d).split("T")[0];
        return [key, r.c];
      })
    );

    const slots = [];
    const now = new Date();
    for (let i = days - 1; i >= 0; i--) {
      const dt = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate() - i));
      const key = dt.toISOString().slice(0, 10);
      const dayLabel = dt.toLocaleDateString("en-US", { weekday: "short", timeZone: "UTC" });
      slots.push({ dayLabel, count: countsByDate[key] ?? 0 });
    }

    const counts = slots.map((s) => s.count);
    const dayLabels = slots.map((s) => s.dayLabel);
    const max = Math.max(1, ...counts);

    res.json({
      success: true,
      days: dayLabels,
      counts,
      normalized: counts.map((c) => Math.round((c / max) * 100)),
    });
  } catch (error) {
    console.error("getQueryVolume error:", error.message);
    res.status(500).json({ success: false, message: "Failed to fetch query volume" });
  }
};

export const logEmployeeLogin = async (userId, userName, userEmail) => {
  return logActivity({
    userId,
    userName,
    userEmail,
    eventType: "LOGIN",
    eventDescription: `${userName || userEmail || "User"} logged in`,
    status: "ok",
    moduleName: "AUTH",
  });
};

export const logEmployeeLogout = async (userId, userName, userEmail, durationSeconds) => {
  return logActivity({
    userId,
    userName,
    userEmail,
    eventType: "LOGOUT",
    eventDescription: `Employee logged out${durationSeconds ? ` (Active for ${durationSeconds}s)` : ""}`,
    status: "ok",
    moduleName: "AUTH",
  });
};


export const logCleaningActivity = async (userId, userName, userEmail, datasetId, datasetName, type, status, detail) => {
  return logActivity({
    userId,
    userName,
    userEmail,
    eventType: type || "CLEAN", // CLEAN_START, CLEAN_DONE, CLEAN_PAUSE
    eventDescription: detail || `Data cleaning activity: ${type} on ${datasetName}`,
    datasetId,
    datasetName,
    detail,
    status,
    moduleName: "PIPELINE",
  });
};


export const logQueryActivity = async (userId, userName, userEmail, datasetId, datasetName, query, status, durationSeconds) => {
  return logActivity({
    userId,
    userName,
    userEmail,
    eventType: "QUERY",
    eventDescription: durationSeconds != null ? `Query (${durationSeconds}ms)` : `Query on ${datasetName}`,
    datasetId,
    datasetName,
    detail: query,
    status,
    moduleName: "CHAT",
  });
};

export const logVisualizationActivity = async (userId, userName, userEmail, datasetId, datasetName, status, detail) => {
  return logActivity({
    userId,
    userName,
    userEmail,
    eventType: "VISUALIZE",
    eventDescription: `Visualization for ${datasetName}`,
    datasetId,
    datasetName,
    detail,
    status,
    moduleName: "DATASET",
  });
};

export const logViewSummaryActivity = async (userId, userName, userEmail, datasetId, datasetName, status) => {
  return logActivity({
    userId,
    userName,
    userEmail,
    eventType: "VIEW_SUMMARY",
    eventDescription: `Viewed summary for ${datasetName}`,
    datasetId,
    datasetName,
    detail: "Employee viewed data summary",
    status,
    moduleName: "DATASET",
  });
};

export const logDatasetAccess = async (userId, userName, userEmail, datasetId, datasetName, status) => {
  return logActivity({
    userId,
    userName,
    userEmail,
    eventType: "ACCESS_DATASET",
    eventDescription: `Accessed dataset ${datasetName}`,
    datasetId,
    datasetName,
    detail: `Viewed dataset: ${datasetName}`,
    status,
    moduleName: "DATASET",
  });
};
export const logChatActivity = async (userId, userName, userEmail, datasetId, datasetName, type, detail) => {
  return logActivity({
    userId,
    userName,
    userEmail,
    eventType: type, // CHAT_START, CHAT_END
    eventDescription: detail || `Chat session ${type === 'CHAT_START' ? 'started' : 'ended'} on ${datasetName}`,
    datasetId,
    datasetName,
    detail,
    status: "ok",
    moduleName: "CHAT",
  });
};

export const logPermissionActivity = async (userId, userName, userEmail, datasetId, datasetName, type, detail) => {
  return logActivity({
    userId,
    userName,
    userEmail,
    eventType: type, // PERM_REQUEST, PERM_DENIED
    eventDescription: detail || `Permission activity: ${type}`,
    datasetId,
    datasetName,
    detail,
    status: type === 'PERM_DENIED' ? "failed" : "ok",
    moduleName: "PERMISSIONS",
  });
};
