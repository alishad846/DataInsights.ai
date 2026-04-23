import { pool } from "../config/db.js";
import { logActivity } from "./activityController.js";

export const submitPermissionRequest = async (req, res) => {
  try {
    const { datasetId, permissionType } = req.body;
    const userId = req.user.id;
    const email = req.user.email;

    if (!datasetId || !permissionType) {

      return res.status(400).json({ success: false, message: "Dataset ID and permission type are required." });
    }

    // Check if a pending request already exists
    const existing = await pool.query(
      "SELECT 1 FROM permission_requests WHERE user_id = $1 AND dataset_id = $2 AND permission_type = $3 AND status = 'pending'",
      [userId, datasetId, permissionType]
    );

    if (existing.rows.length > 0) {
      return res.status(400).json({ success: false, message: "A request for this permission is already pending." });
    }

    const companyRes = await pool.query("SELECT company_id FROM users WHERE user_id = $1", [userId]);
    const companyId = companyRes.rows[0]?.company_id;

    const result = await pool.query(
      "INSERT INTO permission_requests (company_id, user_id, dataset_id, permission_type) VALUES ($1, $2, $3, $4) RETURNING request_id",
      [companyId, userId, datasetId, permissionType]
    );

    // Fetch dataset name for logging
    const dsRes = await pool.query("SELECT dataset_name FROM datasets WHERE dataset_id = $1", [datasetId]);
    const dsName = dsRes.rows[0]?.dataset_name || "Unknown Dataset";

    // Log the request
    await logActivity({
      userId,
      userEmail: email,
      eventType: "PERM_REQUEST",
      datasetId,
      datasetName: dsName,
      detail: `Employee requested ${permissionType} access for ${dsName}`,
      moduleName: "PERMISSIONS"
    });

    res.json({ success: true, requestId: result.rows[0].request_id, message: "Permission request submitted successfully." });
  } catch (err) {
    console.error("Error submitting permission request:", err);
    res.status(500).json({ success: false, message: "Internal server error." });
  }
};

export const getPendingRequests = async (req, res) => {
  try {
    const query = `
      SELECT 
        pr.request_id,
        pr.permission_type,
        pr.status,
        pr.requested_at,
        u.full_name as user_name,
        u.email as user_email,
        d.dataset_name,
        d.dataset_id
      FROM permission_requests pr
      JOIN users u ON pr.user_id = u.user_id
      JOIN datasets d ON pr.dataset_id = d.dataset_id
      WHERE pr.status = 'pending'
      ORDER BY pr.requested_at DESC
    `;
    const result = await pool.query(query);
    res.json({ success: true, requests: result.rows });
  } catch (err) {
    console.error("Error fetching pending requests:", err);
    res.status(500).json({ success: false, message: "Internal server error." });
  }
};

export const resolvePermissionRequest = async (req, res) => {
  try {
    const { requestId, status } = req.body; // status: 'accepted' or 'rejected'
    
    if (!['accepted', 'rejected'].includes(status)) {
      return res.status(400).json({ success: false, message: "Invalid status." });
    }

    const reqRes = await pool.query(
      "SELECT user_id, dataset_id, permission_type, company_id FROM permission_requests WHERE request_id = $1",
      [requestId]
    );
    
    if (reqRes.rows.length === 0) {
      return res.status(404).json({ success: false, message: "Request not found." });
    }

    const { user_id, dataset_id, permission_type, company_id } = reqRes.rows[0];

    // Update request status
    await pool.query(
      "UPDATE permission_requests SET status = $1, updated_at = NOW() WHERE request_id = $2",
      [status, requestId]
    );

    if (status === 'accepted') {
      // For DATASET_ACCESS, the admin manually assigns from the Datasets page after redirection.
      // So we skip the auto-upsert for this type.
      if (permission_type !== 'DATASET_ACCESS') {
        const column = permission_type; // e.g. 'can_edit'
        
        // Ensure column is simple and valid (basic security check)
        if (!['can_view', 'can_edit', 'can_query', 'can_delete'].includes(column)) {
          return res.status(400).json({ success: false, message: "Invalid permission type." });
        }

        const checkPerm = await pool.query(
          "SELECT 1 FROM permissions WHERE user_id = $1 AND dataset_id = $2",
          [user_id, dataset_id]
        );

        if (checkPerm.rows.length > 0) {
          await pool.query(
            `UPDATE permissions SET ${column} = TRUE, updated_at = NOW() WHERE user_id = $1 AND dataset_id = $2`,
            [user_id, dataset_id]
          );
        } else {
          await pool.query(
            `INSERT INTO permissions (company_id, user_id, dataset_id, ${column}) VALUES ($1, $2, $3, TRUE)`,
            [company_id, user_id, dataset_id]
          );
        }
      }
    }


    // Log the resolution
    const adminId = req.user.user_id;
    const adminEmail = req.user.email;
    
    await logActivity({
      userId: adminId,
      userEmail: adminEmail,
      eventType: status === 'accepted' ? "PERM_GRANTED" : "PERM_REJECTED",
      datasetId: dataset_id,
      detail: `Admin ${status} ${permission_type} request for user ${user_id}`,
      moduleName: "PERMISSIONS"
    });

    res.json({ success: true, message: `Request ${status} successfully.` });
  } catch (err) {
    console.error("Error resolving permission request:", err);
    res.status(500).json({ success: false, message: "Internal server error." });
  }
};

export const getPermissionBadgeCounts = async (req, res) => {
  try {
    const userCountRes = await pool.query("SELECT COUNT(*) FROM users WHERE is_active = FALSE");
    const permCountRes = await pool.query("SELECT COUNT(*) FROM permission_requests WHERE status = 'pending'");
    
    res.json({
      success: true,
      pendingUsers: parseInt(userCountRes.rows[0].count),
      pendingPermissions: parseInt(permCountRes.rows[0].count),
      total: parseInt(userCountRes.rows[0].count) + parseInt(permCountRes.rows[0].count)
    });
  } catch (err) {
    console.error("Error getting badge counts:", err);
    res.status(500).json({ success: false, message: "Internal server error." });
  }
};
