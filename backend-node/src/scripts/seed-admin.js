import dotenv from "dotenv";
import pg from "pg";

dotenv.config();

const { Pool } = pg;

const pool = new Pool({
    host: process.env.DB_HOST || "localhost",
    port: Number(process.env.DB_PORT) || 5432,
    database: process.env.DB_NAME,
    user: process.env.DB_USER || "postgres",
    password: process.env.DB_PASSWORD,
});

const createAdmin = async () => {
    const client = await pool.connect();
    
    try {
        // Check if admin user exists
        const existing = await client.query("SELECT email FROM users WHERE email = 'admin@example.com'");
        if (existing.rows.length > 0) {
            console.log("Admin user already exists");
            process.exit(0);
        }
        
        // Hash password (using simple comparison for demo)
        // In production, use bcrypt
        const bcrypt = await import("bcryptjs");
        const passwordHash = await bcrypt.default.hash("admin123", 12);
        
        // Get admin role id
        const roleResult = await client.query("SELECT role_id FROM roles WHERE role_name = 'admin'");
        const adminRoleId = roleResult.rows[0]?.role_id;
        
        if (!adminRoleId) {
            console.error("Admin role not found in database. Run init-db.js first.");
            process.exit(1);
        }
        
        // Create admin user
        const result = await client.query(
            `INSERT INTO users (user_id, email, name, password_hash, company_id, is_active)
             VALUES (gen_random_uuid(), 'admin@example.com', 'Admin User', $1, '00000000-0000-0000-0000-000000000001', true)
             RETURNING user_id`,
            [passwordHash]
        );
        
        const userId = result.rows[0].user_id;
        
        // Assign admin role
        await client.query(
            "INSERT INTO user_roles (user_id, role_id) VALUES ($1, $2)",
            [userId, adminRoleId]
        );
        
        console.log("✅ Admin user created successfully!");
        console.log("   Email: admin@example.com");
        console.log("   Password: admin123");
        
    } catch (error) {
        console.error("❌ Error:", error.message);
        process.exit(1);
    } finally {
        client.release();
        await pool.end();
    }
};

createAdmin();