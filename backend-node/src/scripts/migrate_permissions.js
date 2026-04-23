import { pool } from "../config/db.js";

const migrate = async () => {
    try {
        console.log("🚀 Starting permission_requests migration...");
        
        const query = `
            CREATE TABLE IF NOT EXISTS permission_requests (
                request_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                company_id UUID NOT NULL REFERENCES companies(company_id) ON DELETE CASCADE,
                user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                dataset_id UUID NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
                permission_type TEXT NOT NULL,
                status TEXT DEFAULT 'pending', -- 'pending', 'accepted', 'rejected'
                requested_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            );
        `;
        
        await pool.query(query);
        console.log("✅ Table permission_requests created successfully.");
        
        process.exit(0);
    } catch (err) {
        console.error("❌ Migration failed:", err.message);
        process.exit(1);
    }
};

migrate();
