import pg from "pg";
import dotenv from "dotenv";
import fs from "fs/promises";
import path from "path";

dotenv.config();

const { Pool } = pg;

const pool = new Pool({
    host: process.env.DB_HOST || "localhost",
    port: Number(process.env.DB_PORT) || 5432,
    database: process.env.DB_NAME,
    user: process.env.DB_USER || "postgres",
    password: process.env.DB_PASSWORD || "",
});

const ensureUploadDirectories = async () => {
    const uploadsRoot = path.resolve(process.cwd(), "..", "uploads");
    const dirs = [
        uploadsRoot,
        path.join(uploadsRoot, "raw"),
        path.join(uploadsRoot, "temp"),
        path.join(uploadsRoot, "cleaned"),
    ];

    await Promise.all(dirs.map((dir) => fs.mkdir(dir, { recursive: true })));
};

const ensureRuntimeSchema = async (client) => {
    await client.query('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"');
    await client.query(`
        ALTER TABLE IF EXISTS users
        ADD COLUMN IF NOT EXISTS failed_attempts INT DEFAULT 0
    `);
    await client.query(`
        ALTER TABLE IF EXISTS users
        ADD COLUMN IF NOT EXISTS lock_until TIMESTAMP
    `);
    await client.query(`
        CREATE TABLE IF NOT EXISTS permission_requests (
            request_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            company_id UUID NOT NULL REFERENCES companies(company_id) ON DELETE CASCADE,
            user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            dataset_id UUID NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
            permission_type TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            requested_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        )
    `);
    await client.query(`
        CREATE TABLE IF NOT EXISTS password_reset_tokens (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            token TEXT NOT NULL,
            expires_at TIMESTAMP NOT NULL,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT NOW()
        )
    `);
    await client.query(`
        CREATE TABLE IF NOT EXISTS activity_logs (
            log_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            company_id UUID REFERENCES companies(company_id),
            user_id UUID REFERENCES users(user_id),
            dataset_id UUID REFERENCES datasets(dataset_id),
            activity_type TEXT,
            activity_description TEXT,
            module_name TEXT,
            status TEXT,
            ip_address INET,
            timestamp TIMESTAMP DEFAULT NOW()
        )
    `);
    await client.query(`
        CREATE TABLE IF NOT EXISTS query_logs (
            log_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            company_id UUID REFERENCES companies(company_id),
            user_id UUID REFERENCES users(user_id),
            dataset_id UUID REFERENCES datasets(dataset_id),
            query_text TEXT,
            query_type TEXT,
            execution_time_ms INTEGER,
            status TEXT,
            rows_returned INTEGER,
            generated_code TEXT,
            error_msg TEXT,
            timestamp TIMESTAMP DEFAULT NOW()
        )
    `);
    await client.query(`
        ALTER TABLE IF EXISTS query_logs
        ADD COLUMN IF NOT EXISTS generated_code TEXT
    `);
    await client.query(`
        ALTER TABLE IF EXISTS query_logs
        ADD COLUMN IF NOT EXISTS error_msg TEXT
    `);
};

const connectDB = async () => {
    let client;

    try {
        client = await pool.connect();
        await ensureRuntimeSchema(client);
        await ensureUploadDirectories();
        console.log("PostgreSQL Connected");
        client.release();
    } catch (error) {
        if (client) {
            client.release();
        }
        console.error("PostgreSQL Connection Error:", error.message);
        process.exit(1);
    }
};

export default connectDB;
export { pool };
