import dotenv from "dotenv";
import pg from "pg";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

dotenv.config();

const { Pool, Client } = pg;

const pool = new Pool({
    host: process.env.DB_HOST || "localhost",
    port: Number(process.env.DB_PORT) || 5432,
    database: "postgres",
    user: process.env.DB_USER || "postgres",
    password: process.env.DB_PASSWORD || "",
});

const runSchema = async () => {
    // We connect to 'postgres' first to drop/create the DB if needed
    const adminPool = new Pool({
        host: process.env.DB_HOST || "localhost",
        port: Number(process.env.DB_PORT) || 5432,
        database: "postgres",
        user: process.env.DB_USER || "postgres",
        password: process.env.DB_PASSWORD || "",
    });

    try {
        const adminClient = await adminPool.connect();
        const dbName = process.env.DB_NAME || "newdatainsights";

        console.log(`Checking database: ${dbName}...`);
        const dbCheck = await adminClient.query(
            "SELECT 1 FROM pg_database WHERE datname = $1",
            [dbName]
        );

        if (dbCheck.rows.length === 0) {
            await adminClient.query(`CREATE DATABASE ${dbName}`);
            console.log(`✅ Database ${dbName} created`);
        }

        adminClient.release();
        await adminPool.end();

        // Now connect to the target database
        const targetPool = new Pool({
            host: process.env.DB_HOST || "localhost",
            port: Number(process.env.DB_PORT) || 5432,
            database: dbName,
            user: process.env.DB_USER || "postgres",
            password: process.env.DB_PASSWORD || "",
        });

        const client = await targetPool.connect();

        // RUN SCHEMA (from root)
        console.log("Applying schema from database_schema.sql...");
        const schemaPath = path.join(process.cwd(), "..", "database_schema.sql");
        if (fs.existsSync(schemaPath)) {
            const schema = fs.readFileSync(schemaPath, "utf8");
            // Execute schema segments (splitting by semi-colon to handle large scripts if needed)
            await client.query(schema);
            console.log("✅ Schema applied successfully");
        } else {
            console.warn("⚠️ Warning: database_schema.sql not found at project root");
        }

        // RUN SEED DATA (if exists)
        const seedPath = path.join(process.cwd(), "src", "config", "seed_data.sql");
        if (fs.existsSync(seedPath)) {
            console.log("Loading snapshot data from seed_data.sql...");
            const seed = fs.readFileSync(seedPath, "utf8");
            await client.query(seed);
            console.log("✅ Data snapshot restored successfully");
        } else {
            console.log("ℹ️ No seed_data.sql found. Using clean schema.");
        }

        // Verification
        const result = await client.query(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        );
        console.log("\n🚀 Setup Complete. Tables ready:", result.rows.map(r => r.table_name).join(", "));

        client.release();
        await targetPool.end();
        process.exit(0);
    } catch (error) {
        console.error("❌ Initialization failed:", error.message);
        process.exit(1);
    }
};

runSchema();