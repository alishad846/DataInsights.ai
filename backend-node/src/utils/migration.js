import { pool } from "../config/db.js";

async function run() {
  try {
    await pool.query(`ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS generated_code TEXT;`);
    console.log("Added generated_code column");
  } catch (e) {
    console.error(e.message);
  }
  
  try {
    await pool.query(`ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS error_msg TEXT;`);
    console.log("Added error_msg column");
  } catch (e) {
    console.error(e.message);
  }

  try {
    await pool.query(`ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS rows_returned INTEGER;`);
    console.log("Added rows_returned column");
  } catch (e) {
    console.error(e.message);
  }
  
  process.exit(0);
}

run();
