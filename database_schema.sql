-- Create new database (run this first, then connect to it)
-- CREATE DATABASE newdatainsights;

-- Connect to the new database (in psql: \c newdatainsights)
-- Then run the following table creation statements

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =========================
-- COMPANIES
-- =========================
CREATE TABLE IF NOT EXISTS companies (
    company_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    company_name TEXT NOT NULL,
    company_email TEXT,
    industry TEXT,
    subscription_plan TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE
);

-- =========================
-- USERS
-- =========================
CREATE TABLE IF NOT EXISTS users (
    user_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    company_id UUID NOT NULL REFERENCES companies(company_id) ON DELETE CASCADE,
    full_name TEXT,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT,
    role TEXT,
    department TEXT,
    designation TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    failed_attempts INT DEFAULT 0,
    lock_until TIMESTAMP,
    last_login TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- =========================
-- PASSWORD RESET TOKENS
-- =========================
CREATE TABLE IF NOT EXISTS password_reset_tokens (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    token TEXT NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- =========================
-- DATASETS
-- =========================
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    company_id UUID NOT NULL REFERENCES companies(company_id) ON DELETE CASCADE,
    uploaded_by UUID REFERENCES users(user_id),
    dataset_name TEXT NOT NULL,
    file_name TEXT,
    file_size BIGINT,
    schema_json JSONB,
    upload_status TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- =========================
-- PERMISSIONS
-- =========================
CREATE TABLE IF NOT EXISTS permissions (
    company_id UUID NOT NULL REFERENCES companies(company_id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    dataset_id UUID NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    can_view BOOLEAN DEFAULT TRUE,
    can_edit BOOLEAN DEFAULT FALSE,
    can_delete BOOLEAN DEFAULT FALSE,
    can_query BOOLEAN DEFAULT FALSE,
    granted_by UUID REFERENCES users(user_id),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (user_id, dataset_id)
);

-- =========================
-- ACTIVITY_LOGS
-- =========================
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
);

-- =========================
-- QUERY_LOGS
-- =========================
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
);

-- =========================
-- PERMISSION REQUESTS
-- =========================
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

-- =========================
-- INDEXES
-- =========================
CREATE INDEX IF NOT EXISTS idx_companies_company_id ON companies(company_id);
CREATE INDEX IF NOT EXISTS idx_users_company_id ON users(company_id);
CREATE INDEX IF NOT EXISTS idx_users_user_id ON users(user_id);
CREATE INDEX IF NOT EXISTS idx_datasets_company_id ON datasets(company_id);
CREATE INDEX IF NOT EXISTS idx_datasets_dataset_id ON datasets(dataset_id);
CREATE INDEX IF NOT EXISTS idx_permissions_user_id ON permissions(user_id);
CREATE INDEX IF NOT EXISTS idx_permissions_dataset_id ON permissions(dataset_id);
CREATE INDEX IF NOT EXISTS idx_activity_logs_company_id ON activity_logs(company_id);
CREATE INDEX IF NOT EXISTS idx_activity_logs_user_id ON activity_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_activity_logs_dataset_id ON activity_logs(dataset_id);
CREATE INDEX IF NOT EXISTS idx_activity_logs_timestamp ON activity_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_query_logs_company_id ON query_logs(company_id);
CREATE INDEX IF NOT EXISTS idx_query_logs_user_id ON query_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_query_logs_dataset_id ON query_logs(dataset_id);
CREATE INDEX IF NOT EXISTS idx_query_logs_timestamp ON query_logs(timestamp);







-- INSERT QUERY
-- ----------------------------------------------------------------------------------

BEGIN;

WITH company AS (
  INSERT INTO companies(company_name, company_email, industry, subscription_plan, is_active)
  VALUES ('explainableAI', 'contact@explainableai.com', 'AI/ML', 'pro', true)
  RETURNING company_id
),
users_data AS (
  INSERT INTO users(company_id, full_name, email, password_hash, role, department, designation, is_active, last_login)
  SELECT 
    (SELECT company_id FROM company),
    CASE WHEN num = 0 THEN 'Admin User' ELSE 'Employee ' || LPAD(num::text, 2, '0') END,
    CASE WHEN num = 0 THEN 'admin@explainableai.com' ELSE 'employee' || LPAD(num::text, 2, '0') || '@explainableai.com' END,
    CASE WHEN num = 0 THEN '$2b$12$EBPxdxc1nnPJifgzHQ./cuojUh9Y5JnEHs1r.ZteikOkprgcTR1da' ELSE '$2b$12$cuWg9qvPwO7R14ItjGyJoeRjv.mNTDVU./wRfAOr553NGojA9GRWi' END,
    CASE WHEN num = 0 THEN 'admin' ELSE 'employee' END,
    CASE WHEN num = 0 THEN 'Management' WHEN num <= 5 THEN 'Data Science' WHEN num <= 10 THEN 'Analytics' WHEN num <= 15 THEN 'Engineering' ELSE 'Operations' END,
    CASE WHEN num = 0 THEN 'System Admin' WHEN num <= 5 THEN 'Data Scientist' WHEN num <= 10 THEN 'Business Analyst' WHEN num <= 15 THEN 'Data Engineer' ELSE 'Ops Analyst' END,
    true, NOW()
  FROM generate_series(0, 20) as num
  RETURNING user_id, email
),
datasets_data AS (
  INSERT INTO datasets(company_id, uploaded_by, dataset_name, file_name, file_size, upload_status)
  SELECT 
    (SELECT company_id FROM company),
    CASE 
      WHEN ds_num IN (1, 2) THEN (SELECT user_id FROM users_data WHERE email = 'employee01@explainableai.com')
      WHEN ds_num = 3 THEN (SELECT user_id FROM users_data WHERE email = 'employee02@explainableai.com')
      WHEN ds_num = 4 THEN (SELECT user_id FROM users_data WHERE email = 'employee03@explainableai.com')
      WHEN ds_num = 5 THEN (SELECT user_id FROM users_data WHERE email = 'employee04@explainableai.com')
    END,
    'Sales Dataset ' || ds_num,
    'sales_data_' || ds_num || '.csv',
    (100000 + ds_num * 50000),
    'completed'
  FROM generate_series(1, 5) as ds_num
  RETURNING dataset_id, uploaded_by
),
permissions_data AS (
  INSERT INTO permissions(company_id, user_id, dataset_id, can_view, can_edit, can_delete, can_query, granted_by)
  SELECT 
    (SELECT company_id FROM company),
    u.user_id,
    d.dataset_id,
    true,
    CASE WHEN u.email = 'admin@explainableai.com' THEN true
         WHEN d.dataset_id = (SELECT dataset_id FROM datasets_data ORDER BY dataset_id LIMIT 1 OFFSET 0) AND u.email IN ('employee01@explainableai.com','employee02@explainableai.com','employee03@explainableai.com','employee04@explainableai.com','employee05@explainableai.com') THEN true
         WHEN d.dataset_id = (SELECT dataset_id FROM datasets_data ORDER BY dataset_id LIMIT 1 OFFSET 2) AND u.email IN ('employee11@explainableai.com','employee12@explainableai.com','employee13@explainableai.com','employee14@explainableai.com','employee15@explainableai.com') THEN true
         WHEN d.dataset_id = (SELECT dataset_id FROM datasets_data ORDER BY dataset_id LIMIT 1 OFFSET 3) AND u.email IN ('employee16@explainableai.com','employee17@explainableai.com','employee18@explainableai.com') THEN true
         ELSE false END,
    false,
    CASE WHEN u.email = 'admin@explainableai.com' THEN true
         WHEN d.dataset_id = (SELECT dataset_id FROM datasets_data ORDER BY dataset_id LIMIT 1 OFFSET 0) AND u.email IN ('employee01@explainableai.com','employee02@explainableai.com','employee03@explainableai.com','employee04@explainableai.com','employee05@explainableai.com') THEN true
         WHEN d.dataset_id = (SELECT dataset_id FROM datasets_data ORDER BY dataset_id LIMIT 1 OFFSET 1) AND u.email IN ('employee06@explainableai.com','employee07@explainableai.com','employee08@explainableai.com','employee09@explainableai.com','employee10@explainableai.com') THEN true
         WHEN d.dataset_id = (SELECT dataset_id FROM datasets_data ORDER BY dataset_id LIMIT 1 OFFSET 4) AND u.email IN ('employee01@explainableai.com','employee02@explainableai.com','employee06@explainableai.com','employee07@explainableai.com','employee11@explainableai.com','employee12@explainableai.com','employee16@explainableai.com','employee17@explainableai.com') THEN true
         ELSE false END,
    (SELECT user_id FROM users_data WHERE email = 'admin@explainableai.com')
  FROM users_data u CROSS JOIN datasets_data d
  WHERE u.email != 'admin@explainableai.com' OR u.user_id IS NOT NULL
),
activity_logs_data AS (
  INSERT INTO activity_logs(company_id, user_id, dataset_id, activity_type, activity_description, module_name, status, timestamp)
  VALUES
    -- LOGIN - Today
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'admin@explainableai.com'), NULL, 'LOGIN', 'Admin logged in', 'AUTH', 'ok', NOW() - interval '5 minutes'),
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'employee01@explainableai.com'), NULL, 'LOGIN', 'Employee 01 logged in', 'AUTH', 'ok', NOW() - interval '4 hours'),
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'employee02@explainableai.com'), NULL, 'LOGIN', 'Employee 02 logged in', 'AUTH', 'ok', NOW() - interval '3 hours'),
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'employee05@explainableai.com'), NULL, 'LOGIN', 'Employee 05 logged in', 'AUTH', 'ok', NOW() - interval '2 hours'),
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'employee10@explainableai.com'), NULL, 'LOGIN', 'Employee 10 logged in', 'AUTH', 'ok', NOW() - interval '1 hour 30 minutes'),
    
    -- LOGIN - Yesterday
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'employee15@explainableai.com'), NULL, 'LOGIN', 'Employee 15 logged in', 'AUTH', 'ok', NOW() - interval '1 day 2 hours'),
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'employee06@explainableai.com'), NULL, 'LOGIN', 'Employee 06 logged in', 'AUTH', 'ok', NOW() - interval '1 day 5 hours'),
    
    -- LOGIN - 2 Days ago
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'employee20@explainableai.com'), NULL, 'LOGIN', 'Employee 20 logged in', 'AUTH', 'ok', NOW() - interval '2 days 1 hour'),
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'admin@explainableai.com'), NULL, 'LOGIN', 'Admin logged in', 'AUTH', 'ok', NOW() - interval '2 days 6 hours'),
    
    -- VIEW - Today
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'employee01@explainableai.com'), (SELECT dataset_id FROM datasets_data ORDER BY dataset_id LIMIT 1 OFFSET 0), 'VIEW', 'Viewed Dataset 1', 'DATASET', 'ok', NOW() - interval '3 hours 30 minutes'),
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'employee02@explainableai.com'), (SELECT dataset_id FROM datasets_data ORDER BY dataset_id LIMIT 1 OFFSET 1), 'VIEW', 'Viewed Dataset 2', 'DATASET', 'ok', NOW() - interval '2 hours 45 minutes'),
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'employee05@explainableai.com'), (SELECT dataset_id FROM datasets_data ORDER BY dataset_id LIMIT 1 OFFSET 2), 'VIEW', 'Viewed Dataset 3', 'DATASET', 'ok', NOW() - interval '1 hour 20 minutes'),
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'employee10@explainableai.com'), (SELECT dataset_id FROM datasets_data ORDER BY dataset_id LIMIT 1 OFFSET 0), 'VIEW', 'Viewed Dataset 1', 'DATASET', 'ok', NOW() - interval '45 minutes'),
    
    -- VIEW - Yesterday
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'employee15@explainableai.com'), (SELECT dataset_id FROM datasets_data ORDER BY dataset_id LIMIT 1 OFFSET 3), 'VIEW', 'Viewed Dataset 4', 'DATASET', 'ok', NOW() - interval '1 day 3 hours'),
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'employee06@explainableai.com'), (SELECT dataset_id FROM datasets_data ORDER BY dataset_id LIMIT 1 OFFSET 1), 'VIEW', 'Viewed Dataset 2', 'DATASET', 'ok', NOW() - interval '1 day 4 hours'),
    
    -- QUERY - Today
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'employee01@explainableai.com'), (SELECT dataset_id FROM datasets_data ORDER BY dataset_id LIMIT 1 OFFSET 0), 'QUERY', 'Executed query on Dataset 1', 'CHAT', 'ok', NOW() - interval '3 hours'),
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'employee06@explainableai.com'), (SELECT dataset_id FROM datasets_data ORDER BY dataset_id LIMIT 1 OFFSET 1), 'QUERY', 'Executed query on Dataset 2', 'CHAT', 'ok', NOW() - interval '2 hours 20 minutes'),
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'employee10@explainableai.com'), (SELECT dataset_id FROM datasets_data ORDER BY dataset_id LIMIT 1 OFFSET 4), 'QUERY', 'Executed query on Dataset 5', 'CHAT', 'ok', NOW() - interval '1 hour'),
    
    -- QUERY - Yesterday
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'employee02@explainableai.com'), (SELECT dataset_id FROM datasets_data ORDER BY dataset_id LIMIT 1 OFFSET 0), 'QUERY', 'Executed query on Dataset 1', 'CHAT', 'ok', NOW() - interval '1 day 2 hours'),
    
    -- UPLOAD
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'employee01@explainableai.com'), (SELECT dataset_id FROM datasets_data ORDER BY dataset_id LIMIT 1 OFFSET 0), 'UPLOAD', 'Uploaded Dataset 1', 'DATASET', 'ok', NOW() - interval '10 days'),
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'employee01@explainableai.com'), (SELECT dataset_id FROM datasets_data ORDER BY dataset_id LIMIT 1 OFFSET 1), 'UPLOAD', 'Uploaded Dataset 2', 'DATASET', 'ok', NOW() - interval '9 days'),
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'employee02@explainableai.com'), (SELECT dataset_id FROM datasets_data ORDER BY dataset_id LIMIT 1 OFFSET 2), 'UPLOAD', 'Uploaded Dataset 3', 'DATASET', 'ok', NOW() - interval '8 days'),
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'employee03@explainableai.com'), (SELECT dataset_id FROM datasets_data ORDER BY dataset_id LIMIT 1 OFFSET 3), 'UPLOAD', 'Uploaded Dataset 4', 'DATASET', 'ok', NOW() - interval '7 days'),
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'employee04@explainableai.com'), (SELECT dataset_id FROM datasets_data ORDER BY dataset_id LIMIT 1 OFFSET 4), 'UPLOAD', 'Uploaded Dataset 5', 'DATASET', 'ok', NOW() - interval '6 days'),
    
    -- CLEAN
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'employee01@explainableai.com'), (SELECT dataset_id FROM datasets_data ORDER BY dataset_id LIMIT 1 OFFSET 0), 'CLEAN', 'Started data cleaning on Dataset 1', 'PIPELINE', 'ok', NOW() - interval '9 days 4 hours'),
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'employee05@explainableai.com'), (SELECT dataset_id FROM datasets_data ORDER BY dataset_id LIMIT 1 OFFSET 2), 'CLEAN', 'Started data cleaning on Dataset 3', 'PIPELINE', 'ok', NOW() - interval '7 days 2 hours'),
    
    -- TRAIN
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'employee03@explainableai.com'), (SELECT dataset_id FROM datasets_data ORDER BY dataset_id LIMIT 1 OFFSET 0), 'TRAIN', 'Started training model on Dataset 1', 'ML', 'ok', NOW() - interval '8 days 8 hours'),
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'employee10@explainableai.com'), (SELECT dataset_id FROM datasets_data ORDER BY dataset_id LIMIT 1 OFFSET 1), 'TRAIN', 'Started training model on Dataset 2', 'ML', 'ok', NOW() - interval '5 days 6 hours'),
    
    -- UPDATE
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'employee02@explainableai.com'), (SELECT dataset_id FROM datasets_data ORDER BY dataset_id LIMIT 1 OFFSET 1), 'UPDATE', 'Updated Dataset 2 metadata', 'DATASET', 'ok', NOW() - interval '20 hours'),
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'employee15@explainableai.com'), (SELECT dataset_id FROM datasets_data ORDER BY dataset_id LIMIT 1 OFFSET 3), 'UPDATE', 'Updated Dataset 4 schema', 'DATASET', 'ok', NOW() - interval '18 hours'),
    
    -- LOGOUT - Today
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'employee01@explainableai.com'), NULL, 'LOGOUT', 'Employee 01 logged out', 'AUTH', 'ok', NOW() - interval '3 hours 15 minutes'),
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'employee02@explainableai.com'), NULL, 'LOGOUT', 'Employee 02 logged out', 'AUTH', 'ok', NOW() - interval '2 hours 30 minutes'),
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'employee05@explainableai.com'), NULL, 'LOGOUT', 'Employee 05 logged out', 'AUTH', 'ok', NOW() - interval '1 hour'),
    
    -- LOGOUT - Yesterday
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'employee15@explainableai.com'), NULL, 'LOGOUT', 'Employee 15 logged out', 'AUTH', 'ok', NOW() - interval '1 day 30 minutes'),
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'employee06@explainableai.com'), NULL, 'LOGOUT', 'Employee 06 logged out', 'AUTH', 'ok', NOW() - interval '1 day 2 hours'),
    
    -- LOGOUT - 2 Days ago
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'employee20@explainableai.com'), NULL, 'LOGOUT', 'Employee 20 logged out', 'AUTH', 'ok', NOW() - interval '2 days 30 minutes'),
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'admin@explainableai.com'), NULL, 'LOGOUT', 'Admin logged out', 'AUTH', 'ok', NOW() - interval '2 days 4 hours'),
    
    -- Additional logs for completeness
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'employee03@explainableai.com'), (SELECT dataset_id FROM datasets_data ORDER BY dataset_id LIMIT 1 OFFSET 2), 'VIEW', 'Viewed Dataset 3', 'DATASET', 'ok', NOW() - interval '12 hours'),
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'employee07@explainableai.com'), (SELECT dataset_id FROM datasets_data ORDER BY dataset_id LIMIT 1 OFFSET 1), 'QUERY', 'Executed query on Dataset 2', 'CHAT', 'ok', NOW() - interval '11 hours'),
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'employee12@explainableai.com'), (SELECT dataset_id FROM datasets_data ORDER BY dataset_id LIMIT 1 OFFSET 2), 'VIEW', 'Viewed Dataset 3', 'DATASET', 'ok', NOW() - interval '8 hours'),
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'employee18@explainableai.com'), (SELECT dataset_id FROM datasets_data ORDER BY dataset_id LIMIT 1 OFFSET 3), 'QUERY', 'Executed query on Dataset 4', 'CHAT', 'ok', NOW() - interval '6 hours')
  RETURNING log_id
),
query_logs_data AS (
  INSERT INTO query_logs(company_id, user_id, dataset_id, query_text, query_type, execution_time_ms, status, rows_returned, timestamp)
  VALUES
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'employee01@explainableai.com'), (SELECT dataset_id FROM datasets_data ORDER BY dataset_id LIMIT 1 OFFSET 0), 'SELECT * FROM sales WHERE year=2024', 'select', 125, 'success', 5420, NOW() - interval '3 hours'),
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'employee06@explainableai.com'), (SELECT dataset_id FROM datasets_data ORDER BY dataset_id LIMIT 1 OFFSET 1), 'SELECT COUNT(*) FROM transactions GROUP BY category', 'select', 87, 'success', 42, NOW() - interval '2 hours 20 minutes'),
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'employee10@explainableai.com'), (SELECT dataset_id FROM datasets_data ORDER BY dataset_id LIMIT 1 OFFSET 4), 'SELECT AVG(amount) FROM orders WHERE date > 2024-01-01', 'select', 156, 'success', 8921, NOW() - interval '1 hour'),
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'employee02@explainableai.com'), (SELECT dataset_id FROM datasets_data ORDER BY dataset_id LIMIT 1 OFFSET 0), 'SELECT region, SUM(revenue) FROM sales GROUP BY region', 'select', 210, 'success', 12, NOW() - interval '1 day 2 hours'),
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'employee07@explainableai.com'), (SELECT dataset_id FROM datasets_data ORDER BY dataset_id LIMIT 1 OFFSET 1), 'SELECT * FROM dataset WHERE status = active LIMIT 100', 'select', 98, 'success', 100, NOW() - interval '11 hours'),
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'employee12@explainableai.com'), (SELECT dataset_id FROM datasets_data ORDER BY dataset_id LIMIT 1 OFFSET 2), 'SELECT COUNT(DISTINCT user_id) FROM events', 'select', 145, 'success', 1, NOW() - interval '8 hours'),
    ((SELECT company_id FROM company), (SELECT user_id FROM users_data WHERE email = 'employee18@explainableai.com'), (SELECT dataset_id FROM datasets_data ORDER BY dataset_id LIMIT 1 OFFSET 3), 'SELECT * FROM analytics WHERE metric > 100', 'select', 167, 'success', 456, NOW() - interval '6 hours')
  RETURNING log_id
)
SELECT 'Successfully inserted all test data: 1 admin, 20 employees, 5 datasets, permissions, 40+ activity logs, and query logs!' as status;

COMMIT;

-- update in users table
-- -----------------------------------------------------
UPDATE users SET password_hash = '$2b$12$glClmdFA.DD6O.v4WdyfY.zEfCnSDszc1.Vz1rIa2w0Iq6Y5iJNsO' WHERE email = 'employee01@explainableai.com';
UPDATE users SET password_hash = '$2b$12$CGOwHgE7.FDHR/j/2u2bW.hPk1hUFHDg/WBelvIZh1e8wDPjrRszm' WHERE email = 'employee02@explainableai.com';
UPDATE users SET password_hash = '$2b$12$EGVSfDKSEin.G9peNd1ceOc4S6F4FlQRnaJR.juIztEw6YqWcuOLS' WHERE email = 'employee03@explainableai.com';
UPDATE users SET password_hash = '$2b$12$5sxlqTaN2Gu.h8xlSQlgjOCouVRW5TrV84hRpnw6LpMSJFNQTxB.K' WHERE email = 'employee04@explainableai.com';
UPDATE users SET password_hash = '$2b$12$t7Dcvh/y8codGGBO1WXPr.up8Qlq49Oni/FgKfoVxjbvnmIlGFqCa' WHERE email = 'employee05@explainableai.com';
UPDATE users SET password_hash = '$2b$12$jjGlHR1MclxdAQ5FtPQUN.7PUacNUK2zFufrEnd5.Ig5a2Awo8e46' WHERE email = 'employee06@explainableai.com';
UPDATE users SET password_hash = '$2b$12$9ndE9/BC0Mk6kX/D9yNM1uOad2T1unbcZx3j/nL./DcKvJRDClm7y' WHERE email = 'employee07@explainableai.com';
UPDATE users SET password_hash = '$2b$12$WrqkrKgd4CYFlWDafIb2te.98v4z7TFmXx8wY4uv.4Tx8h6N9U0Ki' WHERE email = 'employee08@explainableai.com';
UPDATE users SET password_hash = '$2b$12$zyOnzwuI2Wboj6.SrJqMYOE1IHwoZHGf9UWPovrSaUwvnoHLK0ppG' WHERE email = 'employee09@explainableai.com';
UPDATE users SET password_hash = '$2b$12$siW72f2qHxYaP4NIrfr0Per.YMqX2Sk.odS981FDY3zTRwfhEsCNW' WHERE email = 'employee10@explainableai.com';
UPDATE users SET password_hash = '$2b$12$/F/Mq9tmHjWGcEzOyScXRuoBJKxjq5W9.jrblKTherpULdGxXPxWC' WHERE email = 'employee11@explainableai.com';
UPDATE users SET password_hash = '$2b$12$U8kRp7P8/lpLb42840P67ebWV2Yp.LvEPmAcCxUoVh.9DJvSquLme' WHERE email = 'employee12@explainableai.com';
UPDATE users SET password_hash = '$2b$12$XWGGJxHCoKxjwTnE7uGGZueUNahzZRjuLKsbWngaffr4V4ITTdhaG' WHERE email = 'employee13@explainableai.com';
UPDATE users SET password_hash = '$2b$12$tbioXk82dW0zhcuJ.VVtZ.bKgxSaOjX.EpScpszaUHUQvJesNCuHa' WHERE email = 'employee14@explainableai.com';
UPDATE users SET password_hash = '$2b$12$htOGP0EJ/npnJ/rTRLqkm.O49CKolAWSSIjIXEWDPeTXcmsHg6lFC' WHERE email = 'employee15@explainableai.com';
UPDATE users SET password_hash = '$2b$12$JwKGJve1PYoeK8AALXjK1eFsao/ms2TPwHvUHtJszsN0wWrUo3siK' WHERE email = 'employee16@explainableai.com';
UPDATE users SET password_hash = '$2b$12$nI84VP5Se2OxqBfy1mCwye27IhC8DnHbdcmVNlura2WZilm7DhVFe' WHERE email = 'employee17@explainableai.com';
UPDATE users SET password_hash = '$2b$12$AX18reQYaePBZld1.zGbr.R7ckPTPS7/u9wRI3FiM7mf2ZJlj2P1O' WHERE email = 'employee18@explainableai.com';
UPDATE users SET password_hash = '$2b$12$aSbn1eEd38WiC/KOqkG67.u.wtD6uSWaBw5Vl5NF0c9gbgFaGJmca' WHERE email = 'employee19@explainableai.com';
UPDATE users SET password_hash = '$2b$12$J.xaTVDEVhOQjBGi68mSvOemk7/v7wfFuwp6GvpryoJPMIfjpHMm2' WHERE email = 'employee20@explainableai.com';
