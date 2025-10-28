SELECT 
    schemaname || '.' || relname AS table_name,
    n_live_tup AS estimated_row_count
FROM pg_stat_user_tables
ORDER BY n_live_tup DESC;
