SELECT
    c.table_schema,
    c.table_name,
    STRING_AGG(c.column_name, ', ' ORDER BY c.ordinal_position) AS columns
FROM information_schema.columns c
WHERE c.table_schema NOT IN ('pg_catalog', 'information_schema')
GROUP BY c.table_schema, c.table_name
HAVING SUM(CASE WHEN c.column_name = 'nct_id' THEN 1 ELSE 0 END) > 0
ORDER BY c.table_schema, c.table_name;
