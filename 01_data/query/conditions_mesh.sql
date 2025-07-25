SELECT
	s.nct_id,
	c.name,
	bc.mesh_term AS conditions_mesh_term,
	mt.mesh_term,
	mt.tree_number
FROM
	studies s
	LEFT JOIN conditions c ON s.nct_id = c.nct_id
	LEFT JOIN browse_conditions bc ON s.nct_id = bc.nct_id
	LEFT JOIN mesh_terms mt ON bc.mesh_term = mt.mesh_term
WHERE
	s.overall_status IN ('COMPLETED', 'TERMINATED')
	AND s.plan_to_share_ipd = 'YES';