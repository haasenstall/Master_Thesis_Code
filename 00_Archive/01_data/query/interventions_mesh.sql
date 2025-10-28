SELECT
	s.nct_id,
	i.name,
	bi.mesh_term AS conditions_mesh_term,
	mt.mesh_term,
	mt.tree_number
FROM
	studies s
	LEFT JOIN interventions i ON s.nct_id = i.nct_id
	LEFT JOIN browse_interventions bi ON s.nct_id = bi.nct_id
	LEFT JOIN mesh_terms mt ON bi.mesh_term = mt.mesh_term
WHERE
	s.overall_status IN ('COMPLETED', 'TERMINATED')
	AND s.plan_to_share_ipd = 'YES';