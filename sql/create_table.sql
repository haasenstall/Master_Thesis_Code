-- 1. Platform (Vivli, CSDR, YODA etc.)
CREATE TABLE platforms (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    url TEXT,
    description TEXT
);

-- 2. Clinical Trials
CREATE TABLE clinical_trials (
    nct_id TEXT PRIMARY KEY,
    title TEXT,
	investigator TEXT,
	institution TEXT,
    condition TEXT,
	condition_code TEXT,
	intervention TEXT,
    intervention_code TEXT,
	phase TEXT,
    status TEXT,
	enrollment INTEGER,
	enrollment_type TEXT,
	study_type TEXT,
    sponsor TEXT,
    start_date DATE,
    completion_date DATE,
	date_published TEXT,
	plan_to_share TEXT,
	brief_summary TEXT,
    country TEXT
);
CREATE TABLE eligibility_criteria (
	id SERIAL PRIMARY KEY,
	nct_id TEXT NOT NULL REFERENCES clinical_trials(nct_id),
	sampling_method TEXT,
	gender TEXT,
	minimum_age TEXT,
	maximum_age TEXT	
);
CREATE TABLE institution (
	id SERIAL PRIMARY KEY,
	institution_name TEXT UNIQUE,
	street TEXT,
	street_number TEXT,
	zip_code TEXT,
	city TEXT,
	country TEXT	
);
CREATE TABLE investigator (
	id SERIAL PRIMARY KEY,
	investigator_name TEXT,
	institution_id INTEGER NOT NULL REFERENCES institution(id),
	institution_name TEXT
);
CREATE TABLE sponsors (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
	agency_class TEXT,
    type TEXT
);
CREATE TABLE outcomes (
	id SERIAL PRIMARY KEY,
	nct_id TEXT NOT NULL REFERENCES clinical_trials(nct_id),
	measure_type TEXT CHECK (measure_type IN ('primary', 'secondary')),
	measure TEXT
);
CREATE TABLE design (
	id SERIAL PRIMARY KEY,
	nct_id TEXT NOT NULL REFERENCES clinical_trials(nct_id),
	allocation TEXT,
	interventional_model TEXT,
	observational_model TEXT,
	primary_pupose TEXT
);

-- 3. Trials on Platforms
CREATE TABLE data_access (
    id SERIAL PRIMARY KEY,
    trial_nct_id TEXT REFERENCES clinical_trials(nct_id) ON DELETE CASCADE,
    platform_id INTEGER REFERENCES platforms(id) ON DELETE CASCADE,
    platform_name TEXT,
	data_available BOOLEAN
);

-- 4. Requests
CREATE TABLE requests (
    request_id TEXT PRIMARY KEY,
    platform_id INTEGER REFERENCES platforms(id) ON DELETE CASCADE,
	title TEXT,
	investigator TEXT,
	institution TEXT,
	date_of_request DATE,
	number_of_trials_requested TEXT	
);

-- 5. Requests and Trials (Many-to-Many)
CREATE TABLE request_trial_links (
    request_id TEXT REFERENCES requests(request_id) ON DELETE CASCADE,
    nct_id TEXT REFERENCES clinical_trials(nct_id) ON DELETE CASCADE,
    PRIMARY KEY (request_id, nct_id)
);