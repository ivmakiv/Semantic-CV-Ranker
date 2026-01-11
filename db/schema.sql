CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS cv_profiles (
  file_name TEXT PRIMARY KEY,

  education_degree TEXT,
  education_field vector(384),

  experience_years DOUBLE PRECISION,
  experience_position vector(384),

  location_lat DOUBLE PRECISION,
  location_lon DOUBLE PRECISION,

  skills vector(384)
);
