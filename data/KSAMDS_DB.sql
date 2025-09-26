-- CREATE SCHEMA
CREATE SCHEMA IF NOT EXISTS ksamds;

-- SET SEARCH PATH
SET search_path TO ksamds, public;

-- Enable useful extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp" SCHEMA ksamds;
CREATE EXTENSION IF NOT EXISTS pg_trgm SCHEMA ksamds;

-- =========================
-- Core entities
-- =========================
CREATE TABLE IF NOT EXISTS ksamds.knowledge (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  name TEXT NOT NULL UNIQUE,
  definition TEXT,
  domain TEXT,      
  source_ref TEXT,          
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ksamds.skill (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  name TEXT NOT NULL UNIQUE,
  definition TEXT,
  domain TEXT,
  source_ref TEXT,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ksamds.ability (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  name TEXT NOT NULL UNIQUE,
  definition TEXT,
  domain TEXT,
  source_ref TEXT,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ksamds.function (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  name TEXT NOT NULL UNIQUE,
  definition TEXT,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ksamds.task (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  name TEXT NOT NULL UNIQUE,
  definition TEXT,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ksamds.occupation (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  title TEXT NOT NULL,      -- e.g., "Data Engineer"
  description TEXT,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

-- =========================
-- Dimensions
-- =========================
-- scope keeps each dimension constrained to the intended anchors

CREATE TABLE IF NOT EXISTS ksamds.type_dim (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  scope TEXT NOT NULL CHECK (scope IN ('K','S','A','T')),
  name TEXT NOT NULL,
  description TEXT
);

CREATE TABLE IF NOT EXISTS ksamds.level_dim (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  scope TEXT NOT NULL CHECK (scope IN ('K','S','A')),
  name TEXT NOT NULL,          -- e.g., Basic, Intermediate, Advanced
  description TEXT
);

CREATE TABLE IF NOT EXISTS ksamds.basis_dim (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  scope TEXT NOT NULL CHECK (scope IN ('K','S','A')),
  name TEXT NOT NULL,          -- e.g., Theory, OJT
  description TEXT
);

CREATE TABLE IF NOT EXISTS ksamds.environment_dim (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  scope TEXT NOT NULL CHECK (scope IN ('F','T')),
  name TEXT NOT NULL,          -- e.g., Office, Outdoor, Lab
  description TEXT
);

CREATE TABLE IF NOT EXISTS ksamds.mode_dim (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  scope TEXT NOT NULL CHECK (scope IN ('T')),
  name TEXT NOT NULL,          -- e.g., Tool, Process, Theory
  description TEXT
);

CREATE TABLE IF NOT EXISTS ksamds.physicality_dim (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  name TEXT NOT NULL,          -- Light, Moderate, Heavy
  description TEXT
);

CREATE TABLE IF NOT EXISTS ksamds.cognitive_dim (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  name TEXT NOT NULL,          -- Light, Moderate, Heavy
  description TEXT
);

CREATE TABLE IF NOT EXISTS ksamds.education_level (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  name TEXT NOT NULL                 -- e.g., "High school", "Bachelor's", "Master's", "PhD"
);

CREATE TABLE IF NOT EXISTS ksamds.certification (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  issuer TEXT,                         -- e.g., "AWS"
  name TEXT NOT NULL                  -- e.g., "AWS Solutions Architect – Associate"
);

-- =========================
-- Core ↔ Dimension junctions
-- =========================
CREATE TABLE IF NOT EXISTS ksamds.knowledge_type (
  knowledge_id UUID REFERENCES ksamds.knowledge(id) ON DELETE CASCADE,
  type_id UUID REFERENCES ksamds.type_dim(id) ON DELETE CASCADE,
  PRIMARY KEY (knowledge_id, type_id)
);

CREATE TABLE IF NOT EXISTS ksamds.knowledge_level (
  knowledge_id UUID REFERENCES ksamds.knowledge(id) ON DELETE CASCADE,
  level_id UUID REFERENCES ksamds.level_dim(id) ON DELETE CASCADE,
  PRIMARY KEY (knowledge_id, level_id)
);

CREATE TABLE IF NOT EXISTS ksamds.knowledge_basis (
  knowledge_id UUID REFERENCES ksamds.knowledge(id) ON DELETE CASCADE,
  basis_id UUID REFERENCES ksamds.basis_dim(id) ON DELETE CASCADE,
  PRIMARY KEY (knowledge_id, basis_id)
);

CREATE TABLE IF NOT EXISTS ksamds.skill_type (
  skill_id UUID REFERENCES ksamds.skill(id) ON DELETE CASCADE,
  type_id UUID REFERENCES ksamds.type_dim(id) ON DELETE CASCADE,
  PRIMARY KEY (skill_id, type_id)
);

CREATE TABLE IF NOT EXISTS ksamds.skill_level (
  skill_id UUID REFERENCES ksamds.skill(id) ON DELETE CASCADE,
  level_id UUID REFERENCES ksamds.level_dim(id) ON DELETE CASCADE,
  PRIMARY KEY (skill_id, level_id)
);

CREATE TABLE IF NOT EXISTS ksamds.skill_basis (
  skill_id UUID REFERENCES ksamds.skill(id) ON DELETE CASCADE,
  basis_id UUID REFERENCES ksamds.basis_dim(id) ON DELETE CASCADE,
  PRIMARY KEY (skill_id, basis_id)
);

CREATE TABLE IF NOT EXISTS ksamds.ability_type (
  ability_id UUID REFERENCES ksamds.ability(id) ON DELETE CASCADE,
  type_id UUID REFERENCES ksamds.type_dim(id) ON DELETE CASCADE,
  PRIMARY KEY (ability_id, type_id)
);

CREATE TABLE IF NOT EXISTS ksamds.ability_level (
  ability_id UUID REFERENCES ksamds.ability(id) ON DELETE CASCADE,
  level_id UUID REFERENCES ksamds.level_dim(id) ON DELETE CASCADE,
  PRIMARY KEY (ability_id, level_id)
);

CREATE TABLE IF NOT EXISTS ksamds.ability_basis (
  ability_id UUID REFERENCES ksamds.ability(id) ON DELETE CASCADE,
  basis_id UUID REFERENCES ksamds.basis_dim(id) ON DELETE CASCADE,
  PRIMARY KEY (ability_id, basis_id)
);

CREATE TABLE IF NOT EXISTS ksamds.function_env (
  function_id UUID REFERENCES ksamds.function(id) ON DELETE CASCADE,
  environment_id UUID REFERENCES ksamds.environment_dim(id) ON DELETE CASCADE,
  PRIMARY KEY (function_id, environment_id)
);

CREATE TABLE IF NOT EXISTS ksamds.function_physicality (
  function_id UUID REFERENCES ksamds.function(id) ON DELETE CASCADE,
  physicality_id UUID REFERENCES ksamds.physicality_dim(id) ON DELETE CASCADE,
  PRIMARY KEY (function_id, physicality_id)
);

CREATE TABLE IF NOT EXISTS ksamds.function_cognitive (
  function_id UUID REFERENCES ksamds.function(id) ON DELETE CASCADE,
  cognitive_id UUID REFERENCES ksamds.cognitive_dim(id) ON DELETE CASCADE,
  PRIMARY KEY (function_id, cognitive_id)
);

CREATE TABLE IF NOT EXISTS ksamds.task_env (
  task_id UUID REFERENCES ksamds.task(id) ON DELETE CASCADE,
  environment_id UUID REFERENCES ksamds.environment_dim(id) ON DELETE CASCADE,
  PRIMARY KEY (task_id, environment_id)
);

CREATE TABLE IF NOT EXISTS ksamds.task_type (
  task_id UUID REFERENCES ksamds.task(id) ON DELETE CASCADE,
  type_id UUID REFERENCES ksamds.type_dim(id) ON DELETE CASCADE,
  PRIMARY KEY (task_id, type_id)
);

CREATE TABLE IF NOT EXISTS ksamds.task_mode (
  task_id UUID REFERENCES ksamds.task(id) ON DELETE CASCADE,
  mode_id UUID REFERENCES ksamds.mode_dim(id) ON DELETE CASCADE,
  PRIMARY KEY (task_id, mode_id)
);

CREATE TABLE IF NOT EXISTS ksamds.occupation_knowledge (
  occupation_id UUID REFERENCES ksamds.occupation(id) ON DELETE CASCADE,
  knowledge_id UUID REFERENCES ksamds.knowledge(id) ON DELETE CASCADE,
  PRIMARY KEY (occupation_id, knowledge_id)
);

CREATE TABLE IF NOT EXISTS ksamds.occupation_skill (
  occupation_id UUID REFERENCES ksamds.occupation(id) ON DELETE CASCADE,
  skill_id UUID REFERENCES ksamds.skill(id) ON DELETE CASCADE,
  PRIMARY KEY (occupation_id, skill_id)
);

CREATE TABLE IF NOT EXISTS ksamds.occupation_ability (
  occupation_id UUID REFERENCES ksamds.occupation(id) ON DELETE CASCADE,
  ability_id UUID REFERENCES ksamds.ability(id) ON DELETE CASCADE,
  PRIMARY KEY (occupation_id, ability_id)
);

CREATE TABLE IF NOT EXISTS ksamds.occupation_function (
  occupation_id UUID REFERENCES ksamds.occupation(id) ON DELETE CASCADE,
  function_id UUID REFERENCES ksamds.function(id) ON DELETE CASCADE,
  PRIMARY KEY (occupation_id, function_id)
);

CREATE TABLE IF NOT EXISTS ksamds.occupation_task (
  occupation_id UUID REFERENCES ksamds.occupation(id) ON DELETE CASCADE,
  task_id UUID REFERENCES ksamds.task(id) ON DELETE CASCADE,
  PRIMARY KEY (occupation_id, task_id)
);

CREATE TABLE IF NOT EXISTS ksamds.occupation_education (
  occupation_id UUID REFERENCES ksamds.occupation(id) ON DELETE CASCADE,
  education_level_id UUID REFERENCES ksamds.education_level(id),
  PRIMARY KEY (occupation_id, education_level_id)
);

CREATE TABLE IF NOT EXISTS ksamds.occupation_certification (
  occupation_id UUID REFERENCES ksamds.occupation(id) ON DELETE CASCADE,
  certification_id UUID REFERENCES ksamds.certification(id) ON DELETE CASCADE,
  PRIMARY KEY (occupation_id, certification_id)
);

-- =========================
-- Core ↔ Core relationships (graph)
-- =========================
CREATE TABLE IF NOT EXISTS ksamds.knowledge_skill (
  knowledge_id UUID REFERENCES ksamds.knowledge(id) ON DELETE CASCADE,
  skill_id UUID REFERENCES ksamds.skill(id) ON DELETE CASCADE,
  PRIMARY KEY (knowledge_id, skill_id)
);

CREATE TABLE IF NOT EXISTS ksamds.knowledge_function (
  knowledge_id UUID REFERENCES ksamds.knowledge(id) ON DELETE CASCADE,
  function_id UUID REFERENCES ksamds.function(id) ON DELETE CASCADE,
  PRIMARY KEY (knowledge_id, function_id)
);

CREATE TABLE IF NOT EXISTS ksamds.skill_ability (
  skill_id UUID REFERENCES ksamds.skill(id) ON DELETE CASCADE,
  ability_id UUID REFERENCES ksamds.ability(id) ON DELETE CASCADE,
  PRIMARY KEY (skill_id, ability_id)
);

CREATE TABLE IF NOT EXISTS ksamds.function_task (
  function_id UUID REFERENCES ksamds.function(id) ON DELETE CASCADE,
  task_id UUID REFERENCES ksamds.task(id) ON DELETE CASCADE,
  PRIMARY KEY (function_id, task_id)
);

CREATE TABLE IF NOT EXISTS ksamds.ability_task (
  ability_id UUID REFERENCES ksamds.ability(id) ON DELETE CASCADE,
  task_id UUID REFERENCES ksamds.task(id) ON DELETE CASCADE,
  PRIMARY KEY (ability_id, task_id)
);

-- =========================
-- Helpful indexes for fast text search now; pgvector can be added later
-- =========================
CREATE INDEX ON ksamds.occupation_skill (occupation_id);
CREATE INDEX ON ksamds.occupation_knowledge (occupation_id);
CREATE INDEX ON ksamds.occupation_ability (occupation_id);
CREATE INDEX ON ksamds.occupation_task (occupation_id);
CREATE INDEX ON ksamds.occupation_function (occupation_id);
CREATE INDEX IF NOT EXISTS idx_occ_title_trgm ON ksamds.occupation USING gin (title gin_trgm_ops);