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
  id UUID PRIMARY KEY DEFAULT ksamds.uuid_generate_v4(),
  name TEXT NOT NULL UNIQUE,    
  source_ref TEXT,          
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ksamds.skill (
  id UUID PRIMARY KEY DEFAULT ksamds.uuid_generate_v4(),
  name TEXT NOT NULL UNIQUE,
  source_ref TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ksamds.ability (
  id UUID PRIMARY KEY DEFAULT ksamds.uuid_generate_v4(),
  name TEXT NOT NULL UNIQUE,
  source_ref TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ksamds.function (
  id UUID PRIMARY KEY DEFAULT ksamds.uuid_generate_v4(),
  name TEXT NOT NULL UNIQUE,
  source_ref TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ksamds.task (
  id UUID PRIMARY KEY DEFAULT ksamds.uuid_generate_v4(),
  name TEXT NOT NULL UNIQUE,
  source_ref TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ksamds.occupation (
  id UUID PRIMARY KEY DEFAULT ksamds.uuid_generate_v4(),
  title TEXT NOT NULL UNIQUE,
  description TEXT,
  source_ref TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- =========================
-- Dimensions
-- =========================
-- scope keeps each dimension constrained to the intended anchors

CREATE TABLE IF NOT EXISTS ksamds.type_dim (
  id UUID PRIMARY KEY DEFAULT ksamds.uuid_generate_v4(),
  scope TEXT NOT NULL CHECK (scope IN ('K','S','A','T')),
  name TEXT NOT NULL,
  description TEXT,
  UNIQUE(scope, name)
);

CREATE TABLE IF NOT EXISTS ksamds.level_dim (
  id UUID PRIMARY KEY DEFAULT ksamds.uuid_generate_v4(),
  scope TEXT NOT NULL CHECK (scope IN ('K','S','A')),
  name TEXT NOT NULL,          -- e.g., Basic, Intermediate, Advanced
  ordinal INTEGER,                     -- For ordering: 1=Basic, 2=Intermediate, 3=Advanced
  description TEXT,
  UNIQUE(scope, name)
);

CREATE TABLE IF NOT EXISTS ksamds.basis_dim (
  id UUID PRIMARY KEY DEFAULT ksamds.uuid_generate_v4(),
  scope TEXT NOT NULL CHECK (scope IN ('K','S','A')),
  name TEXT NOT NULL,          -- e.g., Theory, OJT
  description TEXT,
  UNIQUE(scope, name)
);

CREATE TABLE IF NOT EXISTS ksamds.environment_dim (
  id UUID PRIMARY KEY DEFAULT ksamds.uuid_generate_v4(),
  scope TEXT NOT NULL CHECK (scope IN ('F','T')),
  name TEXT NOT NULL,          -- e.g., Office, Outdoor, Lab
  description TEXT,
  UNIQUE(scope, name)
);

CREATE TABLE IF NOT EXISTS ksamds.mode_dim (
  id UUID PRIMARY KEY DEFAULT ksamds.uuid_generate_v4(),
  name TEXT NOT NULL UNIQUE,   -- e.g., Tool, Process, Theory
  description TEXT
);

CREATE TABLE IF NOT EXISTS ksamds.physicality_dim (
  id UUID PRIMARY KEY DEFAULT ksamds.uuid_generate_v4(),
  name TEXT NOT NULL UNIQUE,          -- Light, Moderate, Heavy
  description TEXT
);

CREATE TABLE IF NOT EXISTS ksamds.cognitive_dim (
  id UUID PRIMARY KEY DEFAULT ksamds.uuid_generate_v4(),
  name TEXT NOT NULL UNIQUE,          -- Light, Moderate, Heavy
  description TEXT
);

CREATE TABLE IF NOT EXISTS ksamds.education_level (
  id UUID PRIMARY KEY DEFAULT ksamds.uuid_generate_v4(),
  name TEXT NOT NULL UNIQUE,          -- e.g., "High school", "Bachelor's", "Master's", "PhD"
  ordinal INTEGER                     -- For ordering education levels
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

-- =========================
-- Occupation relationships with level tracking
-- =========================
-- Track specific dimension requirements per occupation
CREATE TABLE IF NOT EXISTS ksamds.occupation_knowledge (
  occupation_id UUID REFERENCES ksamds.occupation(id) ON DELETE CASCADE,
  knowledge_id UUID REFERENCES ksamds.knowledge(id) ON DELETE CASCADE,
  type_id UUID REFERENCES ksamds.type_dim(id),        -- Track required type
  level_id UUID REFERENCES ksamds.level_dim(id),      -- Track required level
  basis_id UUID REFERENCES ksamds.basis_dim(id),      -- Track required basis
  importance_score NUMERIC(4,2),                      -- Optional: O*NET importance rating
  PRIMARY KEY (occupation_id, knowledge_id)
);

CREATE TABLE IF NOT EXISTS ksamds.occupation_skill (
  occupation_id UUID REFERENCES ksamds.occupation(id) ON DELETE CASCADE,
  skill_id UUID REFERENCES ksamds.skill(id) ON DELETE CASCADE,
  type_id UUID REFERENCES ksamds.type_dim(id),        -- Track required type
  level_id UUID REFERENCES ksamds.level_dim(id),      -- Track required level
  basis_id UUID REFERENCES ksamds.basis_dim(id),      -- Track required basis
  importance_score NUMERIC(4,2),                      -- Optional: O*NET importance rating
  PRIMARY KEY (occupation_id, skill_id)
);

CREATE TABLE IF NOT EXISTS ksamds.occupation_ability (
  occupation_id UUID REFERENCES ksamds.occupation(id) ON DELETE CASCADE,
  ability_id UUID REFERENCES ksamds.ability(id) ON DELETE CASCADE,
  type_id UUID REFERENCES ksamds.type_dim(id),        -- Track required type
  level_id UUID REFERENCES ksamds.level_dim(id),      -- Track required level
  basis_id UUID REFERENCES ksamds.basis_dim(id),      -- Track required basis
  importance_score NUMERIC(4,2),                      -- Optional: O*NET importance rating
  PRIMARY KEY (occupation_id, ability_id)
);

CREATE TABLE IF NOT EXISTS ksamds.occupation_function (
  occupation_id UUID REFERENCES ksamds.occupation(id) ON DELETE CASCADE,
  function_id UUID REFERENCES ksamds.function(id) ON DELETE CASCADE,
  environment_id UUID REFERENCES ksamds.environment_dim(id),  -- Track required environment
  physicality_id UUID REFERENCES ksamds.physicality_dim(id),  -- Track required physicality
  cognitive_id UUID REFERENCES ksamds.cognitive_dim(id),      -- Track required cognitive load
  importance_score NUMERIC(4,2),                              -- Optional: importance rating
  PRIMARY KEY (occupation_id, function_id)
);

CREATE TABLE IF NOT EXISTS ksamds.occupation_task (
  occupation_id UUID REFERENCES ksamds.occupation(id) ON DELETE CASCADE,
  task_id UUID REFERENCES ksamds.task(id) ON DELETE CASCADE,
  type_id UUID REFERENCES ksamds.type_dim(id),        -- Track required task type
  environment_id UUID REFERENCES ksamds.environment_dim(id),  -- Track required environment
  mode_id UUID REFERENCES ksamds.mode_dim(id),        -- Track required mode
  importance_score NUMERIC(4,2),                      -- Optional: importance rating
  PRIMARY KEY (occupation_id, task_id)
);

CREATE TABLE IF NOT EXISTS ksamds.occupation_education (
  occupation_id UUID REFERENCES ksamds.occupation(id) ON DELETE CASCADE,
  education_level_id UUID REFERENCES ksamds.education_level(id),
  is_required BOOLEAN DEFAULT false,              -- Is this level required vs. preferred?
  PRIMARY KEY (occupation_id, education_level_id)
);

-- =========================
-- Core ↔ Core relationships (graph)
-- =========================
-- These represent inferred/semantic relationships between entities
-- Optional: Add confidence_score to track relationship strength from embeddings
CREATE TABLE IF NOT EXISTS ksamds.knowledge_skill (
  knowledge_id UUID REFERENCES ksamds.knowledge(id) ON DELETE CASCADE,
  skill_id UUID REFERENCES ksamds.skill(id) ON DELETE CASCADE,
  confidence_score NUMERIC(4,3),                  -- From embedding similarity
  PRIMARY KEY (knowledge_id, skill_id)
);

CREATE TABLE IF NOT EXISTS ksamds.knowledge_function (
  knowledge_id UUID REFERENCES ksamds.knowledge(id) ON DELETE CASCADE,
  function_id UUID REFERENCES ksamds.function(id) ON DELETE CASCADE,
  confidence_score NUMERIC(4,3),                  -- From embedding similarity
  PRIMARY KEY (knowledge_id, function_id)
);

CREATE TABLE IF NOT EXISTS ksamds.skill_ability (
  skill_id UUID REFERENCES ksamds.skill(id) ON DELETE CASCADE,
  ability_id UUID REFERENCES ksamds.ability(id) ON DELETE CASCADE,
  confidence_score NUMERIC(4,3),                  -- From embedding similarity
  PRIMARY KEY (skill_id, ability_id)
);

CREATE TABLE IF NOT EXISTS ksamds.function_task (
  function_id UUID REFERENCES ksamds.function(id) ON DELETE CASCADE,
  task_id UUID REFERENCES ksamds.task(id) ON DELETE CASCADE,
  confidence_score NUMERIC(4,3),                  -- From embedding similarity
  PRIMARY KEY (function_id, task_id)
);

CREATE TABLE IF NOT EXISTS ksamds.ability_task (
  ability_id UUID REFERENCES ksamds.ability(id) ON DELETE CASCADE,
  task_id UUID REFERENCES ksamds.task(id) ON DELETE CASCADE,
  confidence_score NUMERIC(4,3),                  -- From embedding similarity
  PRIMARY KEY (ability_id, task_id)
);

-- =========================
-- Indexes for performance
-- =========================
-- Text search indexes
CREATE INDEX IF NOT EXISTS idx_occ_title_trgm ON ksamds.occupation USING gin (title gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_knowledge_name_trgm ON ksamds.knowledge USING gin (name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_skill_name_trgm ON ksamds.skill USING gin (name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_ability_name_trgm ON ksamds.ability USING gin (name gin_trgm_ops);

-- Occupation relationship indexes
CREATE INDEX IF NOT EXISTS idx_occ_knowledge_occupation ON ksamds.occupation_knowledge(occupation_id);
CREATE INDEX IF NOT EXISTS idx_occ_knowledge_knowledge ON ksamds.occupation_knowledge(knowledge_id);
CREATE INDEX IF NOT EXISTS idx_occ_knowledge_type ON ksamds.occupation_knowledge(type_id);
CREATE INDEX IF NOT EXISTS idx_occ_knowledge_level ON ksamds.occupation_knowledge(level_id);
CREATE INDEX IF NOT EXISTS idx_occ_knowledge_basis ON ksamds.occupation_knowledge(basis_id);

CREATE INDEX IF NOT EXISTS idx_occ_skill_occupation ON ksamds.occupation_skill(occupation_id);
CREATE INDEX IF NOT EXISTS idx_occ_skill_skill ON ksamds.occupation_skill(skill_id);
CREATE INDEX IF NOT EXISTS idx_occ_skill_type ON ksamds.occupation_skill(type_id);
CREATE INDEX IF NOT EXISTS idx_occ_skill_level ON ksamds.occupation_skill(level_id);
CREATE INDEX IF NOT EXISTS idx_occ_skill_basis ON ksamds.occupation_skill(basis_id);

CREATE INDEX IF NOT EXISTS idx_occ_ability_occupation ON ksamds.occupation_ability(occupation_id);
CREATE INDEX IF NOT EXISTS idx_occ_ability_ability ON ksamds.occupation_ability(ability_id);
CREATE INDEX IF NOT EXISTS idx_occ_ability_type ON ksamds.occupation_ability(type_id);
CREATE INDEX IF NOT EXISTS idx_occ_ability_level ON ksamds.occupation_ability(level_id);
CREATE INDEX IF NOT EXISTS idx_occ_ability_basis ON ksamds.occupation_ability(basis_id);

CREATE INDEX IF NOT EXISTS idx_occ_task_occupation ON ksamds.occupation_task(occupation_id);
CREATE INDEX IF NOT EXISTS idx_occ_task_task ON ksamds.occupation_task(task_id);
CREATE INDEX IF NOT EXISTS idx_occ_task_type ON ksamds.occupation_task(type_id);
CREATE INDEX IF NOT EXISTS idx_occ_task_environment ON ksamds.occupation_task(environment_id);
CREATE INDEX IF NOT EXISTS idx_occ_task_mode ON ksamds.occupation_task(mode_id);

CREATE INDEX IF NOT EXISTS idx_occ_function_occupation ON ksamds.occupation_function(occupation_id);
CREATE INDEX IF NOT EXISTS idx_occ_function_function ON ksamds.occupation_function(function_id);
CREATE INDEX IF NOT EXISTS idx_occ_function_environment ON ksamds.occupation_function(environment_id);
CREATE INDEX IF NOT EXISTS idx_occ_function_physicality ON ksamds.occupation_function(physicality_id);
CREATE INDEX IF NOT EXISTS idx_occ_function_cognitive ON ksamds.occupation_function(cognitive_id);

-- Core relationship indexes for graph traversal
CREATE INDEX IF NOT EXISTS idx_knowledge_skill_knowledge ON ksamds.knowledge_skill(knowledge_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_skill_skill ON ksamds.knowledge_skill(skill_id);

CREATE INDEX IF NOT EXISTS idx_skill_ability_skill ON ksamds.skill_ability(skill_id);
CREATE INDEX IF NOT EXISTS idx_skill_ability_ability ON ksamds.skill_ability(ability_id);

CREATE INDEX IF NOT EXISTS idx_knowledge_function_knowledge ON ksamds.knowledge_function(knowledge_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_function_function ON ksamds.knowledge_function(function_id);

CREATE INDEX IF NOT EXISTS idx_function_task_function ON ksamds.function_task(function_id);
CREATE INDEX IF NOT EXISTS idx_function_task_task ON ksamds.function_task(task_id);

CREATE INDEX IF NOT EXISTS idx_ability_task_ability ON ksamds.ability_task(ability_id);
CREATE INDEX IF NOT EXISTS idx_ability_task_task ON ksamds.ability_task(task_id);