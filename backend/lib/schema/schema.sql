-- ============================================================
-- TOKENS
-- ============================================================
CREATE TABLE IF NOT EXISTS tokens (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    category TEXT,
    active BOOLEAN DEFAULT 1
);

-- ============================================================
-- RULES
-- ============================================================
CREATE TABLE IF NOT EXISTS rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_type TEXT NOT NULL CHECK(rule_type IN ('classify', 'profile', 'quantity', 'assign', 'modifier')),
    description TEXT NOT NULL,
    priority INTEGER DEFAULT 0,
    payload_json TEXT,
    active BOOLEAN DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_rules_type_priority 
    ON rules(rule_type, priority DESC) 
    WHERE active = 1;

-- ============================================================
-- RULE TOKENS (triggers and results)
-- ============================================================
CREATE TABLE IF NOT EXISTS rule_tokens (
    rule_id INTEGER NOT NULL,
    token_id INTEGER NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('trigger', 'result')),
    FOREIGN KEY(rule_id) REFERENCES rules(id) ON DELETE CASCADE,
    FOREIGN KEY(token_id) REFERENCES tokens(id),
    PRIMARY KEY (rule_id, token_id, role)
);

CREATE INDEX IF NOT EXISTS idx_rule_tokens_rule 
    ON rule_tokens(rule_id);

CREATE INDEX IF NOT EXISTS idx_rule_tokens_token 
    ON rule_tokens(token_id);
