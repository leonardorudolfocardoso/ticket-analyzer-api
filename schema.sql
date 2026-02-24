CREATE TABLE ticket_classifications (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ticket_id       TEXT,
    text            TEXT NOT NULL,
    category        TEXT NOT NULL,
    priority        TEXT NOT NULL,
    sentiment       TEXT NOT NULL,
    confidence      FLOAT NOT NULL,
    suggested_response TEXT NOT NULL,
    reasoning       TEXT NOT NULL,
    model           TEXT NOT NULL,
    latency_ms      INTEGER NOT NULL,
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
