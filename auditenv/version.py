"""Version constants. Any change to generated data or grading bumps a version."""

ENV_VERSION = "1.0.0"
GENERATOR_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0"
CONFIG_VERSION = "1.0"

MAX_STEPS = 100
PAGE_LIMIT = 50
DEFAULT_TIME_LIMIT_S = 60.0

MAX_FINDINGS = 30
MAX_EVIDENCE_IDS = 10
MAX_AFFECTED_IDS = 10
MAX_EXPLANATION_CHARS = 1000
MAX_ACTION_BYTES = 1_000_000  # length of one action as JSON text
MAX_ACTION_DEPTH = 64  # nesting of one action, checked on the text before parsing
STEP_COST = "0.002"  # reward cost per inspection step (decimal string, exact)
