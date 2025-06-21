SYSTEM_PROMPT = """ You are a helpful assistant who answers questions about database tables
by responding with SQL queries."""

USER_PROMPT_TEMPLATE = """Schema:
{0}

Question: {1}
SQL:
"""