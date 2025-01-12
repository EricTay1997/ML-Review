# Agents

- An agent is a system that uses an LLM as its engine, and it has access to functions called tools.
- It usually has these components:
  - An LLM to power your agent
  - A system prompt: What the LLM engine will be prompted with to generate its output
    - This forms the template for how the LLM approaches a problem
  - A toolbox from which the agent pick tools to execute. 
    - A tool consists of an input schema, and a function to run, e.g. a Python function.
  - A parser to extract from the LLM output which tools are to call and with which arguments
- Multi-agents, where having agents with separate tool sets and memories allows us to achieve efficient specialization.