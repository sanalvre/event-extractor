# Security Policy

## Reporting a vulnerability

If you discover a security issue, please report it privately to the maintainer instead of opening a public issue with exploit details.

When reporting, include:

- affected version or commit
- a short description of the issue
- reproduction steps if available
- impact assessment

## Secret handling

This project uses local environment variables for credentials such as:

- `OPENROUTER_API_KEY`
- `GROQ_API_KEY`
- `MEMORIES_API_KEY`
- `HF_TOKEN`

Do not commit real credentials to the repository.

## Operational guidance

- run the web app on localhost unless you add your own authentication and network controls
- do not expose the local FastAPI app publicly by default
- keep generated artifacts out of version control
- rotate any credential immediately if it is ever pasted into a shell history, log, or tracked file
