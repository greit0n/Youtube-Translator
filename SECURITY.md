# Security Policy

This project is local-first: the extension talks only to a helper running on
`127.0.0.1`, and the helper processes audio locally. That does not remove the
need to protect local secrets.

## Sensitive Local Files

Never publish or attach these files in issues, pull requests, logs, screenshots,
or support requests:

- `helper/cookies.txt`
- `cookies.txt`
- `helper/hf_token.txt`
- `hf_token.txt`
- `helper/cache/`
- `helper/*.log`
- `helper/enroll/*`
- Any custom transcript, audio window, or voice enrollment file

`cookies.txt` contains YouTube login tokens. Treat it like a password.
`hf_token.txt` contains a HuggingFace access token. Enrolled voice clips are
personal biometric data.

## Reporting Security Issues

For now, report security-sensitive issues by opening a GitHub issue with only a
high-level description and no secrets or exploit details. Ask for a private
contact path if the report requires sensitive reproduction steps.

Once GitHub private vulnerability reporting is enabled for the repository, use
that path instead.

## Localhost Trust Boundary

The helper listens on `127.0.0.1:8765`. Do not expose this port publicly. If you
change host binding, CORS, or WebSocket permissions, treat it as a security
change and document the risk.
