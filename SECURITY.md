# Security Policy

## Reporting a vulnerability
If you discover a security issue, please open a private report via GitHub
Security Advisories, or contact the maintainer directly rather than filing a
public issue.

## Scope and good practices
This is a research/portfolio computer-vision project. Even so:

- **Never commit** API keys, passwords, private keys, certificates, or `.env`
  files. `.gitignore` excludes common cases, but review diffs before pushing.
- Do not commit datasets or trained weights that may carry licensing or privacy
  constraints.
- Treat any camera feed as potentially sensitive; the realtime demo processes
  frames locally and does not transmit them.

## Supported versions
The `main` branch receives fixes. Tagged releases are provided as-is.
