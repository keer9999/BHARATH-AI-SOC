# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| main    | ✅ Active support  |
| < 1.0   | ❌ No support      |

## Reporting a Vulnerability

The Bharat AI-SoC PS2 team takes security vulnerabilities seriously. We appreciate your efforts to responsibly disclose your findings.

**Please do NOT open a public GitHub issue for security vulnerabilities.**

### How to Report

1. **Email**: Send a detailed report to the project maintainers (open a [private security advisory](https://github.com/keer9999/bharath-ai-soc/security/advisories/new) on GitHub).

2. **Include in your report**:
   - A description of the vulnerability and its potential impact
   - Step-by-step instructions to reproduce the issue
   - Any proof-of-concept code (if applicable)
   - Your suggested fix (if you have one)
   - Your name/handle for attribution (optional)

### What to Expect

- **Acknowledgement**: Within 48 hours of your report
- **Status Update**: Within 7 days with our assessment
- **Resolution Timeline**: We aim to resolve critical issues within 30 days
- **Credit**: We will credit you in the security advisory (unless you prefer anonymity)

## Security Considerations for This Project

### On-Device AI Model Files

- GGUF model files should be downloaded only from trusted sources (Meta official / Hugging Face official)
- Verify SHA-256 checksums of downloaded models before use
- Never load model files from untrusted paths

### Sensor Data & Serial Communication

- Validate all data received over UART from the Pico W
- Sanitize sensor readings before database insertion
- Use parameterised SQLite queries (already enforced in `database.py`)

### Web Application

- The Flask web app is intended for **local network use only**
- Do not expose port 5000 to the public internet without authentication
- If deploying externally, use a reverse proxy (nginx) with TLS and authentication
- API endpoints do not include authentication by default — add it for production

### FPGA / Zynq-7000

- Vivado project files may contain IP core license references — do not share proprietary IP
- AXI bus communication is unencrypted — ensure physical security of the board

## Scope

The following are **in scope** for security reports:

- SQL injection in `database.py`
- Path traversal in report generation or file serving
- Remote code execution via crafted image inputs or serial data
- Credential exposure in configuration files
- Unsafe deserialization of model or data files

The following are **out of scope**:

- Vulnerabilities in third-party dependencies (report to upstream)
- Physical access attacks to hardware
- Denial-of-service on the local web server in a trusted network environment

## Preferred Languages

We accept security reports in English or Hindi.

---

*This security policy follows responsible disclosure best practices.*

