# SAAMOS Development Guide

## 1. Development environment

- Primary programming language: Python 3.
- Target platform: TI SK-AM62 running Linux, pending hardware verification.
- Development environments: Windows, macOS, or Linux.
- Editor: Visual Studio Code recommended.
- Version control: Git and GitHub.
- Shell scripting: Bash for Linux deployment.

The team must agree on a compatible Python version and pinned dependencies after the TI board has been initialized.

## 2. Subsystem ownership

| Member | Primary responsibility | Directory | Branch |
|---|---|---|---|
| Smridhi | Speech recognition and audio | `saam/audio/` | `feature/audio-interface` |
| Arianna | SLM and controller | `saam/core/` | `feature/slm-controller` |
| Marcos | Computer vision | `saam/vision/` | `feature/computer-vision` |
| Sarah | Hardware and deployment | `saam/hardware/` | `feature/hardware-integration` |

Shared files should be changed only after notifying the affected owners.

## 3. Git workflow

1. Pull the latest changes before starting work.
2. Develop on your assigned feature branch.
3. Make small, descriptive commits.
4. Push changes to GitHub regularly.
5. Open a pull request to merge into `main`.
6. Get a teammate to review the pull request before merging.
7. Keep `main` in a working state.

Do not commit directly to `main`.

## 4. Coding requirements

- Use meaningful function and variable names.
- Add docstrings for public functions.
- Use clear error handling.
- Do not hardcode local file paths.
- Use configurable device settings.
- Keep module interfaces consistent with `docs/integration-guide.md`.
- Add tests for critical functionality.
- Document new dependencies and setup instructions.

## 5. Offline operation

All final user-facing functions must operate without an internet connection.

Development may use internet access to download software packages, models, or development tools.

Cloud speech recognition, online inference APIs, and cloud-only text-to-speech services must not be runtime dependencies.

## 6. Definition of done

A task is complete when:

- The assigned functionality is implemented.
- The module can be tested independently.
- Relevant tests pass.
- The code follows the agreed interface.
- Dependencies and known limitations are documented.
- A pull request has been reviewed and merged.

## 7. Team communication

Daily updates should include:

- Completed work
- Current work
- Problems or blockers
- Help needed from teammates

Any blocker that could affect an integration milestone should be reported immediately.

## 8. Development milestones

**October 30:** Independent subsystem demonstrations.

**November 3:** Preliminary integrated pipeline tested.

**November 5:** Round 2 prototype presentation.

**November 13:** Embedded integration and offline functionality checkpoint.

**November 17:** Final feature freeze.

**November 20:** Final demonstration.

## 9. Security and project hygiene

- Never commit passwords, API keys, tokens, or private credentials.
- Do not commit virtual environments or large model binaries.
- Document model licenses and sources.
- Do not push private user recordings without permission.
- Prefer local processing and avoid unnecessary storage of voice or camera data.
