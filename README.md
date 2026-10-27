# SAAMOS
### Secure Autonomous Assistant Monitor Operating System

**SAAM Vision | HackEdge 2026 — Track 3**

SAAMOS is an offline AI-powered accessibility assistant designed to help blind and visually impaired individuals interact with their immediate environment.

The project focuses on integrating small language models, computer vision, offline speech recognition, and spoken feedback on an embedded Linux platform.

## Core Features

- **Object Detection:** Identify selected everyday objects using a camera.
- **Object Localization:** Estimate whether a detected object is to the left, center, or right.
- **Offline Voice Recognition:** Accept spoken instructions without requiring an internet connection.
- **SLM-Based Intent Interpretation:** Interpret user commands and select supported system actions.
- **Audio Feedback:** Communicate object locations and system responses through speech.

## Target Hardware

- Texas Instruments SK-AM62 development board (pending hardware confirmation)
- USB camera
- Microphone
- Audio output device

## Team

| Member | Responsibility |
|---|---|
| Smridhi Subedi | Project Lead, Speech Recognition & Audio Interface |
| Arianna Alvarez | Small Language Model & System Architecture |
| Marcos Benitez | Computer Vision & Object Localization |
| Sarah Azeem | Hardware Integration, Testing & Accessibility |

## Project Structure

- `saam/core/` — Controller and intent interpretation
- `saam/vision/` — Camera, detection and localization
- `saam/audio/` — Speech recognition and audio feedback
- `saam/hardware/` — Board configuration and deployment
- `docs/` — Architecture, setup and testing documentation
- `tests/` — Software verification
- `demo/` — Demonstration resources

## Development Milestones

- **October 24, 2026:** Round 1 — Concept and video submission
- **November 5, 2026:** Round 2 — Prototype presentation
- **November 20, 2026:** Round 3 — Final demonstration

## Development Status

Early prototype development. Core subsystems are being developed independently before integration and deployment.

## Our Goal

Demonstrate an offline accessibility assistant that can understand a spoken request, identify a supported object in a stationary workspace, and communicate its approximate location through audio.
