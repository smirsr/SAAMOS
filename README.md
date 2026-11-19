
# SAAMOS — Secure Autonomous Assistant Monitor

**Offline Edge AI Assistant for People Who Are Blind or Have Low Vision**

UTiD HackEdge 2026 | Track 3: Offline Edge AI Assistant

## Overview

SAAMOS is an accessibility-focused, offline AI assistant designed to help people who are blind or have low vision identify objects and better understand their surroundings.

Built around the Texas Instruments SK-AM62B development board, SAAMOS combines computer vision, offline speech recognition, command interpretation, and spoken responses. Its core functionality operates locally without requiring Wi-Fi, cloud inference, or an internet connection during use.

Our goal is to increase independence by making environmental information accessible through a compact embedded system.

> **Disclaimer:** SAAMOS is a student prototype, not a certified mobility or navigation aid. Object detections may be inaccurate and should not be relied upon for safety-critical decisions.

## Features

| Feature | Implementation Status |
|---|---|
| Offline voice recognition using Vosk | Demonstrated |
| Object detection using SSD MobileNet V2 | Demonstrated |
| Find Mode — locate requested objects | Demonstrated |
| Scan Mode — continuously describe surroundings | Demonstrated |
| Spoken responses and directional feedback | Demonstrated |
| Stop and repeat commands | Demonstrated |
| TI SK-AM62B hardware deployment | Demonstrated |
| USB camera and microphone integration | Demonstrated |
| Wired speaker output | Demonstrated |
| OCR and text reading | Not verified in final integrated demo |
| Dedicated directional audio beeps | Not verified in final integrated demo |
| SW4 physical-button mode control | Hardware explored; full integration not verified |
| SW3 reset prevention | Not implemented; SW3 is a hardware reset |
| Small language model | Disabled in demonstrated workflow |

## System Architecture

SAAMOS processes user commands through the following pipeline:

1. **Audio Input:** A microphone captures the user's spoken command.
2. **Speech Recognition:** Vosk converts audio into text completely offline.
3. **Command Processing:** The controller interprets the requested action.
4. **Computer Vision:** A camera captures the environment, and a quantized SSD MobileNet V2 model identifies supported objects.
5. **Response Generation:** The controller uses detection results to form an appropriate response.
6. **Audio Feedback:** Text-to-speech communicates the response to the user.

The system supports object identification and relative directional feedback. It does not perform full navigation, depth estimation, or guaranteed obstacle avoidance.

## Hardware Components

- Texas Instruments SK-AM62B development board
- USB webcam for visual input
- USB microphone or webcam-integrated microphone
- Wired speaker/audio output
- Power supply and storage
- USB peripherals and connections for deployment

During testing, the webcam was accessible through Linux video devices, including `/dev/video0`, and successfully captured camera frames.

## Software Stack

**Programming Language:** Python 3

**Computer Vision:**
- OpenCV
- NumPy
- Quantized SSD MobileNet V2 COCO object-detection model
- TensorFlow Lite-compatible inference using `ai_edge_litert`

**Speech Recognition:**
- Vosk offline speech recognition
- Locally stored Vosk speech model

**Text-to-Speech:**
- `espeak-ng` on the TI board
- Platform-specific speech output for development/testing

**Development and Testing:**
- Git and GitHub
- Visual Studio Code
- Python automated testing
- Linux terminal and serial communication

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/smirsr/SAAMOS.git
cd SAAMOS
```

### 2. Create a Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

Install the Python packages listed in the dependency manifest included with the repository.

For example, if `requirements.txt` is present:

```bash
pip install -r requirements.txt
```

The installation must include the required speech recognition, computer vision, and inference dependencies.

### 4. Prepare offline models

Make sure the following model assets are available locally:

- Vosk offline speech recognition model
- Quantized SSD MobileNet V2 COCO `.tflite` model
- Matching object-class label file

Place the models in the directories expected by the project configuration.

These dependencies and model files must be installed before operating SAAMOS without internet access.

## Running SAAMOS on the TI Board

The demonstrated deployment used a preconfigured Python environment on the SK-AM62B.

Activate the hardware environment:

```bash
source ~/projects/SAAMOS-main-test/.venv-hardware/bin/activate
cd ~/projects/SAAMOS-main-test
```

Configure the application:

```bash
export SAAM_DISABLE_SLM=1
export SAAM_CAMERA=0
```

Launch the integrated offline demo:

```bash
python3 -m saam.audio.auto_integration_demo
```

The application connects speech recognition, controller logic, object detection, and spoken responses.

### Example Commands

- "Find my bottle."
- "What do you see?"
- "Scan the scene."
- "Repeat."
- "Can you stop."

Recognition accuracy and available responses depend on the camera view, supported model categories, and environment.

## Testing and Validation

SAAMOS underwent software testing, component-level hardware checks, and integrated system testing.

### Automated Software Testing

The November 18 development and integration work reported **250 passing automated tests** before the updated main branch was pushed.

Testing covered software functionality and integration behavior. Automated passing results do not guarantee performance across all physical environments.

Run the local Python checks with:

```bash
python3 -m compileall -q saam
python3 -m pytest -q
```

### Hardware Validation

The following components were exercised on the TI SK-AM62B:

- USB camera capture
- Microphone recording
- Offline Vosk speech recognition
- Wired speaker playback
- Computer vision inference
- Controller and voice-response integration

### Integrated Demonstration

The team demonstrated an offline interaction pipeline:

**Voice Input → Speech Recognition → Controller → Computer Vision → Spoken Response**

Find Mode and Scan Mode were also tested, including spoken guidance, stop behavior, and repeat interactions.

The system does not require an internet connection during its demonstrated operating workflow.

## Limitations and Future Development

SAAMOS is currently a functional research prototype with several opportunities for improvement.

**Current limitations:**
- Accuracy depends on lighting, camera placement, distance, and object visibility.
- Detection is limited to categories supported by the pretrained model.
- Camera and microphone device identifiers can vary.
- OCR is not verified as part of the final integrated demo.
- Dedicated directional audio beeps are not verified as fully integrated.
- Physical-button mode control requires additional integration and validation.
- The demonstrated command pipeline runs without an SLM.

**Future improvements:**
- More reliable tactile and physical-button interactions
- Fully integrated offline text recognition
- Improved spatial feedback through dedicated audio cues
- Smaller and more portable hardware configuration
- Broader environmental testing
- Additional accessibility-focused user evaluation

## Team Contributions

**Smridhi Subedi — Team Lead**
- System integration and audio development
- Hardware deployment and peripheral testing
- Diagnostics, validation, and testing
- Project coordination and final integration

**Ari — Controller and Command Processing**
- Command interpretation
- Controller logic
- Interaction behavior

**Marcus — Computer Vision**
- Object detection pipeline
- Vision model integration
- Camera-based detection functionality

**Sarah — Hardware Development**
- TI SK-AM62B setup
- Hardware bring-up
- Board deployment and testing

Responsibilities overlapped during deployment, debugging, and integration.

## Competition

SAAMOS was developed for **UTiD HackEdge 2026**, Track 3: Offline Edge AI Assistant.

Organized by the University of Texas at Dallas Department of Electrical and Computer Engineering in collaboration with Texas Instruments.

## Repository

https://github.com/smirsr/SAAMOS

## Project Status

SAAMOS has progressed from an initial concept to an offline embedded AI prototype with demonstrated camera, microphone, object detection, controller, and audio-response integration.

The team continues refining accessibility interactions, reliability, and hardware integration.
