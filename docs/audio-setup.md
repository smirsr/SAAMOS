# SAAMOS — Offline Audio Setup Guide

## What this is for

I wrote this guide so the rest of our team can set up the audio part of SAAMOS without having to redo my setup. Our audio prototype records microphone input, converts speech to text locally using Vosk, checks a few supported commands, and speaks a response.

The audio system was developed and tested on macOS first. We still need to confirm microphone input and speaker output on the TI SK-AM62B Linux board.

## What we need

- Our SAAMOS repository and a Python environment compatible with the installed packages (my development setup used Python 3.11)
- A microphone and a working speaker or audio output
- Python packages: `numpy`, `sounddevice`, `vosk`, and `pytest` (for testing)
- Vosk English model: `vosk-model-small-en-us-0.15`
- On Linux, the `espeak-ng` program for speech output

**Internet is only needed to obtain dependencies and the model.** Once everything is installed locally, recognition and speech output do not need an internet connection.

## 1. Open the repository

If you already have the repository, open a terminal in its root folder (`SAAMOS`) and switch to the branch containing the audio code:

```bash
git fetch origin
git switch feature/audio-interface
git pull --ff-only origin feature/audio-interface
```

If you are working on shared `main` instead, these features need to be merged first. Don't switch branches if you have uncommitted work without saving it.

## 2. Create and activate a Python environment

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy sounddevice vosk pytest
```

If the project has an up-to-date dependency file, prefer installing from that file rather than manually installing the packages above.

On some Linux systems, `sounddevice` also needs the system PortAudio library installed. The installation command depends on the Linux distribution and what is already included in the board image.

## 3. Download the offline Vosk model

From the SAAMOS repository root:

```bash
mkdir -p models
curl -fL https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip -o models/vosk-model-small-en-us-0.15.zip
unzip models/vosk-model-small-en-us-0.15.zip -d models/
```

The extracted files should be directly inside:

```text
SAAMOS/
└── models/
    └── vosk-model-small-en-us-0.15/
        ├── am/
        ├── conf/
        ├── graph/
        └── ivector/
```

Verify with:

```bash
ls models/vosk-model-small-en-us-0.15/
```

Do **not** commit the model files to GitHub. The `.gitignore` should ignore `models/vosk-model-*/` and the downloaded ZIP file. You can delete the downloaded ZIP after verifying extraction:

```bash
rm models/vosk-model-small-en-us-0.15.zip
```

Official model catalog: https://alphacephei.com/vosk/models

## 4. Test the Python modules

From the SAAMOS root folder, with `.venv` activated:

```bash
python -m pytest tests/ -v
```

The Mac development version passed 10 tests on November 3, 2026. This does not guarantee success on every computer or the TI board; run the tests in each environment.

## 5. Check microphones and try the audio demo

List available devices:

```bash
python -m sounddevice
```

Run the microphone module by itself:

```bash
python -m saam.audio.microphone
```

Run the offline recognition test:

```bash
python -m saam.audio.stt
```

Run the complete audio demo:

```bash
python -m saam.audio.demo
```

Wait for `SAAMOS is listening...` before speaking. Try saying “hello,” “what can you do,” “help,” or “stop.” The terminal prints the words Vosk actually recognized; short commands can sometimes be misheard. Press **Control+C** to exit manually.

## 6. Notes for macOS and Linux

- **macOS:** The text-to-speech module uses the built-in `say` command.
- **Linux / TI board:** The text-to-speech module expects `espeak-ng` to be installed and a usable sound output device. Test this once the actual board is available.
- Recognition uses **mono 16-bit audio at 16,000 samples per second**. Check that the microphone accepts this format if recording fails.
- The Vosk model path is fixed in `saam/audio/stt.py`; do not rename the extracted model folder without updating that code.
- Running the demo on a Mac proves the basic program works; it does **not** prove the TI board's sound hardware, Linux packages, or audio routing work yet.

## 7. Using the TI board without internet

Once the board boots into Linux and we can use its terminal:

1. On an internet-connected computer, download the Vosk ZIP and obtain Linux-compatible dependencies for the board's architecture and operating-system image.
2. Copy the SAAMOS repository, the extracted model folder, and any required offline installation files to a USB flash drive.
3. Mount the USB drive on the board and copy the files into the board's SAAMOS folder. Keep the model at `models/vosk-model-small-en-us-0.15/`.
4. Install the appropriate Python/system dependencies using the board's supported installation method. **Do not assume Mac-installed Python packages can be copied to Linux.**
5. Connect a supported microphone and speaker; check Linux sound devices and test `espeak-ng` before running the complete demo.
6. Run the tests, then `python -m saam.audio.demo`.

These are preparation instructions rather than a verified TI-board deployment procedure. Exact Linux package commands will depend on the board image, Python version, available repositories, and attached peripherals.

## 8. If something goes wrong

| Problem | What to check |
| --- | --- |
| `ModuleNotFoundError` | Activate `.venv` and install the missing Python package. |
| `FileNotFoundError` for the model | Check the model folder name and `models/` location. |
| No microphone detected | Connect a supported device, confirm permissions and list devices with `python -m sounddevice`. |
| Vosk prints the wrong word | Look at `SAAMOS heard:`; speak a slightly longer phrase in a quiet location. |
| Speech output fails on Linux | Check that `espeak-ng` is installed and that Linux can play sound through the selected output device. |
| All tests pass but the demo fails | The tests also use simulated input, so separately check real microphone and speaker hardware. |

## 9. My current status

- Microphone recording: working on my Mac
- Offline Vosk speech recognition: working on my Mac
- Voice-command demo with spoken output: working on my Mac
- Automated tests: 10 passing on my Mac
- Linux text-to-speech command-selection: tested with a simulated Linux environment
- TI SK-AM62B deployment: **not yet tested**

This guide should be updated with the board-specific installation steps after the hardware is available.
