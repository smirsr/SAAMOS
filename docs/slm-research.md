\# SLM Research and Recommendation



\## Target hardware

\- TI SK-AM62 Edge AI Development Kit: quad-core Arm Cortex-A53, no GPU/NPU for llama.cpp.

\- RAM depends on the exact variant (the SK-AM62-LP has 2 GB LPDDR4). Exact board and RAM still to be confirmed.

\- The SLM shares RAM with Linux, the speech recognizer, the vision model, and text-to-speech, so the budget for it is roughly 500 MB or less (estimate).



\## Provided resources (from project resource table)

\- TI SK-AM62 Edge AI Development Kit with TI Processor SDK Linux

\- llama.cpp provided as the local inference engine

\- Python and C/C++ development environment

\- MicroSD card for the OS, models, and application storage

\- Network access for development only; the final system must run fully offline



\## Candidates (all run fully offline via llama.cpp)

| Rank | Model | Params | Approx. quantized size | Notes |

|---|---|---|---|---|

| 1 | Qwen2.5-0.5B-Instruct | 0.5B | \~430-490 MB (Q3\_K\_M-Q4\_K\_M) | Recommended first choice: best balance of accuracy and size for our task |

| 2 | SmolLM2-360M-Instruct | 360M | \~230 MB (Q4) | Faster, lighter backup if the Qwen model is too slow or uses too much RAM |

| 3 | Qwen3-0.6B | 0.6B | \~510 MB (Q4) | Worth testing if memory allows; larger and slower than the top two |

| 4 | SmolLM2-135M-Instruct | 135M | \~90-180 MB | Smallest and fastest, but weakest at following instructions and producing JSON; last resort |



Ranking is based on expected accuracy for intent parsing versus the board's memory and speed limits. It should be confirmed with real tests on the board.



Sizes are approximate, from public GGUF listings.



\## Recommendation

Start with \*\*Qwen2.5-0.5B-Instruct (Q4\_K\_M) via llama.cpp\*\*, with \*\*SmolLM2-360M\*\* as the faster backup.



Why:

\- Our task is small: choose one of three intents and extract an object name (about 15 output tokens).

\- It runs fully offline once the model file is on the board.

\- It should fit in 2 GB alongside the other modules.

\- A rule-based keyword fallback remains the safety net if the SLM is too slow or returns invalid output.



Speed note: a Cortex-A53 Raspberry Pi Zero 2W ran SmolLM2-135M (Q4\_K\_M) at roughly 9-11 tokens/sec. Larger models will be several times slower, so real measurements on the board are needed. Laptop or phone numbers should not be used.



Implementation notes:

\- Keep the prompt short with a few examples; prompt processing is slow on this CPU.

\- Constrain output to JSON (llama.cpp supports grammars) to reduce invalid responses.

\- The model is stored on the microSD card, which reads slowly. Load it once at startup and keep it in memory instead of reloading for each command.



\## Still to verify



Needs the board or the hardware/deployment owner:

1\. Exact board variant and RAM.

2\. Python version on the TI Linux image.

3\. Free RAM at runtime with all modules loaded.

4\. Tokens/sec on the board for each candidate.

5\. Whether the provided llama.cpp is prebuilt for the board and which version it is.

6\. Model load time from the microSD card.



Handled on the software side:

7\. Calling llama.cpp through a subprocess (assumed; isolated in `slm\_engine.py` so it can be changed).

8\. Model licenses (to be checked on the model pages).

9\. Download all model files while the network is available, before the offline evaluation.



Until the board results are in, the rule-based fallback keeps the controller working.

