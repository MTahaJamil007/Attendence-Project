# Hand-Raise Attendance System — Review & 1-Day Build Plan

> **Build status (2026-09-27): all four phases are built and verified on the real
> hardware.** 185 tests pass, the config contract is clean, the end-to-end
> pipeline runs, the application launches with both windows, a complete roll call
> exports a register, a workbook, per-decision telemetry and evidence images, and
> the calibration toolchain sweeps thresholds against labelled footage.
>
> Two things still need a human rather than more code: the **physical camera
> test** (highest remaining risk, and not a software risk), and a **real
> classroom recording** to calibrate against. Until that recording exists the
> thresholds are reasoned defaults validated on synthetic poses, not measured
> ones. See README.md.
>
> Phase 2 exposed a design bug in Rule A: it counted only *confirmed* raises, but
> two people never raise in the same frame, so the marginally earlier hand was
> always credited and the second was silently ignored. It now counts every hand
> that is up (confirmed or not) while only a confirmed track can be credited.
>
> The first TTS utterance also cost 2.5 s against 0.15 s for later ones - the same
> cold-start shape as the GPU clocks - so the speech engine is warmed at startup.
>
> Four things this plan did not anticipate, all found by measuring:
> 1. BoT-SORT's `gmc_method: sparseOptFlow` costs 23% of frame time and is useless
>    on a fixed camera — now disabled via `data/botsort_static.yaml`.
> 2. The camera is hard-capped at **15 FPS**, so the pipeline is camera-bound, not
>    GPU-bound. The frame-rate-independent design absorbs this.
> 3. A cold GPU runs at ~half speed (300 MHz idle vs 2100 MHz max, ~8 s to ramp),
>    so warm-up had to become sustained and moved to just before first use.
> 4. ultralytics 8.4 replaced `half=True` with `quantize=16`.
>
> Elbow extension also had to become a **multiplier** rather than a weighted term:
> as a weighted feature, a head-scratch scored 0.61 against a 0.55 threshold.

**Date:** 2026-09-27
**Target:** USB webcam faces the class. Laptop screen shows a roll number. That student raises a hand.
System detects the raise → **Present**. No raise within the window → **Absent**.

---

## Part 1 — Review of the existing `SmartRollCall.exe`

There was no source on disk, only `C:\Users\mtaha\Desktop\SmartRollCall.exe` (389 MB). I unpacked the
PyInstaller archive and recovered the app's structure, dependency set, and full default config.

### 1.1 What it actually is

| Item | Value |
|---|---|
| Entry script | `smart_roll_call.py` — single file, ~2000 lines |
| Python | 3.11 (`python311.dll`) |
| **Torch** | **`torch-2.7.1+cpu` / `torchvision-0.22.1+cpu`** |
| Detector | `ultralytics 8.4.158`, model `yolo26n-pose.pt` |
| GUI | `tkinter` + `ttk` + PIL/ImageTk (not a web UI) |
| TTS | `pyttsx3 2.99` + `comtypes`/`win32com` (SAPI) |
| Mic | `sounddevice 0.5.2` — RMS energy VAD, no speech-to-text |
| Output | `~/Documents/SmartRollCall Attendance/attendance_<ts>.csv` |

**Recovered class structure** (this design is sound and worth reusing):

- `VideoWorker` — capture + inference thread. Has `_ai_runtime_self_test` (runs `torchvision::nms`
  to catch the "operator torchvision::nms does not exist" error), `_discover_cameras` (scans indices
  0–7 across `CAP_DSHOW`/`CAP_MSMF`/`CAP_ANY`), `_open_selected`, camera hot-switching.
- `HandRaiseDetector` — `_side_raised(side)` over COCO-17 keypoints. Computes `torso`, `lift`,
  `above_shoulder`, `wrist_above_elbow`, `elbow_not_low`, `hand_high`, `plausible`. Boolean AND rule.
- `SpeechWorker` — queued pyttsx3 thread.
- `MicrophoneWorker` — docstring: *"Fast local voice-activity detector. It does not need Internet or
  speech-to-text."* Tracks `noise_floor`, `last_voice_at`.
- `RollDisplay` — separate `Toplevel`, F11 fullscreen, dark bg, huge roll number. **Student-facing
  display.** Good idea, keep it.
- `AttendanceApp` — state machine: `IDLE → CALLING → RESPONSE → RESULT → FINISHED`
  (`_begin → _response → _result → _advance → _finish`), plus `_manual`, `_undo`, `_export`.
- Roll-number speech optimizer: `parse_roll` → `assign_short_unique_calls` → `spoken_component` /
  `number_words` / `letters_spoken`. Docstring: *"Shortest sensible unique spoken call. Full roll
  remains visible on screen."* i.e. it says **"seven"**, not "ess-pee-two-five-bee-see-ess-zero-zero-seven",
  while the screen shows `SP25-BCS-007`. **This is genuinely clever — reimplement it.**

### 1.2 Recovered default config (exact values from the bytecode)

```
camera_index             = -1      # auto
camera_scan_max_index    = 7
camera_width             = 1920
camera_height            = 1080
model                    = "yolo26n-pose.pt"
imgsz                    = 768
person_conf              = 0.25
keypoint_conf            = 0.35
min_person_height_px     = 55
wrist_lift_ratio         = 0.15
reaction_delay_sec       = 0.45
response_window_sec      = 4.0
hand_only_hold_sec       = 0.85
hand_voice_hold_sec      = (shorter than hand-only)
max_raise_gap_sec        = 0.28
voice_match_window_sec   = 1.25
voice_rate               = 155
mic_threshold_multiplier = 2.8
mic_min_rms              = 0.012
```

### 1.3 Measured field results (8 real runs, from the saved CSVs)

| Run | Rows | Present | Max gap between marks |
|---|---|---|---|
| 11:59:59 | 10 | **0** | 6 s |
| 12:00:34 | 10 | 1 | **40 s** |
| 12:01:50 | 10 | 6 | 5 s |
| 12:02:10 | 10 | 6 | **22 s** |
| 12:13:59 | 20 | 6 | **52 s** |
| 12:21:35 | 20 | 0 | *(all rows `Pending` — exported with no data)* |
| 12:27:31 | 20 | 7 | **27 s** |
| 12:29:14 | 20 | 2 | 6 s |

Nominal cadence is 5 s per student (4.0 s window + 0.45 s reaction + result hold). Successful
detections land in 1–3 s; every Absent takes exactly ~5 s. **Effective recall is roughly 30%, and
wildly inconsistent run to run.**

### 1.4 Root causes — why it was inaccurate

**① CPU-only PyTorch on a machine with a CUDA GPU.** This is the headline bug.
The build bundles `torch-2.7.1+cpu`, but this laptop has a **Quadro RTX 3000, 6 GB, driver 580.92**.
`yolo26n-pose` at `imgsz=768` on CPU gives roughly 3–8 FPS.

**② The hold rule is mathematically unsatisfiable at low FPS.** It requires `hand_only_hold_sec = 0.85 s`
of continuous detection, tolerating gaps of only `max_raise_gap_sec = 0.28 s`. At 5 FPS the frame
interval is 0.2 s, so **a single dropped detection creates a 0.4 s gap > 0.28 s and resets the chain.**
You need ~4–5 *consecutive* clean frames. That is why so many genuine raises were logged
`Timeout: no valid hand`.

**③ Multi-second pipeline freezes turn into false Absents.** The 52 s, 40 s, 27 s and 22 s gaps are the
inference thread stalling. Meanwhile the state machine kept running on its own timer and marked
everyone Absent. **There is no liveness check — an infrastructure failure was written into the
attendance record.** This single bug ruined three of eight runs.

**④ Nano model + too-small `imgsz` for a whole-class shot.** 1080p downsampled to 768 is a 2.5×
reduction. A back-row student 120 px tall becomes 48 px, and their wrist/elbow keypoints become
noise that `keypoint_conf=0.35` then rejects. `min_person_height_px=55` admits people the nano
model cannot reliably pose.

**⑤ No identity binding — it counts hands, not people.** `raised_count` is a frame-wide total. *Anyone*
raising a hand marks the called student present. One student can answer for the whole class.

**⑥ The microphone feature actively reduced accuracy.** It is an RMS energy threshold
(`2.8 × noise_floor`, floor `0.012`), *not* speech recognition — a cough, a chair scrape, or the teacher
talking passes as "voice." And a voice hit swaps in the *shorter* `hand_voice_hold_sec`, so **ambient
noise made it easier to be marked Present.** The `Hand + voice` evidence label implies verification
that never happened.

**⑦ Export has no guards.** The 12:21:35 file is 20 rows of `Pending` with empty timestamps.

### 1.5 What was good and should be kept

- Overall threading split (video / speech / mic) and the phase state machine.
- The separate fullscreen student-facing `RollDisplay`.
- The shortest-unique spoken call optimizer.
- Camera auto-discovery across multiple Windows backends.
- The `torchvision::nms` runtime self-test.
- Manual Present/Absent/Undo override and the CSV schema.

---

## Part 2 — Build from scratch, or fork a GitHub project?

**Neither extreme. Rewrite the app layer; stand on Ultralytics for the ML.**

**Do not fork a GitHub attendance repo.** What exists in this space is either (a) face-recognition
attendance systems — a different problem with different failure modes, or (b) MediaPipe "raise your
hand" demos — one person, close range, no roll-call flow, no evidence trail, no roster. Adapting
either costs more than it returns and you inherit their bugs along with their design.

**Do not write pose estimation from scratch.** Not in one day, not ever for this.

**Do stand on Ultralytics YOLO-pose.** It is pretrained on COCO-pose, ships BoT-SORT/ByteTrack
tracking built in, and exports to ONNX/TensorRT. *That* is the "use someone else's project" move —
at the library layer, where it pays off.

**And reuse the existing app's own architecture as the blueprint** (Section 1.1/1.5). Its structure was
not the problem; the runtime and the decision rule were. Rebuilding on that recovered skeleton saves
hours of design.

Net: **~800–1000 lines of your own code**, on a mature GPU-accelerated pose backend.

---

## Part 3 — The build plan

### Phase 0 — Environment (45 min) — do this first, verify before writing app code

Python 3.14.7 is already installed and **the whole dependency chain has cp314 Windows wheels**
(verified against PyPI / download.pytorch.org): torch cu126 + cu130 up to 2.14.0, `lap`, `scipy`,
`pillow`, `onnxruntime-gpu`; `sounddevice`/`pyttsx3`/`comtypes`/`openpyxl` are pure-Python;
`opencv-python` ships `abi3` wheels. Fall back to Python 3.12 only if you hit an import wall.

```bash
py -3.14 -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install ultralytics opencv-python sounddevice pyttsx3 openpyxl
```

`cu126` is the safe choice for **Turing sm_75** (Quadro RTX 3000 = TU106, compute 7.5). `cu130` also
supports Turing and driver 580.92 is new enough, but cu126 is the better-trodden path.

**Startup gate — extend the old `_ai_runtime_self_test`:**
1. `torch.cuda.is_available()` must be `True`; print `torch.cuda.get_device_name(0)`.
2. Keep the `torchvision::nms` check.
3. **Refuse to start a session on CPU** — show a hard error instead. Silent CPU fallback is exactly
   how the old build shipped broken.

**Also in Phase 0: do a 10-minute physical camera test before any coding.** Mount the webcam where
it will actually live and confirm every seat's upper body is visible and ≥150 px tall. Room geometry
is the one failure no model can fix.

### Phase 1 — Perception core (2.5 h) — where the accuracy comes from

**1a. Model and resolution.** Benchmark on a real classroom frame and pick the largest config that
holds **≥ 25 FPS**:

- sizes: `yolo11s-pose`, `yolo11m-pose`, `yolo26s-pose`, `yolo26m-pose`
- `imgsz`: 960 and 1280
- `half=True` (FP16 — free 1.5–2× on Turing tensor cores)

`yolo11m-pose` is the known-good fallback if a YOLO26 pose weight is not downloadable.
At 1280/FP16 a medium pose model uses ~1.5–2 GB, comfortable in 6 GB.

> **Raising `imgsz` from 768 → 1280 is the single biggest recall win after enabling the GPU**, because
> back-row students are small and keypoint accuracy scales with pixels on target.

**1b. Replace the brittle boolean + hold-chain rule.**

- Compute a **graded raise score in [0,1]** per person per side, not a boolean AND. Inputs: wrist-vs-shoulder
  height normalized by torso length (shoulder→hip), wrist vs elbow, wrist vs nose, elbow angle, and the
  keypoint confidences. Grading removes the cliff-edge behavior at the threshold.
- **Hysteresis:** fire at a high threshold, release at a lower one. No more flicker.
- **EMA smoothing per track ID**, not a fragile consecutive-frame chain.
- **Make tolerance FPS-aware:** `gap_tolerance = max(0.30, 3.0 / measured_fps)`. Never hard-code 0.28 s.
- **Widen the window:** `response_window_sec` 4.0 → **6.0**, and extend it by 1.5 s if a partial raise is
  in progress at the deadline ("I see you — hold it").

**1c. Tracking-based identity gating** — the anti-cheat layer the old app lacked.

Use `model.track(persist=True, tracker="botsort.yaml")` for stable IDs across the session.

- **Rule A — exactly one hand.** If ≥2 distinct tracks are raised during a call, do **not** credit
  silently. Show `AMBIGUOUS — 2 HANDS UP` and require a teacher tap. Closes the "friend raises for
  an absent student" hole.
- **Rule B — one credit per track per session.** Once track #7 has been credited, it cannot be
  credited again. One person cannot answer for five absentees. Cheap to implement, enormous
  trust gain.
- **Rule C — seat-zone lock (optional).** Store the credited person's bbox centroid. On later periods,
  flag a raise for the same roll coming from a very different zone. Also builds a seat map over time.

**1d. Pipeline liveness watchdog** — fixes root cause ③.

`VideoWorker` stamps `last_infer_at`. The state machine **must not mark Absent** if
`now - last_infer_at > 0.75 s`. Instead: pause the countdown, show `CAMERA STALLED`, resume when
frames return. **Never convert an infrastructure failure into an attendance record.** This alone
would have saved three of the eight recorded runs.

### Phase 2 — Roll-call flow and UI (2 h)

Reuse the recovered two-window design:

- **Teacher console:** live annotated video, roster `Treeview`, controls (Load CSV / Paste Rolls /
  Start Camera / Switch Camera / Start Attendance / Pause / Export), status bar with
  `People n | Raised n | FPS n`.
- **Student display** (`Toplevel`, fullscreen on the projector): huge roll number + name, plus two
  additions the old app lacked:
  - a **countdown ring/bar**, and
  - **live closed-loop feedback**: `HAND DETECTED — HOLD…` → `✓ PRESENT`.
    The old app showed hold progress on the *teacher* side only. Students who can see whether
    they've been seen raise higher and hold longer — this measurably lifts real-world hit rate.
- Reimplement the shortest-unique spoken call (speak "seven", display `SP25-BCS-007`).
- Keep manual **Present / Absent / Undo**. Non-negotiable in a real classroom.
- Phases: `IDLE → CALLING → RESPONSE → RESULT → FINISHED`, plus new **`STALLED`** and **`AMBIGUOUS`**.

Roster loading: read as `utf-8-sig` (the sample CSV has a BOM) and `.strip()` every cell — the sample
`sample attendence.csv` has a leading space on ` SP25-BCS-003`.

### Phase 3 — Evidence and export (1 h) — what makes a mark defensible

- On every mark, save a **cropped JPEG of the credited person at the detection instant** to
  `evidence/<session>/<roll>.jpg`, plus raise score, track ID, FPS and bbox. The old app stored only
  the string `"Hand only"`. With an image, a disputed absence is resolvable.
- CSV: keep the existing schema (`Roll Number, Name, Status, Marked At, Evidence`) for compatibility,
  and add `TrackID, Score, EvidenceImage`.
- XLSX export via `openpyxl` with conditional formatting — teachers want this.
- **Guard export:** warn/refuse when any row is still `Pending` (root cause ⑦).
- Write `session.json` with the full config plus per-call telemetry, so Phase 4 can tune from real data.

### Phase 4 — Calibration and validation (1.5 h) — do not skip this

**This is what separates "works in my chair" from "works in the room."**

1. Record one 3–5 minute video of the actual classroom with known ground truth: volunteers raising
   hands on cue, **plus deliberate negatives** — stretching, scratching head, waving, raising a phone,
   leaning on an elbow.
2. Run offline sweeps over `wrist_lift_ratio`, hold time, EMA alpha, hysteresis thresholds, `imgsz`
   and model size. Report **recall and precision per setting**.
3. Pick the operating point and hard-code it as the default.

**Targets: recall ≥ 98% on genuine raises, false positives ≤ 1% on distractor motions.**

An hour of this is worth more than any model upgrade.

### Phase 5 — Packaging (30 min, optional)

Run from the venv for the demo. If you must ship an `.exe`, PyInstaller works — but the CUDA torch
build will be **~2–4 GB** (the CPU build was already 389 MB). For a one-day deliverable, ship the
venv plus a `.bat` launcher instead.

---

## Timeline — 1 day ≈ 9 working hours

| Time | Phase |
|---|---|
| 0:00 – 0:45 | Phase 0 — env, CUDA verified, physical camera test |
| 0:45 – 3:15 | Phase 1 — GPU, model/imgsz benchmark, scoring + hysteresis, tracking gates, watchdog |
| 3:15 – 5:15 | Phase 2 — flow + two-window UI |
| 5:15 – 6:15 | Phase 3 — evidence + export |
| 6:15 – 7:45 | Phase 4 — calibration on real footage |
| 7:45 – 8:15 | Buffer / bug fixes |
| 8:15 – 9:00 | Dry run in the real room, final tuning |

---

## Proposed file layout

```
Attendence Project/
├─ PLAN.md
├─ requirements.txt
├─ run.bat
├─ config.json                 # all thresholds, no magic numbers in code
├─ src/
│  ├─ main.py                  # entry, startup self-test gate
│  ├─ runtime_check.py         # CUDA + torchvision::nms gate
│  ├─ video_worker.py          # capture + track(), FPS + last_infer_at stamps
│  ├─ raise_detector.py        # graded score, hysteresis, EMA per track
│  ├─ gating.py                # Rules A/B/C — one hand, one credit, seat zone
│  ├─ roster.py                # CSV load (utf-8-sig, strip), short unique calls
│  ├─ speech.py                # pyttsx3 worker
│  ├─ session.py               # state machine incl. STALLED + AMBIGUOUS
│  ├─ evidence.py              # crop JPEG + telemetry
│  ├─ export.py               # CSV + XLSX, Pending guard
│  └─ ui/
│     ├─ console.py            # teacher window
│     └─ roll_display.py       # student fullscreen window
└─ tools/
   └─ calibrate.py             # Phase 4 offline sweep over recorded video
```

---

## Risks and mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| **Camera placement / room geometry** | **Highest** | Physical test in Phase 0, *before* coding. No model fixes an occluded back row. |
| Python 3.14 is newer ground for ultralytics | Medium | Chain verified to have cp314 wheels; fall back to 3.12 if imports break. |
| TensorRT export flakiness | Low | Optional stretch only. Skip if tight on time. |
| 6 GB VRAM | Low | Medium pose model at 1280 FP16 ≈ 1.5–2 GB. Don't load a second model. |
| Lighting / backlit windows | Medium | Test in real lighting; raise camera exposure, avoid shooting into windows. |

## Cut list, in order, if you fall behind

Cut: TensorRT → XLSX export → seat-zone lock (Rule C) → voice entirely.

**Never cut:** GPU torch, the liveness watchdog, evidence images, manual override, Phase 4 calibration.

## On the microphone / voice feature

**Drop it, or demote it to advisory-only metadata that never shortens the hold requirement.**
As built it was an accuracy *regression* (root cause ⑥): RMS energy is not speech recognition, and a
noise hit made it *easier* to be marked Present while labelling the record `Hand + voice`. Real voice
confirmation needs a keyword spotter (openWakeWord / Vosk) — out of scope for one day.

---

## The four changes that matter most

If you only do four things:

1. **Install CUDA PyTorch** (`cu126`) and hard-fail on CPU. Fixes ~3–8 FPS → 25–40 FPS.
2. **Raise `imgsz` to 1280** and move up from the nano model. Fixes back-row recall.
3. **Add the liveness watchdog.** Stops infrastructure stalls from being written as Absent.
4. **Calibrate on real footage with real negatives** (Phase 4). Turns "seems to work" into a number.
