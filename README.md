# Hand-Raise Attendance System

A roll number appears on screen, the student raises a hand, a USB webcam facing the
class detects it, and the student is marked Present. No raise inside the response
window means Absent.

See [PLAN.md](PLAN.md) for the full build plan and the review of the predecessor
(`SmartRollCall.exe`) whose defects this design exists to fix.

**Status: all four phases are built and verified on the real hardware.**

225 tests pass, the config contract check is clean, the end-to-end pipeline runs,
the application launches with both windows, a complete roll call exports a
register, a workbook, per-decision telemetry and evidence images, and the
calibration toolchain sweeps thresholds against labelled footage.

Two things still need *you*, not more code: a real classroom recording to
calibrate against, and the physical camera test. See
[What still needs doing](#what-still-needs-doing).

### Latest revision: the voice calls every student, and a slight raise counts

* **The machine now reads each registration number aloud** - "S P two five,
  B C S, zero zero seven" - and the student answers by raising a hand. Speech
  existed before but was silent after the first sentence: pyttsx3 2.99 plays only
  its first `runAndWait()`, and the startup announcement used it up. Speech now
  drives Windows SAPI directly. See [The roll-call voice](#the-roll-call-voice).
* **A slight raise counts.** Before, the elbow had to reach shoulder height -
  in practice the whole arm went up. Now a hand held up beside the head counts
  from about chin height with the elbow left down. A hand at that height *in
  front of* the face (chin-rest, phone, hand on cheek) still does not. See
  [Slight raises](#slight-raises).
* Two bugs fixed on the way. A student who raised while the (now 4-5 s) number
  was still being read was treated as a hand "parked" from the previous call and
  never credited. And Pause did not stop the clock, so resuming after a long
  pause marked the student Absent on the spot.

---

## Setup

One command, from the project root:

```bash
setup.bat
```

It creates `.venv`, installs **CUDA** PyTorch, installs the remaining
dependencies, and runs the runtime gate.

The CUDA index is not optional:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Installing torch from plain PyPI gets the CPU-only build. That is precisely what
the predecessor shipped (`torch-2.7.1+cpu`), and on a whole-class frame it runs at
3-8 FPS, which makes any hold-based decision rule unsatisfiable. `cu126` is the
right choice for this machine's **Quadro RTX 3000** (Turing, sm_75, 6 GB).

The torch wheel is ~2.44 GB, so the first install takes a while.

## Verify, in this order

```bash
run.bat app          # the application
run.bat check        # runtime gate: CUDA, torchvision::nms, FP16
run.bat test         # config contract + 225 tests (no GPU or camera needed)
run.bat ui           # UI construction/rendering tests only
run.bat voice        # speak a few calls and confirm the audio really plays
run.bat cameras      # list every working camera
run.bat benchmark    # measure FPS, pick model + imgsz
run.bat smoke        # headless end-to-end pipeline check (no window, RDP-safe)
run.bat preview      # live detector view - run this in the real classroom
```

`python tools/check_config.py` additionally verifies that code and `config.json`
agree, and sanity-checks the tuned values against each other.

### Do the physical camera test before trusting any threshold

Mount the webcam where it will actually live, run `run.bat preview`, and confirm
every seat you care about gets a skeleton drawn on it. If the back row has no
skeleton, that is a camera-placement problem and no threshold will fix it. This is
the highest risk in the project and it is not a software risk.

---

## What is built

```
config.json              every tunable; no thresholds are hard-coded in src/
src/
  config.py              dotted-path config loader with deep-merge override
  runtime_check.py       Phase 0 gate: CUDA + torchvision::nms + FP16 self-tests
  geometry.py            COCO-17 constants, ramps, torso-normalized measurements
  raise_detector.py      graded raise scoring + FPS-independent temporal filter
  gating.py              Rules A/B/C - single hand, one credit per track, seat zone
  camera.py              multi-backend Windows camera discovery
  video_worker.py        capture + inference threads, FPS and liveness stamps
  annotate.py            operator overlay and evidence crops
  liveness.py            the watchdog rule, isolated so it can be tested alone
  roster.py              CSV/paste loading, spoken calls (full reg or shortest unique)
  speech.py              SAPI voice worker (pyttsx3 fallback), warmed at startup
  session.py             roll-call state machine + per-decision telemetry
  evidence.py            captioned crop of the credited student
  export.py              CSV / XLSX / session JSON, with a Pending guard
  main.py                entry point; runs the Phase 0 gate before any UI
  ui/console.py          teacher window: video, roster, controls, status
  ui/roll_display.py     student-facing fullscreen display
data/
  botsort_static.yaml    BoT-SORT with motion compensation off (see below)
tools/
  test_detector.py       58 scoring / temporal / gating tests, no torch needed
  test_liveness.py       20 watchdog tests, incl. replay of the predecessor's stalls
  test_session.py        64 state-machine tests, incl. a replayed 52s stall
  test_export.py         40 export + evidence tests, incl. the Pending guard
  test_calibrate.py      30 calibration replay/sweep tests on synthetic footage
  record.py              scripted ground-truth recorder
  extract.py             pose inference cached to .npz, once per model
  calibrate.py           threshold sweep, recall/precision report, --apply
  test_ui.py             13 UI construction/render tests, real windows, ~1s
  voice_check.py         audible check that every call actually plays (run.bat voice)
  check_config.py        code/config contract + value sanity checks
  benchmark.py           model x imgsz sweep with stage split and validity warnings
  smoke.py               headless end-to-end pipeline check
  preview.py             live detector preview - the Phase 1 validation tool
```

`geometry.py`, `annotate.py` and `liveness.py` are additions to the layout
sketched in PLAN.md: the scale-normalization primitives, the drawing code the
detector and future UI share, and the watchdog rule extracted so that the single
most important safety property can be unit-tested without a camera.

---

## Measured on this hardware

Everything below is measured, not estimated. The lesson of Phase 1 is that each
layer had to be measured separately, because a single combined FPS number hid
three independent problems.

```
GPU        Quadro RTX 3000, Turing sm_75, 6.0 GB, driver 580.92
runtime    torch 2.14.0+cu126 / torchvision 0.29.0+cu126, FP16 verified
python     3.14.7
camera     built-in, 1280x720 max, hard-capped at 15.0 FPS
```

### Model sweep

`yolo11{n,s,m}-pose` and `yolo26{n,s,m}-pose` at imgsz 960 and 1280, FP16,
1280x720 source, `predict` only, empty frame:

| model | imgsz | FPS | infer ms | VRAM |
|---|---|---|---|---|
| yolo11n-pose | 1280 | 56.0 | 15.0 | 41 MB |
| yolo11s-pose | 1280 | 58.5 | 14.2 | 80 MB |
| **yolo11m-pose** | **1280** | **43.7** | **20.1** | **152 MB** |
| yolo26m-pose | 1280 | 42.0 | 20.9 | 180 MB |
| yolo11l-pose | 1280 | 25.9 | 32.6 | — |

VRAM is trivial next to 6 GB, so memory never constrains the choice.
`yolo11m-pose` at imgsz 1280 is the default: the `l` variant drops to 25.9 FPS in
the worst-case aspect ratio, leaving no margin for a full classroom's detection
load. `yolo11m` and `yolo26m` are within noise of each other — swapping is a
one-line config change, and Phase 4 should decide between them on real footage
rather than on these numbers.

### Three findings that a single FPS number would have hidden

**1. BoT-SORT's motion compensation costs 23% of frame time, for nothing here.**
`botsort.yaml` ships `gmc_method: sparseOptFlow`, which runs CPU optical flow on
every frame to compensate for *camera* motion. The classroom camera does not move.

| mode | ms | FPS |
|---|---|---|
| `predict` (no tracking) | 23.0 | 43.5 |
| track, botsort, gmc=sparseOptFlow | 30.1 | 33.2 |
| **track, botsort, gmc=none** | **23.4** | **42.8** |
| track, bytetrack (no gmc) | 23.3 | 43.0 |
| track, tracktrack (8.4 default, gmc on) | 30.9 | 32.4 |

With GMC off, tracking is essentially free — 23.4 ms against 23.0 ms for plain
detection. `data/botsort_static.yaml` is `botsort.yaml` with `gmc_method: none`,
and it is the configured default. Note that ultralytics 8.4 changed the default
tracker to `tracktrack.yaml`, which also enables GMC.

**2. The camera, not the GPU, is the bottleneck.** The built-in camera returns
exactly 15.0 FPS (66.7 ms/frame) in *every* configuration tested — 1080p, 720p,
480p, MJPG and YUY2 alike. It is a device limit, not a bandwidth or format
problem.

This is where the frame-rate-independent design pays for itself: at 15 FPS a
raise still confirms in 0.80 s, identical to 60 FPS. The predecessor's
0.85 s-hold / 0.28 s-gap rule would have been close to unusable on this camera.
A 30 FPS 1080p USB webcam would roughly halve confirmation latency, and
`camera.fourcc: "MJPG"` is set because uncompressed 1080p30 exceeds USB 2.0
bandwidth and makes the driver silently drop frame rate.

**3. A cold GPU runs at roughly half speed.** The SM clock idles at 300 MHz
against a 2100 MHz maximum and only reaches ~1530 MHz after about 8 seconds of
sustained load, with power going 9 W → 32 W → 79 W. A single warm-up inference
pays for CUDA kernel selection but does nothing for clocks.

`model.warmup_sec: 3.0` runs sustained inference at startup, and the startup
sequence is deliberately ordered **camera scan → model load → warm-up → running**
so the warm-up is the last thing before real frames arrive. Scanning cameras
afterwards would let the clocks fall back to idle and waste it.

At the 15 FPS duty cycle this camera imposes, the GPU never fully ramps, so
in-pipeline inference settles around 42 ms rather than the 23 ms measured under
sustained load. Inference capacity (~24 FPS) still exceeds the camera (15 FPS),
which is the condition `run.bat smoke` actually checks.

### Benchmark caveats, enforced by the tool

`tools/benchmark.py` refuses to present clean-looking numbers it cannot stand
behind. It prints the preprocess / inference / postprocess split and warns when:

- **no people were detected** — figures then exclude keypoint decoding and
  tracker cost, so they are an upper bound. Run
  `run.bat benchmark --source classroom.jpg` for numbers that mean something.
- **a model measured faster at a larger `imgsz`** — impossible for GPU compute,
  so fixed overhead is dominating and model ranking is noise. This actually fired:
  at imgsz 960 a 1280x720 source must be downscaled (2.5 ms) while at 1280 it does
  not (1.8 ms), making the larger input *cheaper*.
- **preprocessing exceeds inference** — the pipeline is input-bound, not GPU-bound.

One caveat could not be resolved: dropping `conf` to 0.01 still produced zero
detections on an empty frame, so **the cost of decoding 40 students' keypoints
plus tracker assignment remains unmeasured.** Expect the real classroom to be
somewhat slower than the table above.

Also worth knowing: aspect ratio changes the cost. A 16:9 frame letterboxes to
1280x736 while 4:3 becomes 1280x960 — 30% more pixels and about 28% slower at the
same `imgsz`.

---

## How Phase 1 fixes the predecessor

### 1. Graded scoring instead of a boolean AND

The old rule was `above_shoulder and wrist_above_elbow and elbow_not_low and
hand_high and plausible`. Every term was a hard threshold, so any single failing
term discarded all other evidence, and a student sitting near a threshold
flickered with pixel-level keypoint noise.

Each term is now a ramp in [0, 1] combined as a weighted mean. One physical
necessity stays hard: a raised hand has its wrist above its shoulder.

Measured on synthetic poses with human proportions (`run.bat test`):

| pose | score |
|---|---|
| straight-up raise | **1.000** |
| forearm-up raise, elbow at shoulder height | **1.000** |
| slight raise, hand at face height | **0.950** |
| slight raise, wrist straight over the shoulder | **0.936** |
| low raise, elbow down, hand beside the head at chin height | **0.821** |
| hand against the cheek | 0.293 |
| phone held up in front of the face | 0.275 |
| sideways stretch | 0.208 |
| chin resting on the hand | 0.172 |
| head-scratch | 0.169 |
| arm at rest | 0.000 (hard gate) |

Enter threshold is 0.55, so the weakest genuine raise sits 0.27 above it and the
strongest distractor 0.26 below it: a 0.53 gap.

**Elbow extension is a multiplier, not a weighted term.** This was a real
correction made during the build: as a 0.14-weight term a head-scratch scored
**0.61** — a false positive above the 0.55 threshold — and 0.61 again with the
nose keypoint occluded. As a multiplier with a 0.30 floor the same pose scores
0.199 while a legitimate forearm-up raise still scores 0.730. The straightness
ramp runs 50->110 degrees rather than 70->150, because a forearm-up raise sits
near 90 degrees and must pass while a hand tucked by the head sits near 45 and
must not.

### Slight raises

The elbow-angle multiplier had a side effect: it made students raise too high. A
relaxed raise - elbow left down, forearm pointing up - also has a sharp elbow
angle (~35 degrees), so it was floored exactly like a head-scratch. Only an elbow
lifted to shoulder height escaped the floor, which in practice means the whole arm
goes up. With the elbow down, **no hand height at all** was enough.

Two changes fix that without letting the distractors in:

1. **Posture is "arm straight" OR "forearm upright".** The multiplier takes the
   better of the elbow angle and the forearm's direction
   (`raise.ramps.forearm_vertical`, degrees from vertical). A raise points the
   forearm at the ceiling; a head-scratch slants it in toward the head (56 degrees
   in the test pose), so it is still floored.
2. **How high is high enough depends on where the hand is.** A chin-rest, a hand on
   the cheek and a phone held up to look at all put the wrist *in front of or
   against the face*, at chin-to-cheek height, so a hand near the face must still
   reach nose height. A hand held *clear of the face* - beside the head, over the
   shoulder - is a raise even at chin height, and none of those distractors sit
   there. `raise.beside_head_allowance` (0.30 torso lengths) is how much lower it
   may be; `raise.ramps.face_clearance` is how far from the face counts as "clear".

With a forearm half a torso long, pointing up, as the elbow is lowered:

| elbow vs shoulder | wrist vs nose | old | new |
|---|---|---|---|
| +0.10 torso | +0.20 | counts | counts |
| level | +0.10 | counts | counts |
| 0.10 below | level | no | counts |
| 0.19 below | 0.09 below | no | counts |
| 0.28 below | 0.17 below | no | counts |
| 0.35 below | 0.25 below (about chin height) | no | counts |
| 0.42 below | 0.32 below | no | no |

A torso length is roughly 50 cm, so the lowest hand that counts has dropped by
about 17 cm at both the elbow and the wrist.

The face-height rule needs a face. The nose is preferred, then the eyes, then the
ears. **With no face keypoint at all**, the upright-forearm route closes and the
original elbow-angle rule applies alone: a phone held to the face also has an
upright forearm, and without a face there is no way to tell them apart. A student
facing the camera always has a face keypoint, so this only affects students
turned away.

Known limit: a phone held **at the ear** (a call) puts the wrist at face height
beside the head and will read as a raise. Seventeen keypoints have no hand shape to
tell them apart. Hands held still for a long time (resting a head on a fist) cannot
answer, because a raise must start after the number is called.

To accept even lower hands, raise `beside_head_allowance` toward 0.4 (the config
check refuses more), or lower both ends of `raise.ramps.vertical_extension`. Each
step brings the chin-rest and phone poses closer to the threshold, so re-run
`run.bat test` and, better, calibrate on real footage.

Scores are normalized by torso length, so they are invariant to how far away a
student sits — verified to within 0.0001 across a 4x scale change.

### 2. A time-constant filter instead of a frame-counting hold chain

The old rule needed 0.85 s of continuous detection while tolerating gaps of only
0.28 s. At 5 FPS the frame interval is 0.2 s, so one dropped detection produced a
0.4 s gap, broke the chain, and logged a genuine raise as
`Timeout: no valid hand`.

Smoothing now uses `alpha = 1 - exp(-dt / tau)`, which makes the filter's response
identical at any frame rate:

| frame rate | confirmed at |
|---|---|
| 5 FPS | 0.800 s |
| 10 FPS | 0.800 s |
| 20 FPS | 0.750 s |
| 40 FPS | 0.750 s |
| 60 FPS | 0.750 s |
| continuous-time ideal | 0.744 s |

Spread is 0.05 s — bounded by the slowest stream's frame interval. With 5 FPS
*and* one in three detections dropped, it still confirms at 0.8 s.

A Schmitt trigger (enter 0.55 / exit 0.35) removes threshold chatter: zero latch
transitions while the input jitters inside the dead band.

### 3. Identity gating

The old app counted hands frame-wide, so *anyone* raising a hand marked the called
student present.

- **Rule A** — two or more confirmed raises is an ambiguity, not a vote. The
  system refuses and asks the teacher.
- **Rule B** — one credit per tracker ID per session. A person already credited
  cannot answer for anyone else.
- **Rule C** (optional) — a student's credited seat position is remembered; an
  answer from far away is flagged.

Plus a **fresh-raise requirement**: a hand parked in the air cannot answer several
consecutive roll numbers. A `fresh_lookback_sec` of 1.5 s still credits a student
who raises the instant they hear their number, slightly before the window opens.

### 4. A liveness watchdog

The predecessor's CSVs show gaps of 52 s, 40 s, 27 s and 22 s against a 5 s
cadence: inference stalled while the state machine kept running on its own timer
and marked everyone Absent. Three of eight runs were destroyed this way.

`VideoWorker` stamps every capture and inference. `is_live()` reports whether a
result arrived within `stall_threshold_sec` (0.75 s). The session state machine is
required to consult it, and `preview.py` shows the state explicitly:
**PIPELINE STALLED — marking is suspended**.

An infrastructure failure must never become an attendance record.

---

## Phase 2 — the roll call

### Running it

```bash
run.bat app
run.bat app --roster "C:\path\to\roster.csv"
```

Order of operations: **Load CSV** (or Paste Rolls) → **Start Camera** → **Roll
Screen** onto the projector, F11 for fullscreen → **Start Attendance**.

Keyboard, so the teacher never has to find a button mid-call:
`P` present, `A` absent, `U` undo, `Space` start/pause, `F11` fullscreen display.

### The state machine

```
IDLE → CALLING → RESPONSE → RESULT → (next) → FINISHED
                     ↓
               STALLED      pipeline not live: countdown suspended
               AMBIGUOUS    two hands, or an already-credited person
               PAUSED
```

`STALLED` and `AMBIGUOUS` are the two phases the predecessor did not have, and
they are the two that stop it writing a confident wrong answer.

* **A stall never becomes an Absent.** Every automatic Absent goes through
  `video.may_record_absence()`. While stalled the deadline is pushed forward by
  the stall duration, so the student gets a full window once frames return. There
  is a test that replays the predecessor's 52-second freeze and asserts the
  student is still Pending afterwards.
* **Ambiguity is escalated, never guessed.** The machine stops and waits for the
  teacher. It does not time out into a decision.
* **Speech is advisory.** The window opens when the utterance completes *or* after
  `max_speech_wait_sec`, whichever is first, so a hung SAPI engine cannot freeze
  the roll call. With no TTS engine at all it simply proceeds.

### The roll-call voice

The screen shows `SP25-BCS-007` and the machine reads it aloud: **"S P two five,
B C S, zero zero seven"**. That is `speech.mode: "full"`, the default. Digits are
read one at a time, the way a registration number is read, and the commas make
the voice pause between parts so it is intelligible from the back row. At
`speech.rate: 175` wpm a call takes about 4.4 s (measured, SAPI, Zira).

**Why the voice was silent before.** Measured on this machine with pyttsx3 2.99,
only the *first* `runAndWait()` plays. Every later call returns in about 0.15 s
having played nothing: a sentence that takes 4 s to say "finished" in 0.24 s. The
startup announcement was that first call, so every roll number after it was
silent. The completion events still arrived on time, so nothing looked wrong. The
worker now drives `SAPI.SpVoice` through comtypes (already a dependency; pyttsx3
uses it underneath), which plays every utterance at full length. pyttsx3 remains
the fallback where SAPI does not exist (`speech.engine: auto | sapi | pyttsx3`).

`run.bat voice` speaks a few calls and times them. An utterance that finishes
faster than its words could be said is reported as **SILENT**, which is exactly how
the old bug would have shown up.

**Raising during the call counts.** The number is on the screen from the moment
the call starts, so many students raise before the voice finishes. Freshness (a
raise must be new, so a hand left up from the previous student cannot answer) is
measured from the start of the call, not from when the window opens. Before this
fix, with a 4-5 s call, a student who raised straight away was treated as a
"parked" hand and was never credited, even with the hand held up.

`session.max_speech_wait_sec` is 8 s, so the window never opens halfway through a
number. Pausing cuts the voice off mid-word, and resuming reads the interrupted
number again from the start.

#### The faster alternative: shortest unique calls

`speech.mode: "short_unique"` says only as much as the roster needs, e.g.
**"seven"**. It is about 4 s faster per student, but students have to know their
number by its last digits. The spoken form escalates only as far as it must to
stay unambiguous across the loaded roster:

| level | form | example |
|---|---|---|
| 0 | number only | `seven` |
| 1 | program + number | `B C S seven` |
| 2 | batch + program + number | `S P two five B C S seven` |
| 3 | whole roll, spelled | fallback |

A single-section roster lands on level 0. Mixing `SP25-BCS-001` with
`FA24-BCS-001` forces level 2 automatically.

Rosters are read as `utf-8-sig` (Excel writes a BOM) with every cell stripped —
the sample roster on this machine has both a BOM and a leading space on one row.
Headers and a name column are detected when present, and duplicate roll numbers
are rejected with the offenders named.

### The student display

Full-width countdown bar across the top, the roll number filling most of the
screen, the prompt below it, and a hold bar underneath.

The countdown is a bar rather than a ring around the number for a measured reason:
a ring capped the roll number at roughly a quarter of the screen width, and a
student in the back row reading the number from fifteen metres away is the entire
point of the window. The number is fitted by measuring the rendered text, so short
and long roll numbers both fill the space without clipping.

Prompts close the loop with the student:

| situation | prompt | hold bar |
|---|---|---|
| window open, nothing seen | RAISE YOUR HAND | empty |
| something detected, not enough | RAISE HIGHER | amber, tracks score |
| raise detected and holding | HAND SEEN - KEEP IT UP | green, fills to the hold time |
| credited | PRESENT | — |
| pipeline stalled | CAMERA PROBLEM - PLEASE WAIT | frozen, bar does not tick |

During a stall the countdown is frozen deliberately. Animating it would tell the
students the clock is running when it is not.

The predecessor showed hold progress on the *operator's* screen only, so a student
had no way to tell whether they had been noticed. A student who can see they are
being detected holds their hand up long enough; one who cannot drops it early and
is marked absent.

### A design bug this phase exposed

Rule A ("two raised hands is an ambiguity") barely worked as originally written,
and a test caught it. It counted only *confirmed* raises — but two people never
raise in the same video frame. In a two-hand trace, one track latched 0.05 s before
the other and therefore confirmed 0.05 s earlier, so it was credited outright and
the second hand was silently ignored. A friend answering slightly ahead of the real
student would have been accepted.

Rule A now counts **contenders** — every track whose hand is currently up,
confirmed or not — while only a confirmed track can actually be credited. This
costs no latency, because a competing hand is already latched by the time the first
one confirms.

The arbitration horizon is emergent rather than configured: a hand counts as
competing if it is up before the first one finishes confirming, which is
`hold_sec` (0.60 s) plus the EMA rise to the enter threshold (~0.15 s). Past that,
the called student has answered promptly and correctly, and a later raise does not
retroactively void it — Rule B stops that person being credited for anyone else
anyway. There are regression tests at 0.05 s, 0.20 s, 0.35 s and 0.45 s offsets,
plus one asserting the boundary behaviour.

### Another cold-start penalty

Same shape as the GPU clock finding in Phase 1: the first utterance pays about
2 s of audio-device initialization that later ones do not. Without a warm-up that
penalty lands on the first student called. The engine speaks a short startup
announcement, which doubles as an audible check that output and volume work before
a class is waiting on it.

(An earlier version of this section said later utterances took 0.10-0.18 s. That
was the pyttsx3 silent-repeat bug being measured, not speech: nothing was playing.
See [The roll-call voice](#the-roll-call-voice).)

### Pause stops the clock

Pause used to leave the deadlines running, so pausing mid-window for longer than
the time left marked the student Absent the moment the teacher resumed. The window
now resumes with exactly the time it had left, a result on screen keeps its hold
time, and a call interrupted mid-sentence is read again from the start.

---

## Phase 3 — evidence and export

**Export** writes three files per session into
`~/Documents/SmartRollCall Attendance` (the predecessor's folder, so an existing
habit keeps working), or wherever `output.dir` points:

```
attendance_<stamp>.csv     the register
attendance_<stamp>.xlsx    formatted, with a Summary sheet
session_<stamp>.json       config + runtime + one record per decision
evidence/<stamp>/<roll>.jpg
```

### CSV

The first five columns are byte-for-byte the predecessor's — `Roll Number, Name,
Status, Marked At, Evidence` — so anything already reading those files keeps
working. Three columns are added: `Track ID`, `Score`, `Evidence Image`.

Written as **utf-8-sig**, because Excel misreads a plain UTF-8 CSV. That is also
why roster loading reads utf-8-sig.

### Pending is never silently exported as Absent

One of the predecessor's eight saved runs is twenty rows of `Pending` with empty
timestamps, written by an export that did not check and told nobody. Here an
unfinished roster is counted, the console asks before continuing, the rows stay
`Pending` with no fabricated timestamp, and the workbook's Summary sheet carries
an explicit `WARNING: n student(s) were never called`.

### Evidence images

A captioned crop of the credited student at the moment of the mark: roll number
and name burned in on the left, score, track ID and time on the right.

The crop is deliberately **not** drawn over with the pose skeleton — the skeleton
is useful for tuning and is shown live in the operator view, but it obscures
exactly the thing an evidence image exists to show. Crops smaller than
`evidence_min_width` are upscaled, because a back-row student 90 px tall is
legitimate for detection but useless to look at.

Paths are recorded **relative to the output folder**, so the CSV, workbook and
images can be moved or archived together without the references breaking.
`session_*.json` records the absolute base, and so does the workbook Summary sheet.

Capture never raises. A failed disk write returns an empty path and the attendance
mark still stands — losing a photo must not lose a register entry.

> **Privacy.** These are photographs of students. Where they are stored, who can
> read them and how long they are kept are decisions for the institution, not for
> this program. `output.save_evidence_images: false` disables capture entirely, and
> the images live next to the attendance files so one retention policy can cover
> both.

### Session telemetry, and the track-reuse audit

`session_*.json` carries the full config snapshot, runtime details (model, camera,
FPS) and one record per decision:

```json
{
  "roll": "SP25-BCS-001", "status": "Present", "track_id": 100,
  "contenders": [100], "confirmed": [100],
  "seconds_to_decide": 1.7, "window_extended": false,
  "fps": 15.2, "capture_fps": 15.0
}
```

This is what makes Phase 4 threshold tuning possible after the fact — you can see
how long real students took to respond and how often the window had to be extended.

It also carries a **`track_reuse`** block. Rule B is supposed to make it impossible
for one tracked person to be credited for two roll numbers, so this should always
be empty:

```json
"track_reuse": { "3": ["SP25-BCS-001", "SP25-BCS-004"] }
```

If it is ever non-empty, tracker IDs are churning and the anti-proxy guarantee is
weaker than designed. This exists because a real session on this machine credited
two different roll numbers from near-identical positions, and nothing in the CSV
could say whether that was one person tracked twice or a legitimate reset. Now it
can.

---

## Phase 4 — calibration

This is the step that turns the thresholds from defensible-looking defaults into
measured choices. Everything before it was validated against synthetic poses, which
prove the maths is right but say nothing about whether 0.55 is the correct enter
threshold for *your* room, *your* camera and *your* students.

Three commands:

```bash
run.bat record    --name classroom-1          # capture labelled footage
run.bat extract   --name classroom-1          # run the model once, cache keypoints
run.bat calibrate --name classroom-1          # sweep thresholds, report recall/precision
run.bat calibrate --name classroom-1 --apply  # write the winner into config.json
```

### Ground truth comes for free

`record.py` runs a scripted protocol: it tells the subject what to do and when, and
writes the intervals it asked for as labels. Nobody hand-labels video afterwards.

The script deliberately contains **more distractors than genuine raises** —
scratching, stretching, waving, holding a phone to the face, leaning on an elbow.
Recall is easy to judge informally ("it saw my hand"); precision against
distractors is not, and a detector that fires on someone scratching their head will
quietly mark absent students present all term.

Per-frame timestamps are recorded as they actually happened, not as
`frame_index / nominal_fps`. This camera delivers 15 FPS with jitter, and the
temporal filter is driven by real elapsed time — calibrating against idealized
timestamps would tune for a frame rate that never existed.

### The pose cache is what makes a sweep possible

Inference costs ~40 ms per frame, so a five-minute recording is about three minutes
of GPU time. A sweep over a few thousand threshold combinations that re-ran the
model each time would take **days**.

But scoring and temporal filtering are pure Python over keypoints that do not
change when a threshold changes. So `extract.py` runs the model **once** per
(model, imgsz) pair and caches keypoints to an `.npz`; every subsequent sweep reads
that cache and finishes in seconds. Extracting with several models lets the sweep
compare them on identical footage.

The sweep itself is two-level: scores are computed once per distinct *scorer*
configuration and reused across every *temporal* combination, rather than
recomputing them for each.

### The replay mirrors production exactly

Not approximately:

* detections are grouped per frame and fed to the tracker in a **single** update —
  feeding one track at a time would make each look momentarily unseen to the
  others, which is the bug that broke Rule A in Phase 2;
* the tracker is armed at each segment boundary, exactly as it is when a roll
  number is called;
* `min_person_height_px` and the untracked-detection filter are applied as
  `VideoWorker` applies them;
* real recorded timestamps drive the time-constant filter.

If the replay did not match production, its numbers would describe a system nobody
ships.

### Reading the output

```
   recall    prec      F1   TP   FN   FP   TN   settings
   100.0%  100.0%   1.000    3    0    0   10   enter=0.55  hold=0.6  tau=0.18  straight=0.3
```

A baseline row shows how the *current* `config.json` performs on the same footage,
so you can see whether tuning actually helped.

Ranking is by F1, then recall: among equally balanced settings it prefers the one
that misses fewer real students, because a false absence is the costlier error.

The recommendation is the best setting meeting `--target-recall` (default 98%) and
`--max-fp-rate` (default 1%). When nothing qualifies it says so plainly and reports
the best available, rather than presenting a threshold that only looks good.

False positives are attributed to the action that caused them:

```
false positives by action: scratch x3, phone x1
```

That is far more actionable than a bare precision number — it tells you which
distractor the geometry is weak against.

### Caveats the tool enforces

* **Fewer than 20 segments** triggers a warning: one flipped decision moves recall
  by several percent, so record more repetitions (`--reps`) before trusting it.
* **No usable detections** stops the sweep entirely with a message pointing at
  camera placement. No threshold can recover a subject the detector never sees, and
  a sweep over zero data would happily report 0% for everything as if that were
  information.
* `--apply` backs up `config.json` to `config.json.bak` before writing.

### This has not been run on real footage yet

The machinery is tested end-to-end on synthetic recordings — clean footage scores
100% recall with zero false positives, a permissive threshold correctly admits
`scratch`, `stretch` and `phone`, and an unreachable one correctly loses recall.
But no real classroom recording exists yet, so **the shipped thresholds remain
reasoned defaults, not measured ones.** Record five minutes from the real camera
position and that changes.

---

## Key configuration

Everything lives in `config.json`.

| key | default | meaning |
|---|---|---|
| `model.weights` | `yolo11m-pose.pt` | set from `run.bat benchmark` |
| `model.imgsz` | `1280` | up from the predecessor's 768; the biggest back-row recall win after the GPU |
| `model.half` | `true` | FP16, honoured only if the runtime proves it works (sent as `quantize=16`) |
| `model.tracker` | `data/botsort_static.yaml` | BoT-SORT with `gmc_method: none` — 23% faster, loses nothing on a fixed camera |
| `model.warmup_sec` | `3.0` | sustained warm-up so the GPU clocks up before the first student |
| `model.max_consecutive_failures` | `10` | transient inference errors tolerated before giving up |
| `model.min_person_height_px` | `90` | below this, keypoints are noise |
| `camera.fourcc` | `MJPG` | required for 1080p30 over USB 2.0 |
| `raise.weights` | — | four ramped features, must sum to 1.0 |
| `raise.straightness_floor` | `0.30` | posture multiplier floor |
| `raise.ramps.forearm_vertical` | `[45, 20]` | degrees from vertical; an upright forearm passes the posture check even with the elbow down |
| `raise.beside_head_allowance` | `0.30` | how far below nose height (torso lengths) a hand held clear of the face may be; higher = slighter raises |
| `raise.ramps.face_clearance` | `[0.12, 0.25]` | wrist-to-face distance (torso lengths) that counts as clear of the face |
| `temporal.ema_tau_sec` | `0.18` | smoothing time constant, not a frame count |
| `temporal.enter/exit_threshold` | `0.55` / `0.35` | Schmitt trigger band |
| `temporal.hold_sec` | `0.60` | wall-clock hold before crediting |
| `temporal.track_grace_sec` | `0.50` | detection dropout absorbed without decay |
| `session.response_window_sec` | `6.0` | up from the predecessor's 4.0 |
| `session.max_speech_wait_sec` | `8.0` | longer than a full registration number takes to say |
| `speech.mode` | `full` | `full` reads the whole registration number; `short_unique` says e.g. "seven" |
| `speech.engine` | `auto` | `auto` = SAPI on Windows, else pyttsx3 |
| `speech.rate` | `175` | words per minute |
| `speech.voice` | `""` | part of a voice name, e.g. `Hazel` for the British voice |
| `session.stall_threshold_sec` | `0.75` | liveness limit for the watchdog |
| `gating.require_single_hand` | `true` | Rule A |
| `gating.one_credit_per_track` | `true` | Rule B |
| `gating.seat_zone_lock` | `false` | Rule C, off by default |
| `output.dir` | `""` | blank = `~/Documents/SmartRollCall Attendance` |
| `output.save_evidence_images` | `true` | set false to disable photo capture entirely |
| `output.evidence_jpeg_quality` | `88` | |
| `output.evidence_min_width` | `240` | upscale small back-row crops to stay legible |
| `output.write_xlsx` | `true` | |
| `output.write_session_json` | `true` | needed for Phase 4 tuning |

After editing, run `python tools/check_config.py`.

---

## What still needs doing

Neither of these is a code task.

**1. The physical camera test.** Mount the webcam where it will actually live and
run `run.bat preview`. Confirm every seat you care about gets a skeleton drawn on
it. If the back row has no skeleton, that is a placement problem and no threshold
fixes it. This remains the highest risk in the project.

**2. A real calibration recording.** The thresholds are reasoned defaults validated
against synthetic poses, not measured ones. Five minutes from the real camera
position turns them into measured choices:

```bash
run.bat record    --name classroom-1 --reps 3
run.bat extract   --name classroom-1
run.bat calibrate --name classroom-1
```

Also worth knowing: the built-in camera on this machine caps at 15 FPS. The
detector is frame-rate independent by design so it works, but a 30 FPS 1080p USB
webcam would roughly halve confirmation latency.

The microphone/voice feature from the predecessor is **deliberately not carried
over**. It was RMS energy thresholding, not speech recognition, and a noise hit
selected the *shorter* hold requirement — so ambient noise made it easier to be
marked Present, while the record claimed `Hand + voice`. See PLAN.md.
