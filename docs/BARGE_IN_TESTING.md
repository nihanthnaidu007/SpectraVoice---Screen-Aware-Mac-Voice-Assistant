# Barge-In Testing Guide

This document describes how to test the barge-in (voice interruption) feature.

## Overview

Barge-in allows users to interrupt the assistant while it's speaking. When the user starts talking:
1. The assistant immediately stops speaking
2. The new speech is recognized as a new query
3. The assistant responds to the new query

## Quick Test

```bash
# Run with debug mode to see barge-in logs
python main.py --debug
```

---

## Manual Test Cases

### Test 1: Basic Barge-In

**Steps:**
1. Ask a question that produces a long answer (e.g., "Explain quantum computing in detail")
2. While the assistant is speaking (mid-sentence):
   - Start speaking a new query (e.g., "What time is it?")

**Expected Results:**
- ✅ Assistant stops speaking immediately (within ~100ms)
- ✅ You see log: `🗣️ Barge-in #1 detected! Stopping TTS...`
- ✅ You see log: `🛑 Stopping TTS (barge-in #1)`
- ✅ New query is transcribed and processed
- ✅ Assistant responds to the new query
- ✅ Old response is abandoned (not resumed)

**Console Output Example:**
```
🔊 Starting TTS for response...
🔊 TTS playback started
🗣️ Barge-in #1 detected! Stopping TTS to handle new query...
📝 New query: 'What time is it?' (conf=75%, words=4)
🛑 Stopping TTS (barge-in #1)
⏹️ Speech interrupted (barge-in) - handling new query
🎤 User (75%): What time is it?
👁️ Analyzing screen...
🤖 Assistant: The current time is...
```

---

### Test 2: Noise Rejection During TTS

**Steps:**
1. Ask a question that produces a long answer
2. While the assistant is speaking:
   - Make brief noise (cough, "um", single word)

**Expected Results:**
- ✅ Assistant continues speaking (NOT interrupted)
- ✅ You see debug log: `🔇 Speech during TTS ignored (conf=XX%, words=1)`
- ✅ Only multi-word (≥2), high-confidence (≥50%) speech triggers barge-in

**Why This Works:**
- `BARGE_IN_MIN_CONFIDENCE = 0.5` (50%)
- `BARGE_IN_MIN_WORDS = 2`
- Random noise doesn't meet these thresholds

---

### Test 3: Repeated Interruptions

**Steps:**
1. Ask a question
2. Interrupt with a new question
3. Interrupt again with another question
4. Repeat 3-5 times

**Expected Results:**
- ✅ Each interruption is handled correctly
- ✅ Barge-in count increments (visible in logs)
- ✅ No crashes or deadlocks
- ✅ Each new query gets a response

---

### Test 4: Echo Rejection

**Steps:**
1. Let the assistant speak a greeting
2. Wait silently for the assistant to finish
3. Listen for any "self-echo" rejections in the logs

**Expected Results:**
- ✅ Assistant doesn't interrupt itself
- ✅ You may see: `🔇 Ignored (self_echo): '...'`
- ✅ Echo cooldown prevents false triggers after TTS

---

### Test 5: Rapid-Fire Commands

**Steps:**
1. Speak a command
2. Immediately speak another command before the first response finishes

**Expected Results:**
- ✅ Second command interrupts first response
- ✅ No audio glitches or overlapping speech
- ✅ Clean transition between responses

---

## Edge Cases

### Edge Case 1: Very Short Interruption

**Scenario:** User says just "stop" or "wait"

**Expected:** 
- If "stop" alone (1 word): May NOT trigger barge-in (below BARGE_IN_MIN_WORDS)
- If "stop that" (2 words): WILL trigger barge-in

**Adjustment:** If you want single-word barge-in, set `BARGE_IN_MIN_WORDS = 1` in `spectravoice_assistant.py`

### Edge Case 2: Low Confidence Speech

**Scenario:** User mumbles or speaks quietly

**Expected:**
- If confidence < 50%: Does NOT trigger barge-in
- You see debug log with confidence value

### Edge Case 3: TTS Still Synthesizing

**Scenario:** User speaks while TTS is still calling OpenAI API (not yet playing)

**Expected:**
- ✅ Barge-in still works (state = SYNTHESIZING)
- ✅ TTS is cancelled before playback starts

---

## Logging Reference

| Log Message | Meaning |
|-------------|---------|
| `🔊 Starting TTS for response...` | TTS synthesis started |
| `🔊 TTS playback started` | Audio playback began |
| `🗣️ Barge-in #N detected!` | User interrupted assistant |
| `🛑 Stopping TTS (barge-in #N)` | TTS stop() called |
| `⏹️ Speech interrupted (barge-in)` | Playback loop exited |
| `✅ TTS playback complete` | TTS finished normally |
| `🔇 Speech during TTS ignored` | Speech below barge-in threshold |
| `🔇 Ignored (self_echo)` | Filtered assistant's own speech |

---

## Configuration

Barge-in thresholds can be adjusted in `spectravoice_assistant.py`:

```python
class SpectraVoiceAssistant:
    # Minimum confidence to trigger barge-in (0.0 - 1.0)
    BARGE_IN_MIN_CONFIDENCE = 0.5
    
    # Minimum words to trigger barge-in
    BARGE_IN_MIN_WORDS = 2
```

TTS chunk size affects interrupt latency in `tts.py`:

```python
class TextToSpeech:
    # Smaller = faster interrupt, but more CPU
    CHUNK_SIZE = 2048  # ~85ms at 24kHz
```

---

## Troubleshooting

### Issue: Barge-in not triggering

**Checklist:**
1. Is `enable_barge_in=True`? (default)
2. Is speech confidence ≥ 50%?
3. Is word count ≥ 2?
4. Check debug logs for rejection reason

### Issue: Random noise triggers barge-in

**Solutions:**
1. Increase `BARGE_IN_MIN_CONFIDENCE` (e.g., 0.6)
2. Increase `BARGE_IN_MIN_WORDS` (e.g., 3)

### Issue: Barge-in feels slow

**Solutions:**
1. Decrease `CHUNK_SIZE` in tts.py (e.g., 1024)
2. Note: Smaller chunks = more CPU usage

---

## Thread Safety Notes

The barge-in implementation is thread-safe:

- **TTS State**: Protected by `_state_lock`
- **stop() method**: Can be called from any thread
- **Audio callback**: Runs in background thread
- **TTS playback**: Runs in separate daemon thread

No deadlocks possible due to lock ordering:
1. `_state_lock` (state changes)
2. `_playback_lock` (player control)
3. `_lock` (speak operation serialization)
