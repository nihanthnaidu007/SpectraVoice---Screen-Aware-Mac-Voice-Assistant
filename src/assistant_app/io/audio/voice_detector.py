"""Smart Voice Detector - Speech validation with noise filtering and robust parsing."""

import re
from difflib import SequenceMatcher


class SmartVoiceDetector:
    """
    Smart voice detection with noise filtering and Whisper hallucination rejection.
    
    Filters out:
    - Common noise phrases (um, uh, etc.)
    - Whisper hallucinations (common false transcriptions from silence/noise)
    - Assistant self-echo (hearing its own speech)
    - Duplicate/repeated queries
    """
    
    MIN_WORDS = 2  # Require at least 2 words to filter noise (was 1)
    MIN_CHARS = 6  # Minimum characters to filter very short noise (was 3)
    MIN_CONFIDENCE = 0.45  # Minimum confidence to accept as valid speech
    
    NOISE_PHRASES = frozenset({
        "", " ", "you", "the", "a", "an", "um", "uh", "hmm", "hm",
        "ah", "oh", "eh", "er", "like", "so", "and", "but", "or",
        "thanks", "thank you", "okay", "ok", "bye", "goodbye",
        "yes", "no", "yeah", "nah", "yep", "nope", "sure", "right",
        "i see", "i know", "got it", "alright", "all right",
        # Additional noise phrases
        "mm", "mhm", "uh huh", "huh", "what", "sorry", "pardon",
    })
    
    # Common Whisper hallucinations from silence/noise
    # These are transcriptions Whisper commonly produces when there's no real speech
    WHISPER_HALLUCINATIONS = frozenset({
        # Music/audio descriptions
        "music", "music playing", "playing music",
        "[music]", "[music playing]", "(music)", "(music playing)",
        "♪", "♪♪", "♪ ♪", "la la la",
        # Silence indicators
        "silence", "[silence]", "(silence)", "...", 
        "inaudible", "[inaudible]", "(inaudible)",
        # Common false positives
        "thank you for watching", "thanks for watching",
        "subscribe", "like and subscribe", "please subscribe",
        "see you next time", "see you later",
        "bye bye", "bye-bye",
        "you", "i", "the", "it", "is", "be",
        # Foreign language fragments Whisper sometimes hallucinates
        "vous", "merci", "bonjour", "gracias", "hola",
        # Repeated characters/sounds
        "ah ah ah", "oh oh oh", "ha ha ha", "he he he",
        # Common ambient noise interpretations
        "breathing", "coughing", "sneezing", "sighing",
        "[breathing]", "[coughing]", "(breathing)",
        # Podcast/video artifacts
        "intro", "outro", "[intro]", "[outro]",
        "chapter", "next chapter",
    })
    
    # Assistant's own phrases to filter out (self-echo)
    ASSISTANT_PHRASES = frozenset({
        "hi i'm ready to help",
        "hello what can i do for you",
        "hey ready when you are",
        "hi there how can i assist you",
        "hi there how can i assist you?",
        "what can i do for you",
        "how can i assist you",
        "how can i help you",
        "ready to help",
        "hello! what can i do for you?",
    })
    
    # Common mishearing corrections (pattern -> intended)
    MISHEARING_CORRECTIONS = {
        # Common speech recognition errors
        r"come[ae]t\s*[ckqv]+o[mn]+t?": "comment",
        r"cometcvomt": "comment",
        r"opan\b": "open",
        r"opin\b": "open",
        r"chroam": "chrome",
        r"crome": "chrome",
        r"safary": "safari",
        r"safaree": "safari",
        r"tiep\b": "type",
        r"tipe\b": "type",
        r"scrol+": "scroll",
        r"clik\b": "click",
        r"clicke?\b": "click",
        r"serch\b": "search",
        r"seartch": "search",
        r"opn\b": "open",
        r"cloes?": "close",
        r"cloze": "close",
        r"coppy": "copy",
        r"pase?te?": "paste",
        r"deleet": "delete",
        r"delet\b": "delete",
        r"savve?": "save",
        r"youtu+be?": "youtube",
        r"goo+gle?": "google",
        r"termin[ao]l": "terminal",
        r"vscode": "vs code",
        r"notpad": "notepad",
        r"noets": "notes",
        r"mesages?": "messages",
        r"setings?": "settings",
        r"preferances?": "preferences",
    }
    
    QUESTION_WORDS = frozenset({
        "what", "who", "where", "when", "why", "how", "which",
        "can", "could", "would", "should", "is", "are", "do", "does",
        "tell", "show", "explain", "help", "find", "search", "look"
    })
    
    COMMAND_WORDS = frozenset({
        "open", "close", "start", "stop", "play", "pause", "read",
        "write", "save", "delete", "create", "make", "set", "change",
        "turn", "switch", "go", "navigate", "click", "type", "scroll"
    })
    
    def __init__(self, duplicate_threshold: float = 0.85, echo_cooldown: float = 3.0):
        self.last_prompts: list[str] = []
        self.max_history = 5
        self.duplicate_threshold = duplicate_threshold
        self.echo_cooldown = echo_cooldown  # Increased from 2.0 to 3.0 seconds
        self.last_tts_end_time: float = 0.0
        self._compiled_corrections = {
            re.compile(pattern, re.IGNORECASE): replacement
            for pattern, replacement in self.MISHEARING_CORRECTIONS.items()
        }
        
        # === DYNAMIC TTS RESPONSE TRACKING ===
        # Track what the assistant actually said to filter self-echo
        self._last_tts_responses: list[str] = []  # Recent TTS responses
        self._max_tts_history = 5  # Keep last 5 responses
        self._tts_active = False  # True while TTS is playing
    
    def is_valid(self, text: str) -> tuple[bool, str, float]:
        """
        Validate if text represents real user speech.
        
        Returns:
            tuple: (is_valid, reason, confidence)
            - is_valid: True if this appears to be real user speech
            - reason: Why it was accepted/rejected
            - confidence: 0.0-1.0 confidence score
        """
        import time
        
        if not text:
            return False, "empty", 0.0
        
        cleaned = text.strip().lower()
        
        # === WHISPER HALLUCINATION CHECK ===
        # Filter out common Whisper false transcriptions from noise/silence
        if cleaned in self.WHISPER_HALLUCINATIONS:
            return False, "whisper_hallucination", 0.0
        
        # Check for hallucination patterns (text in brackets, music symbols, etc.)
        if self._is_hallucination_pattern(cleaned):
            return False, "hallucination_pattern", 0.0
        
        # === DYNAMIC TTS ECHO CHECK ===
        # Check if this is the assistant's own TTS output being picked up by mic
        # This is the PRIMARY defense against self-hearing
        if self.is_tts_echo(cleaned):
            return False, "tts_echo", 0.0
        
        # Check for self-echo (assistant hearing its own speech)
        if self._is_self_echo(cleaned):
            return False, "self_echo", 0.0
        
        # Check echo cooldown (ignore input right after TTS)
        if self.last_tts_end_time > 0:
            time_since_tts = time.time() - self.last_tts_end_time
            if time_since_tts < self.echo_cooldown:
                # Check if it sounds like the assistant's speech
                if self._sounds_like_assistant(cleaned):
                    return False, "echo_cooldown", 0.0
        
        if len(cleaned) < self.MIN_CHARS:
            return False, "too_short", 0.1
        
        if cleaned in self.NOISE_PHRASES:
            return False, "noise_phrase", 0.1
        
        words = cleaned.split()
        if len(words) < self.MIN_WORDS:
            # Exception: Allow single command words (e.g., "stop", "open")
            if len(words) == 1 and words[0] in self.COMMAND_WORDS:
                pass  # Allow single command words through
            else:
                return False, "too_few_words", 0.2
        
        if self._is_duplicate(cleaned):
            return False, "duplicate", 0.3
        
        confidence = self._calculate_confidence(cleaned, words)
        
        # Use class-level minimum confidence threshold
        if confidence < self.MIN_CONFIDENCE:
            return False, "low_confidence", confidence
        
        self._add_to_history(cleaned)
        return True, "valid", confidence
    
    def _is_hallucination_pattern(self, text: str) -> bool:
        """Check if text matches common Whisper hallucination patterns."""
        # Brackets indicate non-speech audio descriptions
        if text.startswith('[') or text.startswith('('):
            return True
        # Music symbols
        if '♪' in text or '♫' in text:
            return True
        # Repeated single characters (noise artifacts)
        if len(set(text.replace(' ', ''))) <= 2 and len(text) > 3:
            return True
        # Very repetitive text (same word repeated)
        words = text.split()
        if len(words) >= 3 and len(set(words)) == 1:
            return True
        return False
    
    def _is_self_echo(self, text: str) -> bool:
        """Check if text is the assistant's own speech (self-echo)."""
        # Exact match
        if text in self.ASSISTANT_PHRASES:
            return True
        
        # Fuzzy match for assistant phrases
        for phrase in self.ASSISTANT_PHRASES:
            if self._similarity(text, phrase) > 0.75:
                return True
            # Check if text contains the assistant phrase
            if phrase in text or text in phrase:
                return True
        
        return False
    
    def _sounds_like_assistant(self, text: str) -> bool:
        """Check if text sounds like something the assistant would say."""
        assistant_patterns = [
            "can i help", "can i assist", "do for you",
            "ready to help", "assist you", "help you"
        ]
        for pattern in assistant_patterns:
            if pattern in text:
                return True
        return False
    
    def mark_tts_complete(self) -> None:
        """Mark when TTS finishes speaking (call this after speech completes)."""
        import time
        self.last_tts_end_time = time.time()
        self._tts_active = False
    
    def mark_tts_start(self, response_text: str) -> None:
        """
        Mark when TTS starts speaking and record what it's saying.
        This allows us to filter out the assistant's own speech if picked up by mic.
        
        Args:
            response_text: The text being spoken by TTS
        """
        self._tts_active = True
        # Store the response for echo detection
        cleaned = response_text.strip().lower()
        self._last_tts_responses.append(cleaned)
        # Keep only recent responses
        if len(self._last_tts_responses) > self._max_tts_history:
            self._last_tts_responses.pop(0)
    
    def is_tts_echo(self, text: str) -> bool:
        """
        Check if text is likely the assistant's own TTS output being picked up by mic.
        
        This uses multiple strategies to detect if the transcribed text matches
        any part of what the assistant recently said.
        
        Args:
            text: Transcribed text to check
            
        Returns:
            True if this appears to be TTS echo
        """
        if not text:
            return False
        
        cleaned = text.strip().lower()
        text_words = cleaned.split()
        text_word_set = set(text_words)
        
        # Common words to ignore when checking overlap (too generic)
        common_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", 
                       "have", "has", "had", "do", "does", "did", "will", "would",
                       "could", "should", "of", "to", "in", "on", "at", "for", "with",
                       "it", "its", "this", "that", "not", "no", "yes"}
        
        # Get meaningful words (exclude common words)
        meaningful_words = text_word_set - common_words
        
        # === STRATEGY 1: If TTS is actively playing, be strict but smart ===
        if self._tts_active:
            first_word = text_words[0] if text_words else ""
            
            # These words indicate a REAL user interruption - always allow
            interrupt_words = {"stop", "wait", "hold", "pause", "quiet", "shut", "hey", "no", "cancel", "enough"}
            if first_word in interrupt_words:
                return False  # Real interruption, not echo
            
            # Question starters indicate a NEW question - check more carefully
            question_starters = {"what", "who", "where", "when", "why", "how", "can", "could", "tell", "show", "open", "close"}
            
            if first_word in question_starters:
                # It's a question - only block if it's clearly from TTS
                # Questions naturally share some words with answers, so be lenient
                for tts_response in self._last_tts_responses:
                    # Only check for consecutive word sequences (strong echo indicator)
                    # Single word overlap is NOT enough to block a question
                    if self._has_consecutive_match(cleaned, tts_response, min_words=3):
                        return True
                return False  # Looks like a real new question
            
            # Not a question or interrupt - likely echo if any meaningful overlap
            for tts_response in self._last_tts_responses:
                tts_words = set(tts_response.split()) - common_words
                if meaningful_words and tts_words:
                    overlap = len(meaningful_words & tts_words)
                    if overlap >= 2:
                        return True
                # Check for exact phrase match
                if cleaned in tts_response or tts_response in cleaned:
                    return True
            
            # Doesn't start with question/interrupt word and TTS is active = likely echo
            return True
        
        # === STRATEGY 2: TTS not active - check against stored responses ===
        if not self._last_tts_responses:
            return False
        
        for tts_response in self._last_tts_responses:
            # Exact substring match (text is part of TTS response)
            if len(cleaned) > 10 and cleaned in tts_response:
                return True
            
            # Check for consecutive word sequences (strong indicator of echo)
            if self._has_consecutive_match(cleaned, tts_response, min_words=4):
                return True
        
        return False
    
    def _has_consecutive_match(self, text: str, tts_response: str, min_words: int = 3) -> bool:
        """Check if text contains a consecutive sequence of words from TTS response."""
        text_words = text.split()
        
        if len(text_words) < min_words:
            return False
        
        # Common sequences that appear in many sentences - ignore these
        common_sequences = {
            "who is the", "what is the", "where is the", "when is the",
            "how is the", "why is the", "can you", "could you",
            "tell me", "show me", "is the", "are the", "was the",
            "of the", "in the", "on the", "to the", "for the",
            "it is", "this is", "that is", "there is",
        }
        
        # Check all consecutive sequences of min_words length
        for i in range(len(text_words) - min_words + 1):
            sequence = ' '.join(text_words[i:i + min_words])
            # Skip common sequences
            if sequence in common_sequences:
                continue
            if sequence in tts_response:
                return True
        
        return False
    
    @property
    def is_tts_active(self) -> bool:
        """Check if TTS is currently playing."""
        return self._tts_active
    
    def _is_duplicate(self, text: str) -> bool:
        for prev in self.last_prompts:
            if self._similarity(text, prev) > self.duplicate_threshold:
                return True
        return False
    
    def _similarity(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        words_a, words_b = set(a.split()), set(b.split())
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / len(words_a | words_b)
    
    def _calculate_confidence(self, text: str, words: list[str]) -> float:
        score = 0.5
        first_word = words[0] if words else ""
        
        if first_word in self.QUESTION_WORDS:
            score += 0.2
        if first_word in self.COMMAND_WORDS:
            score += 0.2
        if "?" in text:
            score += 0.15
        if len(words) >= 4:
            score += 0.1
        if len(words) >= 7:
            score += 0.1
        
        # Penalties
        unique_ratio = len(set(words)) / len(words) if words else 0
        if unique_ratio < 0.5:
            score -= 0.2
        if text.isupper() and len(text) > 3:
            score -= 0.1
        
        return min(1.0, max(0.0, score))
    
    def _add_to_history(self, text: str) -> None:
        self.last_prompts.append(text)
        if len(self.last_prompts) > self.max_history:
            self.last_prompts.pop(0)
    
    def clear_history(self) -> None:
        self.last_prompts.clear()
    
    def correct_mishearings(self, text: str) -> str:
        """Apply corrections for common speech recognition mishearings."""
        corrected = text
        for pattern, replacement in self._compiled_corrections.items():
            corrected = pattern.sub(replacement, corrected)
        return corrected
    
    def fuzzy_match_command(self, word: str, threshold: float = 0.7) -> str | None:
        """
        Fuzzy match a potentially mis-heard word to known commands.
        Returns the matched command or None if no good match.
        """
        all_commands = self.COMMAND_WORDS | self.QUESTION_WORDS
        best_match = None
        best_score = 0.0
        
        for cmd in all_commands:
            score = SequenceMatcher(None, word.lower(), cmd).ratio()
            if score > best_score and score >= threshold:
                best_score = score
                best_match = cmd
        
        return best_match
    
    def normalize_input(self, text: str) -> str:
        """
        Normalize user input by correcting mishearings and cleaning up.
        Returns the normalized text.
        """
        if not text:
            return text
        
        # First apply known corrections
        normalized = self.correct_mishearings(text)
        
        # Then try fuzzy matching on first word if it's not recognized
        words = normalized.split()
        if words:
            first_word = words[0].lower()
            if first_word not in self.COMMAND_WORDS and first_word not in self.QUESTION_WORDS:
                matched = self.fuzzy_match_command(first_word)
                if matched:
                    words[0] = matched
                    normalized = ' '.join(words)
        
        return normalized
