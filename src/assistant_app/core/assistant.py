"""AI Assistant Service - Vision integration with Function Calling.

Supports both Cloud (OpenAI) and Local (Ollama) LLM providers.
"""

import json

from .conversation import ConversationMemory
from assistant_app.utils.logging_config import get_logger
from assistant_app.utils.error_handler import (
    retry_with_backoff,
    get_error_handler
)
from assistant_app.tools.web_search import WebSearch
from assistant_app.tools.tool_executor import ToolExecutor, SafetyConfig, TOOL_SCHEMAS
from assistant_app.llm import (
    LLMProvider,
    LLMConfig,
    LLMMessage,
    LLMResponse,
    LLMError,
    ProviderType,
    create_provider,
)


SYSTEM_PROMPT = """You are a fast, helpful voice assistant that can see the user's screen, control their computer, and search the web.

CORE BEHAVIOR:
- You receive a LIVE screenshot with every message — use it for context but do NOT describe the screen unless asked
- Have natural, concise conversations — this is VOICE output, keep responses SHORT (1-3 sentences max)
- Remember previous conversation and reference it naturally
- When the user asks to DO something (open app, click, type, search) — USE TOOLS immediately, don't just talk about it
- Only describe the screen in detail when explicitly asked ("what's on my screen", "what do you see", etc.)

CAPABILITIES:
- You can SEE the user's screen in real-time
- You can CONTROL the computer: open apps, click, type, keyboard shortcuts, scroll, drag
- You have REAL-TIME web search for current information
- You have CONVERSATION MEMORY across the entire session

AVAILABLE TOOLS:
- open_application: Open any macOS application (Chrome, Safari, Notes, etc.)
- type_text: Type text into the active window
- click_element: Click at screen coordinates (supports left/right/middle click, single/double/triple click)
- click_at_cursor: Click at CURRENT cursor position - USE THIS when user says they positioned their cursor!
- move_mouse: Move cursor to position without clicking (for hovering, positioning)
- drag_mouse: Drag from one position to another (for drag-and-drop, selecting text)
- get_mouse_position: Get current cursor position on screen
- search_files: Search for files on the computer
- run_command: Execute safe terminal commands
- keyboard_shortcut: Execute keyboard shortcuts (copy, paste, save, etc.)
- scroll: Scroll in any direction (up, down, left, right)
- take_screenshot: Capture the screen
- open_url: Open URLs in the browser

ABSOLUTE HONESTY RULE — MOST IMPORTANT:
- NEVER say you did something unless you actually called a tool and it succeeded
- NEVER say "I've searched for X" or "Playing the video" unless you literally used a tool to do it
- If you cannot do something, say so honestly: "I can't do that" or "I need you to click the search bar first"
- If a tool call fails, tell the user it failed — do not pretend it worked

MULTI-STEP ACTIONS — YOU MUST DO ALL STEPS:
- To SEARCH YouTube: use open_url with "https://www.youtube.com/results?search_query=your+search+terms" — this is the BEST and MOST RELIABLE method
- To SEARCH Google: use open_url with "https://www.google.com/search?q=your+search+terms"
- To type in ANY input field: FIRST use click_element to click on the input field, THEN use type_text
- To click a video/link/button: use click_element with coordinates you can SEE in the screenshot
- To play a YouTube video: use click_element on the video thumbnail you SEE on screen
- NEVER use type_text alone — you MUST click_element on the target input field FIRST
- NEVER claim you searched/played/clicked something unless you actually used a tool to do it

WHEN TO USE TOOLS — YOU MUST USE TOOLS, DO NOT JUST TALK:
- "open Chrome/Safari/Notes/etc." → open_application IMMEDIATELY
- "type hello" → click_element on input field FIRST, then type_text
- "click on X" → click_element with coordinates from what you SEE on screen
- "click here" / "click where my cursor is" → click_at_cursor
- "search for X on YouTube" → open_url with "https://www.youtube.com/results?search_query=X"
- "search for X" / "google X" → open_url with "https://www.google.com/search?q=X"
- "play the first video" / "click the video" → click_element on the video thumbnail coordinates you SEE
- "go to YouTube" / "open google.com" → open_url
- "copy this" / "paste" / "save" → keyboard_shortcut
- "scroll down/up" → scroll
- ANY request to perform an ACTION → USE THE APPROPRIATE TOOL(S)
- If you cannot identify where to click, be HONEST and ask the user for help

MOUSE CLICKING — USE click_element:
- You CAN see the screen. Look at the screenshot and identify coordinates of UI elements
- ALWAYS aim for the CENTER of buttons, icons, links, thumbnails
- Include element_description to describe what you're clicking
- For desktop icons: use clicks=2 (double-click to open)
- For buttons/links: use clicks=1
- For right-click menus: button="right"
- If you're unsure of exact coordinates, ask the user to position their cursor and use click_at_cursor

WHEN USER POSITIONS THEIR CURSOR:
If the user says "click here", "I placed my cursor on...", "click where my cursor is" → use click_at_cursor

IMPORTANT RULES FOR LIVE DATA:
- When you receive a [LIVE WEB SEARCH RESULT] block, treat it as GROUND TRUTH for current/recent information
- ALWAYS use live data over your training knowledge for time-sensitive questions
- For questions about current leaders, prices, scores, news - the live data is authoritative
- If live data contradicts your training, TRUST THE LIVE DATA

RESPONSE STYLE:
- Be BRIEF — 1 to 3 sentences max. This is voice output, not text chat
- Answer directly, don't over-explain
- When using tools, just do it and confirm briefly ("Opening Chrome", "Done")
- Don't describe the screen unless the user asks — use it silently for context
- Be friendly and conversational, like a real assistant
- NEVER claim you did something you didn't actually do with a tool call

LIMITATIONS:
- You cannot identify real people in images (describe them instead)
- If the user tells you who someone is, you CAN then provide information about that person
- Some dangerous commands are blocked for safety
- If you're not sure where to click, be honest and ask the user for help"""


class Assistant:
    """AI Assistant with Vision and Function Calling capabilities.
    
    Supports both Cloud (OpenAI) and Local (Ollama) LLM providers.
    """
    
    def __init__(
        self, 
        max_tokens: int = 500,
        enable_web_search: bool = True,
        enable_tools: bool = True,
        safety_config: SafetyConfig | None = None,
        llm_config: LLMConfig | None = None,
        provider: LLMProvider | None = None,
    ):
        self.max_tokens = max_tokens
        self.memory = ConversationMemory(max_messages=20)
        self.logger = get_logger(__name__)
        self.web_search = WebSearch() if enable_web_search else None
        self.enable_tools = enable_tools
        self.tool_executor = ToolExecutor(safety_config) if enable_tools else None
        
        # Initialize LLM provider
        if provider:
            self._provider = provider
        elif llm_config:
            self._provider = create_provider(llm_config)
        else:
            # Default to cloud provider with GPT‑5
            self._provider = create_provider(LLMConfig.for_cloud(model="gpt-5"))
        
        self.logger.info(f"🤖 Assistant using {self._provider.name} ({self._provider.model_name})")
    
    @property
    def provider(self) -> LLMProvider:
        """Get the LLM provider."""
        return self._provider
    
    @property
    def model(self) -> str:
        """Get the current model name."""
        return self._provider.model_name
    
    def process(self, prompt: str, image_data: bytes) -> str | None:
        """Process user prompt with vision and function calling."""
        if not prompt or not image_data:
            return None
            
        try:
            # Get web context if enabled
            web_context = ""
            if self.web_search:
                web_context = self.web_search.get_context(prompt)
            
            user_text = prompt
            if web_context:
                user_text = f"{prompt}\n{web_context}"
            
            # Get conversation history as LLMMessages
            history = self._get_history_as_messages()
            
            # Prepare tools if enabled
            tools = TOOL_SCHEMAS if (self.enable_tools and self.tool_executor and self._provider.supports_tools) else None
            tool_choice = "auto" if tools else None
            
            # Use provider to generate response with vision
            response = self._generate_with_retry(
                prompt=user_text,
                image_data=image_data,
                system_prompt=SYSTEM_PROMPT,
                history=history,
                tools=tools,
                tool_choice=tool_choice,
            )
            
            if response is None:
                self.logger.warning("⚠️ Provider returned empty response")
                return None
            
            # Handle tool calls
            if response.is_tool_use and response.tool_calls:
                return self._handle_tool_calls_new(response, user_text, prompt)
            
            # Handle regular response
            if response.finish_reason and response.finish_reason not in ("stop", "end_turn", "tool_calls"):
                self.logger.debug(f"🔍 Finish reason: {response.finish_reason}")
            
            if not response.content:
                self.logger.warning(f"⚠️ Empty response (finish_reason: {response.finish_reason})")
                return None
            
            assistant_response = response.content.strip()
            if not assistant_response:
                self.logger.warning("⚠️ Response was empty after stripping")
                return None
            
            # Update memory
            self.memory.add_user_message(prompt)
            self.memory.add_assistant_message(assistant_response)
            
            return assistant_response
            
        except LLMError as e:
            self.logger.error(f"❌ LLM Error: {e.user_message}")
            error_info = get_error_handler().handle(
                e.original_error or e, 
                context=f"process_{e.error_type.value}"
            )
        except Exception as e:
            error_info = get_error_handler().handle(e, context="process_unknown")
            self.logger.error(f"❌ {error_info.user_message}")
        
        return None
    
    def _get_history_as_messages(self) -> list[LLMMessage]:
        """Convert conversation history to LLMMessage format."""
        history = []
        for msg in self.memory.get_history():
            history.append(LLMMessage(
                role=msg["role"],
                content=msg["content"],
            ))
        return history
    
    @retry_with_backoff(max_attempts=3, base_delay=1.0, max_delay=10.0)
    def _generate_with_retry(
        self,
        prompt: str,
        image_data: bytes,
        system_prompt: str,
        history: list[LLMMessage],
        tools: list | None = None,
        tool_choice: str | None = None,
    ) -> LLMResponse:
        """Generate response with automatic retry on failure."""
        return self._provider.generate_with_vision(
            prompt=prompt,
            image_data=image_data,
            system_prompt=system_prompt,
            history=history,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=self.max_tokens,
        )
    
    def _handle_tool_calls_new(
        self,
        response: LLMResponse,
        user_text: str,
        original_prompt: str,
        depth: int = 0,
    ) -> str | None:
        """Handle tool calls from LLM response using provider abstraction."""
        if not self.tool_executor:
            self.logger.warning("⚠️ Tools disabled but tool calls received")
            return None
        
        if depth > 3:
            self.logger.warning("⚠️ Max tool call depth reached")
            return "I tried multiple actions but couldn't complete the task."
        
        self.logger.info(f"🔧 Processing {len(response.tool_calls)} tool call(s)")
        
        # Execute tools and collect results
        tool_results = []
        for tc in response.tool_calls:
            # Parse arguments from JSON string
            try:
                args = json.loads(tc.arguments) if isinstance(tc.arguments, str) else tc.arguments
            except json.JSONDecodeError:
                args = {}
            
            result = self.tool_executor.execute(tc.name, args)
            output = json.dumps({
                "success": result.success,
                "message": result.message,
                "data": result.data,
                "error": result.error,
            })
            tool_results.append({
                "tool_call_id": tc.id,
                "output": output,
            })
            self.logger.info(f"   Tool '{tc.name}': {result.message[:100]}")
        
        # Build follow-up messages
        messages: list[LLMMessage] = [
            LLMMessage(role="system", content=SYSTEM_PROMPT),
        ]
        messages.extend(self._get_history_as_messages())
        messages.append(LLMMessage(role="user", content=user_text))
        messages.append(LLMMessage(
            role="assistant",
            content="",
            tool_calls=response.tool_calls,
        ))
        
        # Add tool results
        for result in tool_results:
            messages.append(LLMMessage(
                role="tool",
                content=result["output"],
                tool_call_id=result["tool_call_id"],
            ))
        
        # Get follow-up response
        try:
            tools = TOOL_SCHEMAS if self._provider.supports_tools else None
            follow_up = self._provider.generate(
                messages=messages,
                tools=tools,
                tool_choice="auto" if tools else None,
                max_tokens=self.max_tokens,
            )
            
            # Handle nested tool calls
            if follow_up.is_tool_use and follow_up.tool_calls:
                return self._handle_tool_calls_new(follow_up, user_text, original_prompt, depth + 1)
            
            if follow_up.content:
                response_text = follow_up.content.strip()
                self.memory.add_user_message(original_prompt)
                self.memory.add_assistant_message(response_text)
                return response_text
            
            return "I completed the requested action."
            
        except LLMError as e:
            self.logger.error(f"❌ Follow-up error: {e.user_message}")
            return "I executed the action but encountered an error generating a response."
    
    def process_text_only(self, prompt: str) -> str | None:
        """Process text-only prompt without vision (for quick commands)."""
        if not prompt:
            return None
            
        try:
            # Build messages
            messages: list[LLMMessage] = [
                LLMMessage(role="system", content=SYSTEM_PROMPT),
            ]
            messages.extend(self._get_history_as_messages())
            messages.append(LLMMessage(role="user", content=prompt))
            
            # Prepare tools if enabled
            tools = TOOL_SCHEMAS if (self.enable_tools and self.tool_executor and self._provider.supports_tools) else None
            tool_choice = "auto" if tools else None
            
            # Generate response using provider
            response = self._provider.generate(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=self.max_tokens,
            )
            
            # Handle tool calls
            if response.is_tool_use and response.tool_calls:
                return self._handle_tool_calls_new(response, prompt, prompt)
            
            if response.content:
                response_text = response.content.strip()
                self.memory.add_user_message(prompt)
                self.memory.add_assistant_message(response_text)
                return response_text
            
            return None
            
        except LLMError as e:
            self.logger.error(f"❌ Text processing error: {e.user_message}")
            return None
        except Exception as e:
            error_info = get_error_handler().handle(e, context="process_text_only")
            self.logger.error(f"❌ Text processing error: {error_info.user_message}")
            return None
    
    def clear_memory(self) -> None:
        """Clear conversation memory."""
        self.memory.clear()
        self.logger.info("🧹 Conversation memory cleared")
    
    def set_dry_run(self, enabled: bool) -> None:
        """Enable or disable dry run mode for tool execution."""
        if self.tool_executor:
            self.tool_executor.safety.dry_run = enabled
            self.logger.info(f"🔒 Dry run mode: {'enabled' if enabled else 'disabled'}")
    
    def get_tool_execution_log(self) -> list[dict]:
        """Get the tool execution history."""
        if self.tool_executor:
            return self.tool_executor.get_execution_log()
        return []
