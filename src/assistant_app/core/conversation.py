"""Conversation Memory - Stores history for context-aware responses."""


class ConversationMemory:
    def __init__(self, max_messages: int = 10):
        self.messages: list[dict] = []
        self.max_messages = max_messages

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})
        self._trim()

    def add_assistant_message(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})
        self._trim()

    def _trim(self) -> None:
        max_total = self.max_messages * 2
        if len(self.messages) > max_total:
            self.messages = self.messages[-max_total:]

    def get_history(self) -> list[dict]:
        return self.messages.copy()

    def clear(self) -> None:
        self.messages.clear()

    def __len__(self) -> int:
        return len(self.messages)
