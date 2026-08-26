"""Language-instruction caching (README "Instruction handling"): the
policy node must not require a fresh ``/task_instruction`` message every
control tick -- it caches the latest valid one and reuses it.
"""

DEFAULT_INSTRUCTION = "Pick up the red cube."


class InstructionCache:
    """Holds the most recently received non-empty instruction string.

    Args:
        default: Returned by ``get()`` until the first valid instruction
            is received (README "runtime configuration": instruction is
            configurable, with a sensible default rather than blocking
            control until a language message arrives).
    """

    def __init__(self, default: str = DEFAULT_INSTRUCTION) -> None:
        self._default = default
        self._current = default

    def update(self, instruction: str) -> bool:
        """Cache ``instruction`` if non-empty. Returns whether it was accepted."""
        if not isinstance(instruction, str) or not instruction.strip():
            return False
        self._current = instruction
        return True

    def get(self) -> str:
        return self._current

    def reset(self) -> None:
        self._current = self._default
