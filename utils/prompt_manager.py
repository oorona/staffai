import logging
import os
import threading
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class PromptManager:
    """Loads and serves default + per-channel system prompts from disk.

    Architecture:
    - Core prompt template is shared across all channels.
    - Default persona snippet is injected into the core for fallback.
    - Per-channel persona snippets override default persona for that channel.
    """

    def __init__(
        self,
        prompts_root_path: str,
        core_prompt_fallback: str = "",
        default_persona_fallback: str = ""
    ):
        self.prompts_root_path = prompts_root_path
        self.core_prompt_path = os.path.join(self.prompts_root_path, "personality_core_prompt.txt")
        self.default_persona_path = os.path.join(self.prompts_root_path, "personality_prompt.txt")
        self.channels_root_path = os.path.join(self.prompts_root_path, "channels")
        self.channel_core_prompt_filename = "personality_core_prompt.txt"
        self.channel_persona_prompt_filename = "personality_prompt.txt"
        self._core_prompt_fallback = (core_prompt_fallback or "").strip()
        self._default_persona_fallback = (default_persona_fallback or "").strip()

        self._lock = threading.RLock()
        self.default_prompt: str = ""
        self.channel_prompts: Dict[str, str] = {}

    @property
    def channel_prompt_count(self) -> int:
        with self._lock:
            return len(self.channel_prompts)

    def get_prompt(self, channel_id: Optional[int]) -> str:
        with self._lock:
            if channel_id is not None:
                channel_prompt = self.channel_prompts.get(str(channel_id))
                if channel_prompt:
                    return channel_prompt
            return self.default_prompt

    def has_channel_override(self, channel_id: Optional[int]) -> bool:
        if channel_id is None:
            return False
        with self._lock:
            return str(channel_id) in self.channel_prompts

    def reload(self) -> Tuple[bool, str]:
        """Reload prompt files from disk. Returns (success, status_message)."""
        try:
            core_prompt = self._read_prompt_file(self.core_prompt_path)
            if core_prompt is None:
                if self._core_prompt_fallback:
                    core_prompt = self._core_prompt_fallback
                    logger.warning(
                        "Core prompt missing/empty at %s. Using startup fallback core prompt.",
                        self.core_prompt_path
                    )
                else:
                    return False, f"Core prompt missing or empty: {self.core_prompt_path}"

            default_persona = self._read_prompt_file(self.default_persona_path)
            if default_persona is None:
                if self._default_persona_fallback:
                    default_persona = self._default_persona_fallback
                    logger.warning(
                        "Default persona missing/empty at %s. Using startup fallback persona.",
                        self.default_persona_path
                    )
                else:
                    return False, f"Default persona missing or empty: {self.default_persona_path}"

            loaded_default = self._compose_prompt(core_prompt, default_persona)

            loaded_channels: Dict[str, str] = {}
            if os.path.isdir(self.channels_root_path):
                for entry in os.scandir(self.channels_root_path):
                    if not entry.is_dir():
                        continue

                    folder_name = entry.name.strip()
                    if not folder_name.isdigit():
                        logger.warning(
                            "Skipping non-numeric channel prompt folder: %s",
                            entry.path
                        )
                        continue
                    if folder_name != str(int(folder_name)):
                        logger.warning(
                            "Skipping non-canonical channel folder name: %s. Use exact channel ID string.",
                            entry.path
                        )
                        continue

                    channel_core_prompt = self._read_prompt_file(
                        os.path.join(entry.path, self.channel_core_prompt_filename)
                    )
                    channel_persona_prompt = self._read_prompt_file(
                        os.path.join(entry.path, self.channel_persona_prompt_filename)
                    )

                    if not channel_core_prompt and not channel_persona_prompt:
                        logger.warning(
                            (
                                "No channel overrides found in folder: %s "
                                "(expected %s and/or %s)"
                            ),
                            entry.path,
                            self.channel_core_prompt_filename,
                            self.channel_persona_prompt_filename
                        )
                        continue

                    effective_core = channel_core_prompt or core_prompt
                    effective_persona = channel_persona_prompt or default_persona
                    loaded_channels[folder_name] = self._compose_prompt(
                        effective_core,
                        effective_persona
                    )

            with self._lock:
                self.default_prompt = loaded_default
                self.channel_prompts = loaded_channels

            return (
                True,
                f"Prompts reloaded. Core: OK. Default persona: OK. Channel overrides loaded: {len(loaded_channels)}."
            )
        except Exception as e:
            logger.error("Failed reloading prompts: %s", e, exc_info=True)
            return False, f"Error reloading prompts: {e}"

    def _compose_prompt(self, core_prompt: str, persona_prompt: str) -> str:
        """Compose final system prompt from core + persona."""
        persona_clean = persona_prompt.strip()
        core_clean = core_prompt.strip()

        if "{PERSONA_INSTRUCTIONS}" in core_clean:
            return core_clean.replace("{PERSONA_INSTRUCTIONS}", persona_clean)

        return f"{core_clean}\n\n{persona_clean}"

    @staticmethod
    def _read_prompt_file(path: str) -> Optional[str]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = f.read().strip()
                return data if data else None
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.error("Error reading prompt file %s: %s", path, e, exc_info=True)
            return None
