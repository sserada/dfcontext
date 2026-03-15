"""Token counting with tiktoken and character-based fallback."""

from __future__ import annotations

import warnings
from typing import Any

_FALLBACK_WARNED = False
_CHARS_PER_TOKEN = 4  # Rough estimate for English text


class TokenCounter:
    """Count, check, and truncate text tokens.

    Parameters
    ----------
    encoding_name : str
        tiktoken encoding name (e.g. "cl100k_base"). Used only when
        tiktoken is installed; otherwise falls back to character-based
        estimation.
    """

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self._encoding_name = encoding_name
        self._encoder: Any = self._load_encoder()

    def _load_encoder(self) -> Any:
        try:
            import tiktoken

            return tiktoken.get_encoding(self._encoding_name)
        except ImportError:
            global _FALLBACK_WARNED  # noqa: PLW0603
            if not _FALLBACK_WARNED:
                warnings.warn(
                    "tiktoken is not installed. Token counts will be estimated "
                    "using character length (1 token ≈ 4 chars). Install "
                    "tiktoken for accurate counting: pip install tiktoken",
                    UserWarning,
                    stacklevel=2,
                )
                _FALLBACK_WARNED = True
            return None

    def count(self, text: str) -> int:
        """Count the number of tokens in *text*.

        Parameters
        ----------
        text : str
            The text to count tokens for.

        Returns
        -------
        int
            Token count (exact with tiktoken, estimated otherwise).
        """
        if not text:
            return 0
        if self._encoder is not None:
            return len(self._encoder.encode(text))
        return max(1, len(text) // _CHARS_PER_TOKEN)

    def fits(self, text: str, budget: int) -> bool:
        """Check whether *text* fits within *budget* tokens.

        Parameters
        ----------
        text : str
            The text to check.
        budget : int
            Maximum allowed tokens.

        Returns
        -------
        bool
        """
        return self.count(text) <= budget

    def truncate(self, text: str, budget: int) -> str:
        """Truncate *text* so it fits within *budget* tokens.

        Parameters
        ----------
        text : str
            The text to truncate.
        budget : int
            Maximum allowed tokens.

        Returns
        -------
        str
            The (possibly truncated) text.
        """
        if budget <= 0:
            return ""
        if self.fits(text, budget):
            return text
        if self._encoder is not None:
            tokens = self._encoder.encode(text)
            return str(self._encoder.decode(tokens[:budget]))
        # Fallback: truncate by estimated character count
        max_chars = budget * _CHARS_PER_TOKEN
        return text[:max_chars]
