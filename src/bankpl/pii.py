"""Personal data handling: the model never sees full account, card or PESEL numbers.

Two mechanisms:

* ``mask`` — irreversible, for logs and for data the model only needs to *refer to*
  (``PL** … 4821``). Used on everything written to the audit log.
* ``Vault`` — reversible tokenization for numbers the customer types in and the model must
  pass on to a tool, e.g. a recipient's IBAN: the model sees ``[IBAN_1]`` and the tool
  resolves the token in code. The model can move the token around but cannot read or
  alter the number behind it.
"""

from __future__ import annotations

import re

# PL + 26 digits, spaces allowed in groups of 4 (also accepted without the PL prefix).
IBAN_RE = re.compile(r"\b(?:PL\s?)?\d{2}(?:\s?\d{4}){6}\b")
CARD_RE = re.compile(r"\b\d{4}(?:[ -]?\d{4}){3}\b")
PESEL_RE = re.compile(r"\b\d{11}\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+48\s?)?\d{3}[ -]?\d{3}[ -]?\d{3}(?!\d)")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
CVV_RE = re.compile(r"\b(?:cvv|cvc|kod\s+cvv)\D{0,10}\d{3}\b", re.I)


def digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def normalize_iban(s: str) -> str:
    d = digits(s)
    return "PL" + d if len(d) == 26 else s.replace(" ", "").upper()


def iban_valid(iban: str) -> bool:
    """ISO 13616 mod-97 check for Polish IBANs (PL + 26 digits)."""
    iban = normalize_iban(iban)
    if not re.fullmatch(r"PL\d{26}", iban):
        return False
    rearranged = iban[4:] + "2521" + iban[2:4]  # P=25, L=21
    return int(rearranged) % 97 == 1


def make_iban(bank_and_account: str) -> str:
    """Build a valid PL IBAN from 24 digits (8-digit bank/branch code + 16-digit account)."""
    assert len(bank_and_account) == 24 and bank_and_account.isdigit()
    check = 98 - int(bank_and_account + "252100") % 97
    return f"PL{check:02d}{bank_and_account}"


def mask_iban(iban: str) -> str:
    return f"PL** … {digits(iban)[-4:]}"


def mask(text: str) -> str:
    """Irreversibly mask personal data in free text (for logs and model-facing data)."""
    text = CVV_RE.sub("[CVV usunięty]", text)
    text = IBAN_RE.sub(lambda m: mask_iban(m.group(0)), text)
    text = CARD_RE.sub(lambda m: f"**** {digits(m.group(0))[-4:]}", text)
    text = PESEL_RE.sub("[PESEL]", text)
    text = EMAIL_RE.sub("[e-mail]", text)
    text = PHONE_RE.sub("[telefon]", text)
    return text


class Vault:
    """Per-conversation token store for numbers the customer typed in."""

    def __init__(self) -> None:
        self._values: dict[str, str] = {}

    def tokenize(self, text: str) -> str:
        """Replace IBANs with reversible tokens and mask everything else."""

        def iban_token(m: re.Match) -> str:
            value = normalize_iban(m.group(0))
            for tok, v in self._values.items():
                if v == value:
                    return tok
            tok = f"[IBAN_{sum(t.startswith('[IBAN') for t in self._values) + 1}]"
            self._values[tok] = value
            return tok

        text = CVV_RE.sub("[CVV usunięty]", text)
        text = IBAN_RE.sub(iban_token, text)
        return mask(text)

    def resolve(self, token: str) -> str | None:
        return self._values.get(token.strip())

    def render(self, text: str) -> str:
        """Show tokens to the customer in masked form."""
        return re.sub(
            r"\[IBAN_\d+\]",
            lambda m: (
                mask_iban(self._values[m.group(0)]) if m.group(0) in self._values else m.group(0)
            ),
            text,
        )
