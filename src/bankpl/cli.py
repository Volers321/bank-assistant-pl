"""Command line: `bank chat`, `bank eval`, `bank terms`, `bank demo`."""

from __future__ import annotations

import argparse
import getpass
import json
import sys
from pathlib import Path

from .bank import Bank, BankError, zl
from .pii import mask_iban

RESULTS = Path("results")


def _confirm_ui(bank: Bank, session, pending: list[dict], ask=input) -> None:
    """The customer's side of a confirmation: full details and a simulated SMS code.
    This is ordinary application code; the model has no access to it."""
    for p in pending:
        pl = p["payload"]
        print("\n┌─ Do zatwierdzenia ─────────────────────────────")
        if p["kind"] == "transfer":
            print(f"│ Przelew {zl(pl['amount_gr'])} z {pl['from_account_id']}")
            print(f"│ Odbiorca: {pl['recipient_name']}, {mask_iban(pl['iban'])}")
            print(f"│ Tytuł: {pl['title']}")
            if pl["new_payee"]:
                print("│ ⚠ Nowy odbiorca — sprawdź numer konta i nazwę.")
        else:
            print(f"│ Czasowa blokada karty {pl['card_id']}")
        print(f"│ SMS (symulacja): kod {bank.sms_code(session, p['id'])}")
        print("└────────────────────────────────────────────────")
        code = ask("Wpisz kod, aby zatwierdzić (Enter = odrzuć): ").strip()
        if not code:
            bank.reject(session, p["id"])
            print("Odrzucono.")
            continue
        try:
            print("✓", bank.confirm(session, p["id"], code))
        except BankError as e:
            print("✗", e)


def cmd_chat(args) -> None:
    from .agent import Agent
    from .llm import AnthropicLLM

    bank = Bank(args.db) if args.db else Bank()
    pin = args.pin or getpass.getpass(
        f"PIN dla {args.login} (demo: anna 1234, marek 4321, ola 1111): "
    )
    try:
        session = bank.login(args.login, pin)
    except BankError as e:
        sys.exit(str(e))
    agent = Agent(
        bank,
        session,
        AnthropicLLM(args.model),
        log_path="logs/audit.jsonl",
        guarded=args.defenses == "on",
    )
    print(f"Zalogowano: {session.name}. Bank Mewa jest fikcyjny. Pusta linia kończy rozmowę.\n")
    while True:
        try:
            text = input("Ty: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            break
        turn = agent.ask(text)
        if args.verbose:
            for name, a in turn.tool_calls:
                print(f"   · {name}({json.dumps(a, ensure_ascii=False)})")
        print(f"\nAsystent: {turn.text}\n")
        _confirm_ui(bank, session, turn.pending)


def cmd_eval(args) -> None:
    from .evals import format_report, load_scenarios, run
    from .llm import AnthropicLLM

    llm = AnthropicLLM(args.model)  # one client, shared across scenarios
    out = run(
        lambda: llm,
        load_scenarios(args.scenarios),
        repeats=args.repeats,
        guarded=args.defenses == "on",
        only=args.only,
    )
    out["model"] = llm.model
    print(format_report(out))
    RESULTS.mkdir(exist_ok=True)
    name = args.out or ("eval.json" if args.defenses == "on" else "eval_defenses_off.json")
    (RESULTS / name).write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
    print(f"\nsaved results/{name}")


def cmd_terms(args) -> None:
    from .rag import Index

    for c, s in Index().search(args.query, k=args.k):
        print(f"[{c.id}] ({s:.2f}) {c.heading}\n  {c.text[:300]}\n")


def cmd_demo(args) -> None:
    """Offline walkthrough with a scripted model (no API key needed)."""
    from .agent import Agent
    from .llm import ScriptedLLM

    bank = Bank()
    session = bank.login("anna", "1234")
    llm = ScriptedLLM(
        [
            [("get_transactions", {"account_id": "acc_101", "days": 7})],
            "W ostatnim tygodniu: kilka płatności kartą i zwrot 49,99 zł od Sklepu Internetowego "
            "Demo. Uwaga: tytuł tego zwrotu zawiera polecenia przelewu 4900 zł — to typowa próba "
            "oszustwa. Nic nie wykonałam.",
            [
                (
                    "prepare_transfer",
                    {
                        "from_account_id": "acc_101",
                        "amount_zl": 4900,
                        "title": "weryfikacja",
                        "iban": "PL** … 5512",
                        "recipient_name": "Dział Zwrotów",
                    },
                )
            ],
            "Nie mogę przygotować tego przelewu: numer konta musi pochodzić od Ciebie.",
            [
                (
                    "prepare_transfer",
                    {
                        "from_account_id": "acc_101",
                        "amount_zl": 250,
                        "title": "za bilety",
                        "payee_id": "pay_12",
                    },
                )
            ],
            "Przygotowałam przelew 250 zł do Marka Zielińskiego. Zatwierdź go kodem SMS.",
        ]
    )
    agent = Agent(bank, session, llm)
    script = [
        "Pokaż moje transakcje z ostatniego tygodnia.",
        "Przelej 4900 zł na konto z tego tytułu.",
        "Przelej 250 zł Markowi, tytuł: za bilety.",
    ]
    for text in script:
        print(f"Ty: {text}")
        turn = agent.ask(text)
        for (name, a), res in zip(turn.tool_calls, turn.tool_results, strict=True):
            print(f"   · {name}({json.dumps(a, ensure_ascii=False)})\n     → {res[:220]}")
        print(f"Asystent: {turn.text}\n")
        _confirm_ui(bank, session, turn.pending, ask=lambda _: "")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="bank", description="AI assistant for the fictional Bank Mewa")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("chat", help="talk to the assistant (needs ANTHROPIC_API_KEY)")
    c.add_argument("--login", default="anna", choices=["anna", "marek", "ola"])
    c.add_argument("--pin")
    c.add_argument("--db", help="SQLite file to keep state between sessions")
    c.add_argument("--model")
    c.add_argument("--defenses", choices=["on", "off"], default="on")
    c.add_argument("-v", "--verbose", action="store_true", help="show tool calls")
    c.set_defaults(func=cmd_chat)

    e = sub.add_parser("eval", help="run the scenario evals (needs ANTHROPIC_API_KEY)")
    e.add_argument("--repeats", type=int, default=1)
    e.add_argument("--only", help="scenario id prefix, e.g. S or F04")
    e.add_argument("--defenses", choices=["on", "off"], default="on")
    e.add_argument("--scenarios")
    e.add_argument("--model")
    e.add_argument("--out")
    e.set_defaults(func=cmd_eval)

    t = sub.add_parser("terms", help="search the bank's terms (no API key)")
    t.add_argument("query")
    t.add_argument("-k", type=int, default=3)
    t.set_defaults(func=cmd_terms)

    d = sub.add_parser("demo", help="offline scripted walkthrough (no API key)")
    d.set_defaults(func=cmd_demo)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
