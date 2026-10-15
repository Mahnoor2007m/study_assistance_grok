"""AI Study Assistant (ChromaDB + Groq) - command line version.

Usage:
    python study_assistant.py                      interactive chat
    python study_assistant.py --ask "What is overfitting?"
    python study_assistant.py --rebuild            force re-creating the embeddings
"""
import argparse
import sys

import config
from embedder import get_embedder
from groq_client import GroqClient, GroqError
from rag import StudyAssistant, ensure_index, format_sources

HELP = """Commands:
  <your question>        ask anything from the chapters
  /summary <topic>       short study summary of a topic
  /quiz <topic>          5 practice questions on a topic
  /sources               show the sources of the last answer
  /help                  show this help
  /exit                  quit"""


def show(result):
    print("\n" + result["answer"])
    if result["sources"]:
        print("\nSources:")
        for line in format_sources(result["sources"]):
            print("  " + line)
    print()


def main():
    parser = argparse.ArgumentParser(description="AI Study Assistant (RAG + ChromaDB + Groq)")
    parser.add_argument("--ask", help="ask one question and exit")
    parser.add_argument("--rebuild", action="store_true", help="re-create the embeddings from the PDFs")
    parser.add_argument("--embedder", default="minilm", choices=["minilm", "hash"], help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not config.get_api_key():
        message = "No API key found. Create a .env file with GROQ_API_KEY=your_key (get a free key at https://console.groq.com/keys)."
        if args.ask:
            sys.exit(f"Error: {message}")
        print(f"[Warning] {message}\n          You can still start, but questions will fail until the key is set.")

    embedder = get_embedder(args.embedder)
    try:
        store = ensure_index(embedder, force=args.rebuild)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"Error: {e}")
    assistant = StudyAssistant(embedder, store, GroqClient())

    if args.ask:
        try:
            show(assistant.ask(args.ask))
        except GroqError as e:
            sys.exit(f"Error: {e}")
        return

    print("\n" + "=" * 60 + f"\n   AI STUDY ASSISTANT  (ChromaDB + Groq: {config.get_model()})\n" + "=" * 60)
    print(HELP)
    history, last = [], None
    while True:
        try:
            text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not text:
            continue
        try:
            if text in ("/exit", "/quit"):
                print("Goodbye!")
                break
            elif text == "/help":
                print(HELP)
            elif text == "/sources":
                print("\n".join(format_sources(last["sources"])) if last and last["sources"] else "No sources yet.")
            elif text.startswith("/summary "):
                last = assistant.summarize(text[9:].strip()); show(last)
            elif text.startswith("/quiz "):
                last = assistant.quiz(text[6:].strip()); show(last)
            else:
                last = assistant.ask(text, history)
                history.append((text, last["answer"]))
                show(last)
        except GroqError as e:
            print(f"\n[Error] {e}")


if __name__ == "__main__":
    main()
