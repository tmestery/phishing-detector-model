"""
predict.py — Run inference with your fine-tuned phishing detector.
Run: python predict.py
     python predict.py --url "http://sketchy-site.ru/login"
"""

import argparse
from transformers import pipeline


MODEL_DIR = "phishing-detector-model"

# Demo URLs:
DEMO_URLS = [
    "https://www.google.com/search?q=weather",
    "https://github.com/huggingface/transformers",
    "https://www.cisco.com/c/en/us/products/security",
    "http://paypa1-secure-login.ru/verify/account",
    "http://apple-id-suspended.com/unlock-now",
    "http://192.168.1.1/admin/login?redirect=steal-creds",
    "http://amaz0n-prize-winner.net/claim?user=you",
    "https://en.wikipedia.org/wiki/Phishing",
]


def load_model():
    print(f"🤖 Loading model from ./{MODEL_DIR}...")
    clf = pipeline("text-classification", model=MODEL_DIR, tokenizer=MODEL_DIR)
    print("✅ Model loaded.\n")
    return clf


def predict(clf, url: str) -> dict:
    result = clf(url)[0]
    is_phishing = result["label"] == "LABEL_1"
    return {
        "url": url,
        "verdict": "🚨 PHISHING" if is_phishing else "✅ LEGIT",
        "confidence": result["score"],
        "is_phishing": is_phishing,
    }


def print_result(r: dict):
    bar_len = int(r["confidence"] * 20)
    bar = "█" * bar_len + "░" * (20 - bar_len)
    print(f"{r['verdict']}  [{bar}] {r['confidence']:.1%}")
    print(f"  {r['url']}\n")


def interactive_mode(clf):
    print("🔍 Interactive mode — type a URL to classify (or 'q' to quit)\n")
    while True:
        url = input("URL: ").strip()
        if url.lower() in ("q", "quit", "exit"):
            break
        if not url:
            continue
        r = predict(clf, url)
        print()
        print_result(r)


def main():
    parser = argparse.ArgumentParser(description="Phishing URL Detector")
    parser.add_argument("--url", type=str, help="Single URL to classify")
    parser.add_argument("--demo", action="store_true", help="Run on demo URLs")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    clf = load_model()

    # Single URL from CLI
    if args.url:
        r = predict(clf, args.url)
        print_result(r)

    # Demo mode
    elif args.demo or not args.interactive:
        print("─" * 55)
        print("  DEMO: Classifying sample URLs")
        print("─" * 55 + "\n")
        for url in DEMO_URLS:
            r = predict(clf, url)
            print_result(r)

    # Interactive
    if args.interactive:
        interactive_mode(clf)


if __name__ == "__main__":
    main()