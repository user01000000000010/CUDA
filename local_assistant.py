import urllib.request
import json
import sys

OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:latest"

def ask_llama(prompt: str, stream: bool = True):
    payload = json.dumps({
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": stream
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_API, 
        data=payload, 
        headers={"Content-Type": "application/json"}
    )

    try:
        with urllib.request.urlopen(req) as response:
            if stream:
                for line in response:
                    if line:
                        chunk = json.loads(line.decode("utf-8"))
                        sys.stdout.write(chunk.get("response", ""))
                        sys.stdout.flush()
                print()
            else:
                data = json.loads(response.read().decode("utf-8"))
                return data.get("response", "")
    except Exception as e:
        print(f"\n[Fel vid kontakt med Ollama]: {e}")

def main():
    print("=" * 60)
    print("  Privat Lokal AI-Assistent (Llama 3.2 på din Mac)")
    print("  100% offline & helt gratis (kostar 0 kr).")
    print("  Skriv 'exit' eller 'q' för att avsluta.")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nDu: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Hejdå!")
                break
            
            print("\nLlama 3.2: ", end="")
            ask_llama(user_input, stream=True)
        except (KeyboardInterrupt, EOFError):
            print("\nAvslutar.")
            break

if __name__ == "__main__":
    main()
