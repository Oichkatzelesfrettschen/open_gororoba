import os
import re


def analyze_chat_log(file_path):
    print(f"--- Deep Analysis of {file_path} ---")

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"Total lines read: {len(lines)}")

    non_user_lines = []
    user_block = False

    # Simple state machine to capture non-user content
    # Assuming 'You said:' or similar markers denote user blocks
    # and 'ChatGPT said:' or 'Consensus said:' denote AI blocks.

    current_speaker = "Unknown"
    ai_content = []

    for line in lines:
        if re.search(r"You said:", line, re.IGNORECASE):
            current_speaker = "User"
        elif re.search(r"(ChatGPT|Consensus) said:", line, re.IGNORECASE):
            current_speaker = "AI"

        if current_speaker == "AI":
            ai_content.append(line)

    print(f"Extracted {len(ai_content)} lines of AI/System content.")

    # 1. Keyword Extraction (Math/Physics)
    keywords = {
        "Equation": 0,
        "Theorem": 0,
        "Lemma": 0,
        "Proof": 0,
        "Sedenion": 0,
        "Gravastar": 0,
        "Fractal": 0,
        "Negative Dimension": 0,
        "Zero Divisor": 0,
        "Hamiltonian": 0,
        "Lagrangian": 0,
        "LIGO": 0,
        "Chern": 0,
        "Code": 0
    }

    # 2. Extract Latent Tasks/Gaps
    todos = []

    full_text = "".join(ai_content)

    for k in keywords:
        keywords[k] = len(re.findall(k, full_text, re.IGNORECASE))

    print("\n--- Concept Frequency Analysis ---")
    for k, v in keywords.items():
        print(f"{k}: {v}")

    # 3. Identify Code Blocks for Re-verification
    code_blocks = re.findall(r'```(.*?)```', full_text, re.DOTALL)
    print(f"\nFound {len(code_blocks)} code blocks in AI responses.")

    # 4. Search for specific unaddressed prompts or 'loose threads'
    # Searching for phrases like "Next steps", "Future work", "TODO"
    matches = re.findall(r"(?:Next steps|Future work|To do|We should|explore)(.*)", full_text, re.IGNORECASE)

    print(f"\n--- Potential Action Items Extracted ({len(matches)} found) ---")
    # Show a sample of unique action items to avoid spam
    unique_actions = list(set(matches))
    for i, action in enumerate(unique_actions[:20]):
        if len(action) > 10: # Filter short noise
            print(f"- {action.strip()}")

    # 5. Extract "Hidden" Math (Equations that might have been skipped)
    # Looking for latex delimiters
    latex_eqs = re.findall(r'\$\$(.*?)\$\$', full_text, re.DOTALL)
    print(f"\n--- Distinct Mathematical Formalisms ({len(latex_eqs)} found) ---")

    # Save extracted math to a file for 'math-heavy' review
    with open("data/artifacts/extracted_equations.md", "w") as f:
        f.write("# Extracted Equations from Chat Log\n\n")
        for i, eq in enumerate(latex_eqs):
            f.write(f"## Eq {i+1}\n$$ {eq.strip()} $$\\n\n")

    print("Saved extracted equations to data/artifacts/extracted_equations.md")

if __name__ == "__main__":
    analyze_chat_log("convos/1_read_each_nonuser_line.md")
