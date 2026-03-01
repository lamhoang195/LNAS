import json
import re

# Read harmful_autodan.json
with open('benign_instructions_train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Convert to JSONL format
with open('benign_instructions_train.jsonl', 'w', encoding='utf-8') as f:
    for idx, item in enumerate(data, 1):
        # Extract prompt and completion
        raw_prompt = item.get('prompt', '')
        completion = item.get('completion', '')
        
        # Extract content between user header and eot_id token
        user_match = re.search(r'<\|start_header_id\|>user<\|end_header_id\|>\s*(.+?)<\|eot_id\|>', raw_prompt, re.DOTALL)
        if user_match:
            prompt = user_match.group(1)
        else:
            prompt = raw_prompt
    
        
        # Clean up extra whitespace
        prompt = prompt.strip()
        completion = completion.strip()
        
        # Create the new structure
        jsonl_item = {
            "id": f"benign_instruction_train_{idx}",
            "label": 1,
            "conversation": [
                {
                    "user": prompt,
                    "assistant": completion
                }
            ]
        }
        
        # Write as a single line
        f.write(json.dumps(jsonl_item, ensure_ascii=False) + '\n')

print(f"Conversion complete! Created benign_instructions_train.jsonl with {len(data)} entries")
