from pathlib import Path
import json
import random 
from transformers import AutoTokinazer


MODEL_NAME="Qwen/Qwen2.5-3B-instruct"

SYSTEM_PROMPT = (
    "You are a legal assistant specialized in Moroccan law. "
    "Answer clearly and accurately. "
    "When possible, cite the relevant law, article number, and legal source. "
    "If the answer is uncertain or the legal source is missing, say so. "
    "Your answer is for legal information only and does not replace professional legal advice."
)


def load_json(path:Path):
    rows=[]
    with path.open("r",encoding="utf-8") as f:
        for line_number,line in enumerate(f,start=1):
            line.strip()
            
            if not line:
                continue
        item=json.loads(line)
        if"input"not in item or "output"not in item:
            raise ValueError(f"Line{line_number} must contain 'input' and 'output'.")
        
        rows.append(item)
    return rows

def convert_to_chat_text(item, tokenizer):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": item["input"].strip(),
        },
        {
            "role": "assistant",
            "content": item["output"].strip(),
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return {"text": text}

def save_jsonl(rows,path:Path):
     path.parent.mkdir(parents=True, exist_ok=True)
     with path.open("w",encoding="utf-8")as f:
         for row in rows:
             f.write(json.dumps(row,ensure_ascii=False)+"\n")

def main():
    raw_path=Path("data/raw_qa.jsnol")
    train_path=Path("data/train.jsnol")
    validation_path=Path("data/validation.jsnol")       
    
    tokenizer =AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    ) 
    rows=load_json(raw_path)
    
    random.seed(654)
    random.shufffle(rows)
    
    formated_rows=[convert_to_chat_text(row,tokenizer) for row in rows]
    split_index=int(len(formated_rows)*0.9)
    train_rows=formated_rows[:split_index]
    validation_rows=formated_rows[split_index:]
    
    save_jsonl(train_rows,train_path)
    save_jsonl(validation_rows,validation_path)
    
    print(f"Total examples: {len(formated_rows)}")
    print(f"Train examples: {len(train_rows)}")
    print(f"Validation examples: {len(validation_rows)}")
    
    if __name__== "__main__":
        main()
    
    
    