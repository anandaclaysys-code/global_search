import json
import os

DEFAULT_INPUT_PATH = r"C:\Users\anandha.kumar\OneDrive - ClaySys Technologies\Desktop\Code Base\Global search\data\raw\SCCU-test-enhanced.json"
DEFAULT_OUTPUT_PATH = r"C:\Users\anandha.kumar\OneDrive - ClaySys Technologies\Desktop\Code Base\Global search\data\processed\restructured_dataset.json"

def restructure_dataset(file_path: str = DEFAULT_INPUT_PATH) -> dict:
    """
    Reads the dataset JSON and converts it into a dictionary mapping
    each intent to its unique utterances and entity keywords.
    
    Format:
    {
        "intent_name": [
            "utterance_or_keyword_1",
            "utterance_or_keyword_2",
            ...
        ]
    }
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    restructured = {}

    assets = data.get("assets", {})
    
    # 1. Extract utterances text grouped by intent
    utterances = assets.get("utterances", [])
    for u in utterances:
        intent = u.get("intent")
        text = u.get("text")
        if intent and text:
            # Strip and normalize spaces
            clean_text = text.strip()
            if clean_text:
                if intent not in restructured:
                    restructured[intent] = set()
                restructured[intent].add(clean_text)

    # 2. Extract synonyms (keywords) from entities grouped by intent (listKey matches intent)
    entities = assets.get("entities", [])
    for entity in entities:
        sublists = entity.get("list", {}).get("sublists", [])
        for sublist in sublists:
            intent = sublist.get("listKey")
            if not intent:
                continue
            
            synonyms_list = sublist.get("synonyms", [])
            for synonym_entry in synonyms_list:
                values = synonym_entry.get("values", [])
                for val in values:
                    clean_val = val.strip()
                    if clean_val:
                        if intent not in restructured:
                            restructured[intent] = set()
                        restructured[intent].add(clean_val)

    # Convert sets to sorted lists for JSON serialization and consistency
    final_dict = {intent: sorted(list(items)) for intent, items in restructured.items()}
    return final_dict

if __name__ == "__main__":
    try:
        result = restructure_dataset()
        print(f"Successfully restructured dataset.")
        print(f"Number of unique intents: {len(result)}")
        
        # Display some statistics
        for intent, items in list(result.items())[:5]:
            print(f"  - Intent '{intent}': {len(items)} utterances/keywords")
            
        # Ensure output directory exists and save the result
        os.makedirs(os.path.dirname(DEFAULT_OUTPUT_PATH), exist_ok=True)
        with open(DEFAULT_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        print(f"Saved restructured dataset to: {DEFAULT_OUTPUT_PATH}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

