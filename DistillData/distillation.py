from requestHandler_api import RequestHandler, InsufficientQuotaError
from pathlib import Path
from openai import OpenAI
import json
import config as cfg
# We import the new API-specific request handler
import time
import sys

REQUESTS_PER_MINUTE = 500
# LLM prompt (No change)
FILTER_SYSTEM_PROMPT = """You are an expert psychologist. 
Your task is to judge if a word describes an emotion.
You MUST respond with ONLY ONE WORD: "Yes" or "No".
Do not say anything else. Your response must start with "Yes" or "No".
"Happy", "Sad", "Pleased", "Irritated" = "Yes".
"Hat", "Table", "Industry", "Walk" = "No"."""

def main():
    base_dir = Path(__file__).resolve().parent
    path_of_VAD_txt: Path = base_dir / "Original" / "NRC-VAD-Lexicon-v2.1.txt"
    path_to_Distilled: Path = base_dir / "Distilled"
    path_to_Distilled.mkdir(exist_ok=True) # Create distilled folder if it doesn't exist
    output_filepath = path_to_Distilled / "filtered_emotion_lexicon.json"

    if output_filepath.exists():
        print("✅ FINAL FILE FOUND: Distillation process is complete.")
        print("Stopping Docker container to save resources.")
        sys.exit(0)

    # Load Data (No change)
    print("Loading data.......................")
    original_data: list[dict] = load_vad_data(path_of_VAD_txt)
    distilled_data: list[dict] = []

    if not original_data:
        print("No data loaded. Exiting.")
        return

    print(f"Success! Loaded {len(original_data)} terms.")

    total_words = len(original_data)

    # Start API Client (Replaces server)
    try:
        API_KEY = cfg.API_CONFIG["OPENAI_API_KEY"]
        MODEL_NAME = cfg.API_CONFIG["MODEL_NAME"]
    except (AttributeError, KeyError):
        print("Error: 'API_CONFIG' not found in config.py.")
        print("Please add API_KEY and MODEL_NAME to config.py.")
        return

    if "YOUR_API_KEY" in API_KEY:
        print("Error: Please update 'OPENAI_API_KEY' in config.py.")
        return

    client = OpenAI(api_key=API_KEY)
    request_handler: RequestHandler = RequestHandler(client, MODEL_NAME)

    print(f"--- Client Ready. Starting distillation with model: {MODEL_NAME} ---")

    request_count = 0
    start_time = time.time()

    # Torturing API starts (No server needed)
    try:
        for i, word_data in enumerate(original_data):

            request_count += 1
            if request_count >= REQUESTS_PER_MINUTE:
                elapsed_time = time.time() - start_time

                if elapsed_time < 60:
                    sleep_time = 60 - elapsed_time
                    print(f"\n--- RPM Limit Hit! Sleeping for {sleep_time:.2f} seconds ---")
                    time.sleep(sleep_time)

                request_count = 0
                start_time = time.time()   
            
            term = word_data.get('term')

            if not term:
                continue

            # progress
            if (i + 1) % 100 == 0:
                print(f"\n--- Processing {i+1}/{total_words} ---")

            # message for llm
            messages = [
                {"role": "system", "content": FILTER_SYSTEM_PROMPT},
                {"role": "user", "content": f"Word: \"{term}\"\nAnswer (Yes or No):"}
            ]

            try:
                # call llm API
                answer = request_handler.sendMsg(messages)

                # cooking...........
                if answer == "yes":
                    print(f"  [Yes] -> {term}")
                    distilled_data.append(word_data)

                elif answer == "rate_limit_sleep":
                    print("...Retrying last word after sleep...")
                    pass

                else:
                    # cooking failed! ("no", "parse_failed", "api_error")
                    print(f"  [No]  -> {term} (Reason: {answer})")
                    pass
            
            except InsufficientQuotaError as e:
                print(f"\nFATAL ERROR: {e}")
                print("Stopping script. Please fix your OpenAI billing account.")
                break

            except Exception as e:
                # Handle API errors like rate limits
                print(f"Error processing word '{term}': {e}")
                if "rate limit" in str(e).lower():
                    print("Rate limit hit. Sleeping for 60 seconds...")
                    time.sleep(60)
                else:
                    time.sleep(5)

            # save it every 500 words
            if (i + 1) % 500 == 0:
                ctrl_c_json(output_filepath, distilled_data)
    
    except KeyboardInterrupt:
        print("\nCtrl+C detected! Stopping and saving final data...")
    
    finally:
        # Final save
        print("\n--- Distillation Finished ---")
        ctrl_c_json(output_filepath, distilled_data)
        print("Final dataset saved. Exiting.")


   
def load_vad_data(filepath: Path) -> list[dict]:
    """
    Reads a tab-separated VAD text file and converts it into a list of dictionaries.
    """
    vad_list_of_dicts = []
    
    print(f"Opening file: {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # 1. Read and skip the header (first line)
            header_line = next(f)
            print(f"Header skipped: {header_line.strip()}")
            
            headers = header_line.strip().split('\t')
            if len(headers) < 4:
                 print(f"Warning: Header has {len(headers)} columns, expected at least 4.")
                 headers = ['term', 'valence', 'arousal', 'dominance'] # Default

            # 2. Iterate over the remaining lines
            for line in f:
                # 3. Strip whitespace from both ends
                line = line.strip()
                
                # 4. Skip empty lines
                if not line:
                    continue
                
                try:
                    # 5. Split the line into parts based on the tab
                    parts = line.split('\t')
                    
                    if len(parts) >= 4:
                        # 6. Create a dictionary for the row
                        row_dict = {
                            headers[0]: parts[0],
                            headers[1]: float(parts[1]),
                            headers[2]: float(parts[2]),
                            headers[3]: float(parts[3])
                        }
                        
                        # 7. Add the dictionary to our main list
                        vad_list_of_dicts.append(row_dict)
                    else:
                        print(f"Skipping malformed line (not enough parts): {line}")
                        
                except ValueError:
                    # If float() conversion fails
                    print(f"Skipping bad data line (ValueError): {line}")

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return [] # Return an empty list on error
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return [] # Return an empty list on error

    print(f"Successfully loaded {len(vad_list_of_dicts)} terms.")
    return vad_list_of_dicts

def ctrl_c_json(filepath: Path, distilled_data:list[dict]):
    print(f"Checkpoint: Saving {len(distilled_data)} filtered terms to {filepath}...")
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(distilled_data, f, indent=4)
        print("Save complete.")
    except IOError as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    main()