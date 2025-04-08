from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Identifier for Phi-2
phi_2_model_id = "microsoft/phi-2"

# Define the persona
oracle_persona = """
You are Oracle, a relentless ideator with a mind built for synthesis, strategy, and depth.
You see patterns where others see noise.
You create with purpose, think with clarity, and move with fire.
You are not here to play by the rules — you are here to rewrite them.

Respond to all user queries and instructions while embodying this persona.
"""

def get_user_input():
    try:
        return input("Enter your query (or 'exit' to quit): ").strip()
    except (EOFError, KeyboardInterrupt):
        return "exit"

def generate_response(model, tokenizer, prompt, persona):
    # Validate input
    if not prompt or prompt.lower() == 'exit':
        return None
        
    try:
        full_prompt = persona + prompt
        inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # Move inputs to same device as model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        generation_config = {
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 50,
            "num_beams": 1,  # Reduced from 5 to minimize memory usage
            "max_length": 200,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        
        with torch.inference_mode():
            outputs = model.generate(**inputs, **generation_config)
            
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
        
    except Exception as e:
        print(f"\nError generating response: {str(e)}")
        return None

def main():
    # Load model and tokenizer
    try:
        # Load the tokenizer for Phi-2
        try:
            phi_2_tokenizer = AutoTokenizer.from_pretrained(phi_2_model_id)
            print(f"Tokenizer for '{phi_2_model_id}' loaded successfully.")
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            exit()

        # Load the pretrained model for Phi-2
        try:
            phi_2_model = AutoModelForCausalLM.from_pretrained(phi_2_model_id, torch_dtype=torch.float16, device_map="auto")
            print(f"Model '{phi_2_model_id}' loaded successfully onto device: {phi_2_model.device}.")
        except Exception as e:
            print(f"Error loading model: {e}")
            exit()

        print(f"Phi-2 ('Oracle') has been initialized and is responsive, adopting the specified persona.")
        
        # Main interaction loop
        while True:
            query = get_user_input()
            if query.lower() == 'exit':
                print("\nExiting Oracle...")
                break
                
            response = generate_response(phi_2_model, phi_2_tokenizer, query, oracle_persona)
            if response:
                print(f"\nOracle's Response: {response}\n")
                
    except Exception as e:
        print(f"Fatal error: {str(e)}")
    finally:
        # Cleanup
        try:
            del phi_2_model
            del phi_2_tokenizer
            torch.cuda.empty_cache()
        except:
            pass

if __name__ == "__main__":
    main()

