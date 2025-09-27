import os
import sys 
import time
import urllib.request
import torch
torch.manual_seed(123)

from model import Llama3Model, generate, text_to_token_ids, token_ids_to_text
from tokenizer import Llama3Tokenizer, ChatFormat, clean_text
from model import LLAMA32_CONFIG_1B, LLAMA32_CONFIG_3B

# ===== Hyper-parameter =====
MODEL_FILE = "llama3.2-1B-base.pth"
MODEL_CONTEXT_LENGTH = 8192  # Support up to 131_072
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.
TOP_K = 1
TOKENIZER_FILE = "tokenizer.model"

# device = (
#     torch.device("cuda") if torch.cuda.is_available() else
#     torch.device("cpu")
# )

device = "cpu"


if __name__ == "__main__":
    input_prompt = "What is the capital of Viet Nam?"
    
    # 1. Load model
    if os.path.exists(MODEL_FILE) == False:
        print(f"[ERROR] Model does not exist !!!")
        sys.exit(0)

    if "1B" in MODEL_FILE:
        llama32_config = LLAMA32_CONFIG_1B
    elif "3B" in MODEL_FILE:
        llama32_config = LLAMA32_CONFIG_3B
    else:
        print(f"[ERROR] Check model file again !!!")
        sys.exit(0)

    llama32_config["context_length"] = MODEL_CONTEXT_LENGTH

    model = Llama3Model(llama32_config)
    checkpoint = torch.load(MODEL_FILE, weights_only=True, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)

    print(f"Model {MODEL_FILE} loaded successfully !!!")
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")


    # 2. Load tokenizer
    if not os.path.exists(TOKENIZER_FILE):
        print(f"[ERROR] Does not have TOKENIZER_FILE")
        sys.exit(0)
    
    tokenizer = Llama3Tokenizer("tokenizer.model")

    if "instruct" in MODEL_FILE:
        tokenizer = ChatFormat(tokenizer)
    print(f"Tokenizer loaded successfully.")


    # 3. Generate text
    start = time.time()

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_prompt, tokenizer).to(device),
        max_new_tokens=MAX_NEW_TOKENS,
        context_size=llama32_config["context_length"],
        top_k=TOP_K,
        temperature=TEMPERATURE
    )

    print(f"Generation time: {time.time() - start:.2f} sec")

    if torch.cuda.is_available():
        max_mem_bytes = torch.cuda.max_memory_allocated()
        max_mem_gb = max_mem_bytes / (1024 ** 3)
        print(f"Max memory allocated: {max_mem_gb:.2f} GB")

    output_text = token_ids_to_text(token_ids, tokenizer)

    if "instruct" in MODEL_FILE:
        output_text = clean_text(output_text)

    print("\nOutput text:\n", output_text)
