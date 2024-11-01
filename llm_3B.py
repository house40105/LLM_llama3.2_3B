from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# Define the path to the locally stored model directory
model_path = "/home/house/llama-models/models/Llama3.2-3B-Instruct-hf"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("Tokenizer loaded successfully.")

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    return_dict=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
print("Model initialized successfully.")


# Define the message with a system message and a user prompt
# You can freely adjust the content based on your needs
messages = [
    {
        "role": "system",
        "content": "Answer questions in English.",
    },
    {
        "role": "user",
        "content": "What is Python?",
    },
]
# Concatenate the messages to form the prompt
# Only the 'content' fields of each message are joined to create a single input prompt
prompt = "".join([msg["content"] for msg in messages])


# Tokenize the prompt to prepare it for model input
# The tokenizer converts the text into input tokens and returns a dictionary of tensors
inputs = tokenizer(prompt, return_tensors="pt")

# Generate the output sequence from the model
# Here, we specify the 'pad_token_id' to ensure proper padding if needed
# 'max_new_tokens' sets the maximum number of tokens the model can generate
outputs = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id, max_new_tokens=300)

# Decode the generated tokens to convert them back to readable text
# 'skip_special_tokens' removes any special tokens used for formatting or marking sections
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n\nOutput text:", output_text)
