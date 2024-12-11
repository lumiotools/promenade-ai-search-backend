import tiktoken


def calculate_token(messages, model="gpt-4o-mini"):
    encoding = tiktoken.encoding_for_model(model)
    total_tokens = 0
    for message in messages:
        tokens = len(encoding.encode(message)) + 4
        total_tokens += tokens

    return total_tokens
