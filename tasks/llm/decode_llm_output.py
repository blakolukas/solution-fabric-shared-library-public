from core.task import task


@task(outputs=["decoded_text"])
def decode_llm_output(output_ids, tokenizer):
    """
    Decode LLM output token IDs back to text.

    Args:
        output_ids: Generated token IDs from LLM
        tokenizer: HuggingFace tokenizer instance

    Returns:
        Decoded text string
    """
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return decoded
