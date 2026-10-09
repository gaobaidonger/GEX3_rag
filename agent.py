
"""University IT support RAG agent using Qwen3-0.6B."""

import os
import re
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from rag_tool import search_university_it_policies


DEFAULT_MODEL = os.environ.get(
    "QWEN_MODEL",
    "../GEX3/models/qwen3-0.6b"
)


def answer(question, model_path=DEFAULT_MODEL):
    """Answer using evidence retrieved from university IT policies."""

    # Step 1: Retrieve relevant policy information
    evidence = search_university_it_policies.invoke({
        "question": question
    })

    evidence = str(evidence).strip()

    # Step 2: Do not invent missing IT Service Desk phone numbers
    question_lower = question.lower()

    asks_phone = bool(
        re.search(
            r"phone number|telephone number|contact number|"
            r"电话号码|联系电话",
            question_lower
        )
    )

    asks_it_support = bool(
        re.search(
            r"it service desk|it support|helpdesk|"
            r"service desk|技术支持|信息中心",
            question_lower
        )
    )

    phone_numbers = re.findall(
        r"(?:\+?\d[\d\s().-]{6,}\d)",
        evidence
    )

    if asks_phone and asks_it_support and not phone_numbers:
        return (
            "I don't know based on the provided "
            "university IT policy documents."
        )

    # Step 3: Refuse unsupported questions
    if not evidence or evidence.lower() in [
        "none",
        "no relevant information found.",
        "no relevant policy found."
    ]:
        return (
            "I don't know based on the provided "
            "university IT policy documents."
        )

    # Step 4: Select Apple GPU when available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Step 5: Load Qwen3-0.6B
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.float32
    )

    model.to(device)
    model.eval()

    # Step 6: Construct grounded RAG prompt
    system_prompt = (
        "You are a university IT support assistant.\n"
        "Answer questions using ONLY the policy context provided.\n"
        "Do not invent facts, procedures, contact information, "
        "or telephone numbers.\n"
        "If the policy does not contain the requested information, "
        "reply exactly:\n"
        "I don't know based on the provided university IT "
        "policy documents.\n"
        "Keep answers short and directly relevant."
    )

    user_prompt = (
        "University IT policy context:\n"
        f"{evidence}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Answer:"
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    # Step 7: Prepare inputs for Transformers 5.x
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False
    )

    inputs = inputs.to(device)

    # Step 8: Generate the model response
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Step 9: Decode generated tokens only
    input_length = inputs["input_ids"].shape[-1]

    result = tokenizer.decode(
        output[0][input_length:],
        skip_special_tokens=True
    ).strip()

    # Step 10: Avoid returning empty responses
    if not result:
        return (
            "I don't know based on the provided "
            "university IT policy documents."
        )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="University IT Policy RAG Assistant"
    )

    parser.add_argument(
        "question",
        nargs="?",
        default=(
            "What exact phone number should I call "
            "for the University IT Service Desk?"
        )
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Path to local Qwen3-0.6B model"
    )

    args = parser.parse_args()

    print("QUESTION:", args.question)
    print("ANSWER:", answer(args.question, args.model))


if __name__ == "__main__":
    main()
