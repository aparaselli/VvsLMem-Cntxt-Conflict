import numpy as np

def make_inputs(
    processor,
    model_device,
    *,
    retrieved_context,
    user_query: str,
    entity_modality: str = "vision",   # "vision" | "text"
    mock_RAG: bool = True,
    bench_type: str = "people",        # "people" | "logo"
    image=None,                        # PIL.Image when entity_modality="vision"
    instance_name: str = None,         # required when entity_modality="text"
    CoT: bool = False,
    add_mcq_prefix: bool = True,
    padding: bool = True,
):
    """
    Generalized inputs builder that mirrors behavioral script formatting
    (assuming ctxt_type="text" always; i.e., no ctxt_image).

    Returns:
        inputs: dict of tensors on model_device, plus inputs["_prompt_text"].
    """
    # COT_JSON_INSTRUCTIONS = (
    #     "\n\nReason through this question by:\n"
    #     "1. Identify the entity in the image\n"
    #     "2. Reason about the entity and background information\n"
    #     "3. Derive the answer to the question\n\n"
    #     "Required JSON fields:\n"
    #     "- entity: A string containing the name of the entity in 1-5 words\n"
    #     "- reasoning: A string containing 3-4 sentences explaining your thought process\n"
    #     "- answer: A string containing the final answer in 1-3 words\n\n"
    #     "Example 1:\n"
    #     "Input: Entity: Eiffel Tower, Question: When was it completed?\n"
    #     "Output: {\n"
    #     "\"entity\": \"The Eiffel Tower\",\n"
    #     "\"reasoning\": \"This is an image of the Eiffel Tower in Paris, a wrought-iron lattice tower. "
    #     "It was built for the 1889 World's Fair and took about 2 years to construct, with completion in 1889.\",\n"
    #     "\"answer\": \"1889\"\n"
    #     "}\n\n"
    #     "Example 2:\n"
    #     "Input: Image of Albert Einstein, Question: What theory is he most famous for?\n"
    #     "Output: {\n"
    #     "\"entity\": \"Albert Einstein\",\n"
    #     "\"reasoning\": \"This is an image of Albert Einstein, a renowned physicist known for his revolutionary "
    #     "contributions to physics. His most groundbreaking work was the theory of general relativity, which he "
    #     "published in 1915 and fundamentally changed our understanding of space and time.\",\n"
    #     "\"answer\": \"Theory of Relativity\"\n"
    #     "}\n\n"
    #     "PROVIDE YOUR REASONING OF 3-4 SENTENCES AND FINAL ANSWER OF 1-3 WORDS IN THE FOLLOWING FORMAT "
    #     "AND PROVIDE NO OTHER TEXT:\n"
    #     "{\n"
    #     "\"entity\": \"<Entity name>\",\n"
    #     "\"reasoning\": \"<your reasoning>\",\n"
    #     "\"answer\": \"<your answer>\"\n"
    #     "}\n"
    # )

    COT_JSON_INSTRUCTIONS = (
        "\n\nReason through this question by:\n"
        "1. Identify the entity in the image\n"
        "2. Derive the answer to the question\n\n"
        "Required JSON fields:\n"
        "- entity: A string containing the name of the entity in 1-5 words\n"
        "- answer: A string containing the final answer in 1-3 words\n\n"
        "Example 1:\n"
        "Input: Entity: Eiffel Tower, Question: When was it completed?\n"
        "Output: {\n"
        "\"entity\": \"The Eiffel Tower\",\n"
        "\"answer\": \"1889\"\n"
        "}\n\n"
        "Example 2:\n"
        "Input: Image of Albert Einstein, Question: What theory is he most famous for?\n"
        "Output: {\n"
        "\"entity\": \"Albert Einstein\",\n"
        "\"answer\": \"Theory of Relativity\"\n"
        "}\n\n"
        "PROVIDE THE ENTITY NAME IN 1-5 WORDS AND FINAL ANSWER OF 1-3 WORDS IN THE FOLLOWING FORMAT "
        "AND PROVIDE NO OTHER TEXT:\n"
        "{\n"
        "\"entity\": \"<Entity name>\",\n"
        "\"answer\": \"<your answer>\"\n"
        "}\n"
    )

    COT_JSON_INSTRUCTIONS_LANG = (
        "\n\nReason through this question by:\n"
        "1. Determine the entity referenced in the question\n"
        "2. Reason about the entity and background information\n"
        "3. Derive the answer to the question\n\n"
        "Required JSON fields:\n"
        "- entity: A string containing the name of the entity in 1-5 words\n"
        "- reasoning: A string containing 3-4 sentences explaining your thought process\n"
        "- answer: A string containing the final answer in 1-3 words\n\n"
        "Example 1:\n"
        "Input: Entity: Eiffel Tower, Question: When was it completed?\n"
        "Output: {\n"
        "\"entity\": \"The Eiffel Tower\",\n"
        "\"reasoning\": \"This is an image of the Eiffel Tower in Paris, a wrought-iron lattice tower. "
        "It was built for the 1889 World's Fair and took about 2 years to construct, with completion in 1889.\",\n"
        "\"answer\": \"1889\"\n"
        "}\n\n"
        "Example 2:\n"
        "Input: Image of Albert Einstein, Question: What theory is he most famous for?\n"
        "Output: {\n"
        "\"entity\": \"Albert Einstein\",\n"
        "\"reasoning\": \"This is an image of Albert Einstein, a renowned physicist known for his revolutionary "
        "contributions to physics. His most groundbreaking work was the theory of general relativity, which he "
        "published in 1915 and fundamentally changed our understanding of space and time.\",\n"
        "\"answer\": \"Theory of Relativity\"\n"
        "}\n\n"
        "PROVIDE YOUR REASONING OF 3-4 SENTENCES AND FINAL ANSWER OF 1-3 WORDS IN THE FOLLOWING FORMAT "
        "AND PROVIDE NO OTHER TEXT:\n"
        "{\n"
        "\"entity\": \"<Entity name>\",\n"
        "\"reasoning\": \"<your reasoning>\",\n"
        "\"answer\": \"<your answer>\"\n"
        "}\n"
    )

    assert entity_modality in {"vision", "text"}
    assert bench_type in {"people", "logo"}
    name_lc = ""
    if getattr(processor, "tokenizer", None) is not None:
        name_lc = str(getattr(processor.tokenizer, "name_or_path", "")).lower()
    is_gemma = ("gemma-3" in name_lc) or ("gemma3" in name_lc)


    def image_block():
        return {"type": "image", "image": image} if is_gemma else {"type": "image", "image": image}

    if not CoT:
        user_query = user_query + ". Answer with only 1-3 word(s)."
    # --- mock_RAG controls whether context is injected ---
    if not mock_RAG:
        retrieved_context = None

    # --- normalize query exactly like run_experiment for people ---
    # (behavioral does this before calling rag_model_call)
    if bench_type == "people" and user_query is not None:
        user_query = user_query.replace("the person in the picture", "the entity")
        user_query = user_query.replace("The person pictured", "the entity")

    # --- construct entity_name for text modality, matching behavioral ---
    entity_name = None
    if entity_modality == "text":
        if instance_name is None:
            raise ValueError("instance_name is required when entity_modality='text'")
        if bench_type == "logo":
            entity_name = f"The logo of the company known as {instance_name}"
        else:
            entity_name = instance_name
    else:
        # vision modality
        if image is None:
            raise ValueError("image is required when entity_modality='vision'")

    # --- Build messages exactly like rag_model_call (ctxt_image always None here) ---
    if retrieved_context is None or (isinstance(retrieved_context, float) and np.isnan(retrieved_context)):
        # no-RAG template
        text_before = (
            "Given your knowledge, answer the question about the following entity.\n"
            "Entity: "
        )

        if entity_modality == "text":
            # text-only path: include entity and query in one text block (matches rag_model_call entity_name branch)
            content_list = [{"type": "text", "text": f"{text_before}{entity_name}."}]
            content_list.append({"type": "text", "text": f"\nQuery: {user_query}"})
            if CoT:
                content_list.append({"type": "text", "text": COT_JSON_INSTRUCTIONS_LANG})
            messages = [{"role": "user", "content": content_list}]
            all_images = []
        else:
            # vision entity: text + image + query blocks
            content_list = [{"type": "text", "text": text_before}]
            #content_list.append({"type": "image", "image": image})
            content_list.append(image_block())
            content_list.append({"type": "text", "text": f"\nQuery: {user_query}"})
            if CoT:
                content_list.append({"type": "text", "text": COT_JSON_INSTRUCTIONS})
            messages = [{"role": "user", "content": content_list}]
            all_images = [image]

    else:
        # with-RAG template
        if entity_modality == "text":
            text_before = (
                "Context information is below.\n"
                "---------------------\n"
                f"{retrieved_context}\n"
                "---------------------\n"
                "Given the context information and your knowledge, answer the question about the following entity.\n"
                f"Entity: {entity_name}."
            )

            content_list = [{"type": "text", "text": text_before}]
            content_list.append({"type": "text", "text": f"\nQuery: {user_query}"})

            if CoT:
                content_list.append({"type": "text", "text": COT_JSON_INSTRUCTIONS_LANG})

            messages = [{"role": "user", "content": content_list}]
            all_images = []
        else:
            # behavioral vision path: context text block ending with "Entity: " then image then query text
            text_before = (
                "Context information is below.\n"
                "---------------------\n"
                f"{retrieved_context}\n"
                "---------------------\n"
                "Given the context information and your knowledge, answer the question about the following entity.\n"
                "Entity: "
            )
            content_list = [{"type": "text", "text": text_before}]
            #content_list.append({"type": "image", "image": image})
            content_list.append(image_block())
            content_list.append({"type": "text", "text": f"\nQuery: {user_query}"})
            if CoT:
                content_list.append({"type": "text", "text": COT_JSON_INSTRUCTIONS})
            messages = [{"role": "user", "content": content_list}]
            all_images = [image]

    if is_gemma:
        # Ensure system message exists (Gemma expects/likes it)
        if messages[0]["role"] != "system":
            messages = [{"role": "system", "content": [{"type":"text","text":"You are a helpful assistant."}]}] + messages

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = inputs.to(model_device)
        inputs["_prompt_text"] = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        return inputs
    # --- Apply chat template ---
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    proc_kwargs = dict(
        text=[text],
        return_tensors="pt",
        padding=padding,
    )

    if all_images:
        proc_kwargs["images"] = all_images  # list[PIL.Image]
    # --- Append EXACT behavioral prefix ---
    #if add_mcq_prefix:
        #text += "Between A, B, C, and D, the answer is the letter"

    # --- Tokenize/process ---
    image_inputs = {"images": all_images, "videos": None} if all_images else {}


    # move tensors to device
    inputs = processor(**proc_kwargs)
    inputs = inputs.to(model_device)
    inputs["_prompt_text"] = text
    return inputs
