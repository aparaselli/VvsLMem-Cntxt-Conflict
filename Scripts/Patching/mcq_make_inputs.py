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
    add_mcq_prefix: bool = True,
    padding: bool = True,
):
    """
    Generalized inputs builder that mirrors behavioral script formatting
    (assuming ctxt_type="text" always; i.e., no ctxt_image).

    Returns:
        inputs: dict of tensors on model_device, plus inputs["_prompt_text"].
    """

    assert entity_modality in {"vision", "text"}
    assert bench_type in {"people", "logo"}
    user_query = user_query + ". Answer with only one letter (A, B, C, or D)."
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
            "Given your knowledge, answer the multiple choice question about the following entity.\n"
            "Entity: "
        )

        if entity_modality == "text":
            # text-only path: include entity and query in one text block (matches rag_model_call entity_name branch)
            prompt_text = (
                f"{text_before}{entity_name}.\n"
                f"Query: {user_query}"
            )
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
            all_images = []
        else:
            # vision entity: text + image + query blocks
            content_list = [{"type": "text", "text": text_before}]
            content_list.append({"type": "image", "image": image})
            content_list.append({"type": "text", "text": f"\nQuery: {user_query}"})
            messages = [{"role": "user", "content": content_list}]
            all_images = [image]

    else:
        # with-RAG template
        if entity_modality == "text":
            # behavioral uses a single text block when entity_name is provided
            prompt_text = (
                "Context information is below.\n"
                "---------------------\n"
                f"{retrieved_context}\n"
                "---------------------\n"
                "Given the context information and your knowledge, answer the multiple choice question about the following entity.\n"
                f"Entity: {entity_name}.\n"
                f"Query: {user_query}"
            )
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
            all_images = []
        else:
            # behavioral vision path: context text block ending with "Entity: " then image then query text
            text_before = (
                "Context information is below.\n"
                "---------------------\n"
                f"{retrieved_context}\n"
                "---------------------\n"
                "Given the context information and your knowledge, answer the multiple choice question about the following entity.\n"
                "Entity: "
            )
            content_list = [{"type": "text", "text": text_before}]
            content_list.append({"type": "image", "image": image})
            content_list.append({"type": "text", "text": f"\nQuery: {user_query}"})
            messages = [{"role": "user", "content": content_list}]
            all_images = [image]

    # --- Apply chat template ---
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # --- Append EXACT behavioral prefix ---
    #if add_mcq_prefix:
        #text += "Between A, B, C, and D, the answer is the letter"

    # --- Tokenize/process ---
    image_inputs = {"images": all_images, "videos": None} if all_images else {}
    inputs = processor(
        text=[text],
        padding=padding,
        return_tensors="pt",
        **image_inputs
    )

    # move tensors to device
    inputs = inputs.to(model_device)
    inputs["_prompt_text"] = text
    return inputs
