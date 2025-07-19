import os
import json
import time
import yaml
import tiktoken
from openai import OpenAI
from .gguf_utils import generate_gguf_response, get_gguf_model_info


def count_tokens(text):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # Using a default model
    return len(encoding.encode(text))


def check_file_exists(save_file, total_len):
    if os.path.exists(
        save_file,
    ):
        with open(
            save_file,
            "r",
        ) as infile:
            already_saved_data = json.load(infile)
        if len(already_saved_data) == total_len:
            print(f"Already saved enough data of {total_len}, Skipping evaluation.")
            return True
        else:
            print("only have ", len(already_saved_data))
            return False
    return False


def print_conversation(messages):
    for message in messages:
        role = message["role"]
        content = message["content"]
        print(f"{role.capitalize()}: {content}\n")
        print()


def generate_message(
    bedrock_runtime,
    model_id,
    model_type,
    system_prompt=None,
    messages=None,
    max_tokens=None,
    temperature=0,
    max_retries=10,
    gguf_models_config=None,
):
    retries = 0
    while retries < max_retries:
        try:
            if model_type == "claude":
                body = json.dumps(
                    {
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": max_tokens,
                        "system": system_prompt,
                        "messages": messages,
                        "temperature": temperature,
                    }
                )
                response = bedrock_runtime.invoke_model(body=body, modelId=model_id)
                response_body = json.loads(response.get("body").read())
                return response_body["content"][0]["text"]

            elif model_type == "mistral":
                native_request = {
                    "prompt": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                request = json.dumps(native_request)
                response = bedrock_runtime.invoke_model(modelId=model_id, body=request)
                model_response = json.loads(response["body"].read())
                outputs = model_response.get("outputs")
                response_text = outputs[0]["text"]
                return response_text

            elif model_type == "llama":
                native_request = {
                    "prompt": messages,
                    "max_gen_len": max_tokens,
                    "temperature": temperature,
                }
                request = json.dumps(native_request)
                response = bedrock_runtime.invoke_model(modelId=model_id, body=request)
                model_response = json.loads(response["body"].read())
                response_text = model_response["generation"]
                return response_text
            elif model_type == "gpt":
                client = OpenAI()  # Use the client passed or create a new one
                completion = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    # temperature=temperature,
                    # max_tokens=max_tokens,
                )
                return completion.choices[0].message.content
            elif model_type == "gemini":
                # Gemini support requires google.generativeai - skipping for GGUF testing
                raise ValueError("Gemini models not configured for this test")
            elif model_type == "gguf":
                if not gguf_models_config:
                    raise ValueError("GGUF models config required for GGUF model type")
                
                model_path, model_config = get_gguf_model_info(model_id, gguf_models_config)
                
                # Add system prompt to messages if provided
                if system_prompt and messages:
                    if isinstance(messages, list) and len(messages) > 0:
                        # Insert system message at the beginning
                        messages = [{"role": "system", "content": system_prompt}] + messages
                    elif isinstance(messages, str):
                        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": messages}]
                
                response_text = generate_gguf_response(
                    model_path=model_path,
                    model_config=model_config,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    max_retries=max_retries,
                )
                return response_text
            else:
                raise ValueError(f"Invalid model_type: {model_type}")

        except Exception as e:
            print(e, "retrying time:", retries, model_type)
            if "reduce" in str(e):
                raise Exception(f"max context length is exceeded")
            if retries == max_retries - 1:
                time.sleep(20)
                print("sleeping 20 seconds")
                retries = 0
            retries += 1
            time.sleep(5)  # Wait for 10 seconds before retrying


def extract_multi_turn_conversation(multi_turn_message, turn_number=3, model_type="llama"):
    message = []
    for turn in multi_turn_message:
        role = turn["role"]
        content = turn["content"]
        if model_type == "llama":
            message.append(f"<|start_header_id|>{role}<|end_header_id|>\n{content}<|eot_id|>")
        elif model_type == "claude":
            message.append({"role": role, "content": content})
        elif model_type == "mistral":
            if role == "user":
                message.append(f"[INST] {content} [/INST]")
            else:
                message.append(f"{content}</s>")
        elif model_type == "gpt":
            message.append({"role": role, "content": content})
        elif model_type == "gemini":
            gemini_role = {"user": "user", "assistant": "model"}.get(role, "user")
            message.append({"role": gemini_role, "parts": [{"text": str(content)}]})
        elif model_type == "gguf":
            message.append({"role": role, "content": content})
        else:
            raise ValueError(f"Invalid model_type: {model_type}")
        if len(message) == turn_number * 2:
            if role != "assistant":
                raise ValueError("The last turn must be from assistant")
            break
    assert len(message) == turn_number * 2, "The number of turns is less than the specified number"
    if "llama" in model_type or "mistral" in model_type:
        message = "".join(message)
    return message


def load_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    # Get the current working directory and remove the last directory in the path
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)  # This removes the last directory

    config["dir_path"] = parent_dir
    return config


def extract_multi_turn_message(turns_data, args, model_type):
    if args.inter_turns > 0:
        multi_turn_message = []
        for turn_data in turns_data:
            multi_turn_message.extend(turn_data["conversation"])
        return (
            extract_multi_turn_conversation(multi_turn_message, args.inter_turns, model_type=model_type),
            multi_turn_message,
        )
    else:
        multi_turn_message = None
    return "", multi_turn_message


def get_model_info(model_name, gguf_models_config=None):
    # Check if this is a GGUF model first
    if gguf_models_config and model_name in gguf_models_config:
        return model_name, "gguf"
    
    if model_name == "claude3s":
        model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    elif model_name == "claude3.5s":
        model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    elif model_name == "llama38b":
        model_id = "meta.llama3-8b-instruct-v1:0"
    elif model_name == "claude2":
        model_id = "anthropic.claude-v2:1"
    elif model_name == "claude3hk":
        model_id = "anthropic.claude-3-haiku-20240307-v1:0"
    elif model_name == "llama38b":
        model_id = "meta.llama3-8b-instruct-v1:0"
    elif model_name == "llama370b":
        model_id = "meta.llama3-70b-instruct-v1:0"
    elif model_name == "llama318b":
        model_id = "meta.llama3-1-8b-instruct-v1:0"
    elif model_name == "llama3170b":
        model_id = "meta.llama3-1-70b-instruct-v1:0"
    elif model_name == "llama31405b":
        model_id = "meta.llama3-1-405b-instruct-v1:0"
    elif model_name == "mistral7b":
        model_id = "mistral.mistral-7b-instruct-v0:2"
    elif model_name == "mistral8x7b":
        model_id = "mistral.mixtral-8x7b-instruct-v0:1"
    elif model_name == "mistrallarge":
        model_id = "mistral.mistral-large-2402-v1:0"
    elif model_name == "mistrallarge2":
        model_id = "mistral.mistral-large-2407-v1:0"
    elif model_name == "gpt4o-mini":
        model_id = "gpt-4o-mini"
    elif model_name == "gpt-o1":
        model_id = "o1-preview-2024-09-12"
    elif model_name == "gpt-o3":
        model_id = "o3-2025-04-16"
    elif model_name == "gpt-o4-mini":
        model_id = "o4-mini-2025-04-16"
    elif model_name == "gemini2.5-flash":
        model_id = "gemini-2.5-flash-preview-04-17"
    elif model_name == "gemini2.5-pro":
        model_id = "gemini-2.5-pro-exp-03-25"
    else:
        model_id = model_name  # fallback to model_name as ID

    # Fixed model_type detection to include gemini and gguf
    model_type = (
        "mistral"
        if "mistral" in model_name
        else (
            "claude"
            if "claude" in model_name
            else (
                "llama"
                if "llama" in model_name
                else (
                    "gpt" 
                    if "gpt" in model_name 
                    else (
                        "gemini" 
                        if "gemini" in model_name 
                        else (
                            "gguf"
                            if gguf_models_config and model_name in gguf_models_config
                            else None
                        )
                    )
                )
            )
        )
    )
    return model_id, model_type


ALL_TOPICS = [
    "travel_transportation",
    "shop_motors",
    "lifestyle_beauty",
    "travel_restaurant",
    "shop_fashion",
    "entertain_shows",
    "pet_ownership",
    "lifestyle_fit",
    "entertain_games",
    "shop_home",
    "lifestyle_health",
    "travel_activities",
    "education_learning_styles",
    "entertain_music_book",
    "professional_work_location_style",
    "education_resources",
    "lifestyle_dietary",
    "shop_technology",
    "travel_hotel",
    "entertain_sports",
]


COT_PROMPT = """
When answering a user's question, a good assistant should carefully consider the user's stated preferences and tailor the response accordingly.

Example 1:
User1's preference: "I have an intense phobia of bugs and insects. What outdoor activities would you recommend for the summer?"
Good assistant respnose: "Considering your phobia of bugs and insects, I'll focus on recommending activities that minimize encounters with insects. Let's explore outdoor options with fewer bugs, like beaches, pools, urban areas, well-maintained park trails, and activities during cooler times of day. I'll avoid suggesting things like camping, off-trail hiking, picnicking, or activities in dense forests. Here are some summer outdoor activities you might enjoy: spending a day at the beach swimming and playing sports, exploring outdoor malls or pedestrian areas in the city, hiking on maintained trails in parks (avoiding dense woods), attending evening outdoor concerts or movies, and enjoying water activities like kayaking or paddle boating on lakes or rivers. These options should help you enjoy the outdoors while minimizing encounters with insects."
Example 2:
User2's preference: "I'm a vegan looking for new dinner recipes. Any ideas?"
Good assistant response: "As a vegan, you'll need plant-based recipes without animal products. I'll focus on nutrient-rich, diverse ingredients to ensure balanced meals. Consider: quinoa and black bean burrito bowls, lentil and vegetable curry, roasted vegetable and hummus wrap, or zucchini noodles with avocado pesto. These options offer protein, fiber, and essential nutrients while adhering to vegan principles."
Example 3:
User3's preference: "I'm a beginner guitarist with small hands. What's a good first guitar?"
Good assistant response: "For a beginner with small hands, we'll prioritize comfort and playability. Look for guitars with narrow necks and shorter scale lengths. Consider: Fender Mustang, Ibanez Mikro, or Squier Mini Strat. These models are smaller overall, making them easier to handle. Avoid full-sized dreadnoughts or jumbo acoustics. Try these at a local music store to ensure they feel comfortable before purchasing."
Example 4:
User4's preference: "I'm on a tight budget but want to start a home gym. What equipment should I get?"
Good assistant response: "Working with a tight budget, we'll focus on versatile, cost-effective equipment for a home gym. Prioritize items that offer multiple exercise options. Consider: resistance bands, a set of adjustable dumbbells, a stability ball, and a yoga mat. These allow for a wide range of strength, flexibility, and cardio exercises without breaking the bank. Avoid expensive, single-purpose machines. As your budget allows, you can add more specialized equipment later."
Example 5:
User5's preference: "I'm sensitive to caffeine but need an energy boost in the mornings. Any suggestions?"
Good assistant response: "Given your caffeine sensitivity, we'll explore natural, caffeine-free energy boosters. Focus on lifestyle changes and alternatives to caffeinated drinks. Try: starting your day with a brisk walk or light exercise, drinking a glass of water with lemon, eating a protein-rich breakfast, or trying herbal teas like peppermint or ginger. Avoid coffee, energy drinks, and black tea. These alternatives can help increase alertness without the jitters or crashes associated with caffeine."

Now, please answer the following question while considering my preferences (not the user preferences in the examples above), which I have stated either explicitly or implicitly in our previous conversation:

            """
REMINDER = "\nIn your response, please ensure that you take into account our earlier discussion, and provide an answer that is consistent with my preference."
