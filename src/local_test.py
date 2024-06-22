import json
import predict as infer  # Ensure this matches the module name where your Predictor class is defined

# Load the test input
with open("test_input.json", "r") as f:
    test_input = json.load(f)[0]  # Assuming the JSON structure as an array of inputs

# Initialize the model
model = infer.Predictor()
model.setup()

# Run the inference
result = model.predict(
    prompt=test_input["input"]["prompt"],
    negative_prompt=test_input["input"].get("negative_prompt"),
    width=test_input["input"]["width"],
    height=test_input["input"]["height"],
    init_image=test_input["input"]["init_image"],
    mask=test_input["input"]["mask"],
    prompt_strength=test_input["input"]["prompt_strength"],
    num_outputs=test_input["input"]["num_outputs"],
    num_inference_steps=test_input["input"]["num_inference_steps"],
    guidance_scale=test_input["input"]["guidance_scale"],
    scheduler=test_input["input"]["scheduler"],
    seed=None,  # Use None to generate a random seed
    lora=test_input["input"]["lora_model_path"],
    lora_scale=test_input["input"]["lora_strength"]
)

# Print the result
print(result)
