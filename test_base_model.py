#!/usr/bin/env python3
"""
Test script for Qwen2.5-VL-7B base model
Simple image Q&A test to verify model functionality
"""

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import os

# Configuration
MODEL_NAME = "./models/Qwen2.5-VL-7B-Instruct"  # Local model path
TEST_IMAGE_PATH = "./data/images/test/high_quality"  # Path to test images
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    """Load the Qwen2.5-VL model and processor"""
    print(f"Loading model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")

    # Load processor
    processor = AutoProcessor.from_pretrained(MODEL_NAME, use_fast=False)
    print("Processor loaded successfully")

    # Load model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto"
    )

    print("Model loaded successfully!")
    return model, processor

def test_image_qa(model, processor, image_path, question):
    """Test image Q&A with the model"""
    # Load image
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None

    image = Image.open(image_path).convert("RGB")
    print(f"Testing image: {image_path}")
    print(f"Question: {question}")

    # Prepare conversation
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image", "image": image_path},
            ],
        }
    ]

    # Prepare inputs
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], images=[image], padding=True, return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # Generate response
    print("Generating response...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.1,
        )

    # Decode output
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    response = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]

    print(f"Response: {response}")
    print("-" * 50)
    return response

def main():
    """Main test function"""
    print("=" * 60)
    print("Qwen2.5-VL-7B Base Model Test")
    print("=" * 60)

    try:
        # Load model
        model, processor = load_model()

        # Test questions
        test_questions = [
            "What do you see in this image?",
            "Describe the main features of this medical image.",
            "What type of medical scan is this?",
            "Are there any abnormalities visible in this image?"
        ]

        # Find test images
        if os.path.exists(TEST_IMAGE_PATH):
            image_files = [f for f in os.listdir(TEST_IMAGE_PATH)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            if image_files:
                # Test with first available image
                test_image = os.path.join(TEST_IMAGE_PATH, image_files[0])

                print(f"\nTesting with spine X-ray: {image_files[0]}")
                print("=" * 60)

                # Run tests
                for i, question in enumerate(test_questions, 1):
                    print(f"\nTest {i}:")
                    test_image_qa(model, processor, test_image, question)

            else:
                print(f"No image files found in {TEST_IMAGE_PATH}")

        else:
            print(f"Images directory not found: {TEST_IMAGE_PATH}")

    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\nTest completed!")

if __name__ == "__main__":
    main()