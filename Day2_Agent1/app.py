from google import genai
from google.genai.types import (
    FunctionDeclaration,
    Tool,
    GenerateContentConfig,
    Part,
    Content, 
)

from dotenv import load_dotenv
import os

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Tool implementation
def calculator(a: float, b: float, operation: str):
    if operation == "add":
        return a + b
    if operation == "subtract":
        return a - b
    if operation == "multiply":
        return a * b
    if operation == "divide":
        if b == 0:
            raise ValueError("Division by zero")
        return a / b
    raise ValueError("Invalid operation")

#  Tool schema
calculator_fn = FunctionDeclaration(
    name="calculator",
    description="Perform basic arithmetic operations",
    parameters={
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"},
            "operation": {
                "type": "string",
                "enum": ["add", "subtract", "multiply", "divide"],
            },
        },
        "required": ["a", "b", "operation"],
    },
)

tool = Tool(function_declarations=[calculator_fn])

prompt = "Calculate the multiply of 15 and 30."

# Step 1: Ask Gemini
MODEL_NAME = "gemini-2.5-flash"

response = client.models.generate_content(
    model=MODEL_NAME,
    contents=prompt,
    config=GenerateContentConfig(tools=[tool]),
)

#  Step 2: Execute tool if requested
for part in response.candidates[0].content.parts:
    if part.function_call:
        print(f" Tool called: {part.function_call.name}")
        print(f" Arguments: {part.function_call.args}")
        
        args = part.function_call.args

        result = calculator(
            a=args["a"],
            b=args["b"],
            operation=args["operation"],
        )
        
        print(f" Tool result: {result}")

        # Step 3: Send tool result back WITH conversation history
        final_response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                Content(role="user", parts=[Part(text=prompt)]), 
                Content(role="model", parts=response.candidates[0].content.parts),
                Content(
                    role="user",
                    parts=[
                        Part.from_function_response(
                            name="calculator",
                            response={"result": result},
                        )
                    ],
                ),
            ],
            config=GenerateContentConfig(tools=[tool]),
        )

        print("\n Final Answer:")
        print(final_response.text)
        break
else:
    print("Model answered directly:")
    print(response.text)