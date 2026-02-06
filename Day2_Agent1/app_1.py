import google.generativeai as genai
genai.configure(api_key="AIzaSyD3noCvZm_uxL-bLWYug4t7zFIHOkgyBYA")


model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    generation_config={
        "max_output_tokens":220,
        "temperature":0.7
    }
)

#tool defination
calculator_tool = {
    "name": "calculator",
    "description": "A tool to perform basic arithmetic calculations",
    "parameters":{
        "type": "object",
        "properties": {
            "a" : {
                "type": "number",
                "description" : "the second number to be used un th calculation."
            },
            "b" : {
                "type": "number",
                "description" : "the second number to be used un th calculation."
            },

            "operation":{}
            }
    }

}

#Tool Implementation

def calculator(a,b,operation):
    if operation == "add":
        return a+b
    elif operation == "subst":
        return a-b
    elif operation == "mult":
        return a*b
    elif operation == "divide":
        return a/b


chat = model.start_chat(history=[
    {
        "role": "user",
        "parts": ["you are a helpful assistant in backend development"]
    }
])

print("chat started...........type 'exit' to stop")
while True:
    user = input("You: ")
    if user.lower() == "exit":
        break
    response = chat.send_message(user)
    print("Bot:", response.text)
