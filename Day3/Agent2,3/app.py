import os
import google.genai as genai

client = genai.Client(
    api_key="GENAI_API_KEY"
)

Model = "gemini-2.5-flash"
menu = [
    {"item":"Veg thali","price":150, "type":"veg"},
    {"item":"Non-Veg thali","price":180, "type":"non-veg" },
    {"item":"Veg Biryani","price":120, "type":"veg" },
    {"item":"Non-Veg Biryani","price":160, "type":"non-veg" },
]


def get_menu():
    return menu


def buy_item(item):
    return f"You have purchased: {item['item']} for Rs. {item['price']}"



memory = {
    "budget":150,
    "preference": "veg",
    "attempted_item":[],
    "goal_achieved": False
}

def ai_planner(memory,menu):
    promt = f"""
You are an Ai planning agent.
Goal: {memory['goal']}
CONSTRAINTS:
- Budget: Rs. {memory['budget']}
- Preference: {memory['preference']}
- Do NOT repeat failed item
FAILED ITEMS: {memory[attemted_items]}

MENU: {menu}

TASK:
Choose the Next Best item to try.menu.

RULES:
- Respond with only the item name
- No explanations
- Choose intelligently based on constraint and failed items
"""
    response = client.models.generate_content(
        model= Model,
        contents= promt

    )

    return response.text.strip()

def lunch_agent():
    print("welcome to the Ai lunch Planner!")
    while not memory["goal_achieved"]:
        choice = ai_planner(memory,get_menu())
        print(f"AI suggests: {choice}")
        item = next((i for i in menu if i["item"] == choice),None)
        if not item:
            print("item not found in menu . Try again")
            memory["attempted_item"].append(choice)
            break
        if item["price"] <= memory["budget"] and item["type"] == memory["preference"]:
            print(buy_item(item))
            memory["goal_achieved"] = True
        
    else:
        print(f"Cannot buy {item['item']}".tr)




