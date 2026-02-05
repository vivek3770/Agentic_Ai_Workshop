import os
import google.genai as genai
from dotenv import load_dotenv


load_dotenv()
client = genai.Client(api_key="AIzaSyD3noCvZm_uxL-bLWYug4t7zFIHOkgyBYA")


Model = "gemini-2.5-flash"

# Menu items
menu = [
    {"item": "Veg thali", "price": 150, "type": "veg"},
    {"item": "Non-Veg thali", "price": 180, "type": "non-veg"},
    {"item": "Veg Biryani", "price": 120, "type": "veg"},
    {"item": "Non-Veg Biryani", "price": 160, "type": "non-veg"},
]

# Memory state
memory = {
    "budget": 150,
    "preference": "veg",
    "attempted_item": [],
    "goal_achieved": False,
    "goal": "Buy a suitable lunch item within budget and preference"
}


def get_menu():
    """Return the menu list."""
    return menu


def buy_item(item):
    """Simulate buying an item."""
    return f"You have purchased: {item['item']} for Rs. {item['price']}"


def ai_planner(memory, menu):
    """Ask AI to choose the next best item."""
    prompt = f"""
You are an AI planning agent.
Goal: {memory['goal']}

CONSTRAINTS:
- Budget: Rs. {memory['budget']}
- Preference: {memory['preference']}
- Do NOT repeat failed items
FAILED ITEMS: {memory['attempted_item']}

MENU: {menu}

TASK:
Choose the Next Best item to try.

RULES:
- Respond with only the item name
- No explanations
- Choose intelligently based on constraints and failed items
"""
    response = client.models.generate_content(
        model=Model,
        contents=prompt
    )
    return response.text.strip()


def lunch_agent():
    """Main loop for AI lunch planner."""
    print("Welcome to the AI Lunch Planner!")
    while not memory["goal_achieved"]:
        choice = ai_planner(memory, get_menu())
        print(f"AI suggests: {choice}")

        # Find item in menu
        item = next((i for i in menu if i["item"].lower() == choice.lower()), None)

        if not item:
            print("Item not found in menu. Try again.")
            memory["attempted_item"].append(choice)
            continue

        # Check constraints
        if item["price"] <= memory["budget"] and item["type"] == memory["preference"]:
            print(buy_item(item))
            memory["goal_achieved"] = True
        else:
            print(f"Cannot buy {item['item']} (fails budget/preference).")
            memory["attempted_item"].append(item["item"])

    print("Lunch planning complete!")


# Run the agent
if __name__ == "__main__":
    lunch_agent()
