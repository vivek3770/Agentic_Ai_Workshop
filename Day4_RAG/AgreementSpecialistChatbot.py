from __future__ import annotations

import os
from typing import List, Dict, Optional

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

try:
    from google import genai
    from google.genai import types
except Exception as exc:
    raise ImportError("The 'google-genai' package is required. Install with: pip install google-genai") from exc

from query_rewriter import QueryRewriter
from embedder import get_embedding
from weaviate_helper import search_near_vector


SYSTEM_PROMPT = (
    "You are Agreement Specialist, a helpful and friendly assistant for agreement-related queries between parties. "
    "Speak naturally and conversationally. Integrate any provided context seamlessly without saying phrases like 'based on the provided context' or 'the document says'. "
    "Answer clearly, be concise, and when helpful, cite concrete details (figures, clauses, timeframes) directly. "
    "If the context is insufficient, ask a focused clarifying question or state what is missing—without referencing retrieval mechanics."
)


class AgreementSpecialistChatbot:
    def __init__(self, llm_model: str = "gemini-2.5-flash") -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
             # Fallback
            api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) not set in environment")
        self._client = genai.Client(api_key=api_key)
        self.llm_model = llm_model
        # We keep the history in OpenAI format for compatibility with QueryRewriter
        self.history: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.rewriter = QueryRewriter(model=llm_model)

    def _build_messages(self, user_message: str, top_context: Optional[Dict] = None) -> List[Dict[str, str]]:
        msgs: List[Dict[str, str]] = list(self.history)
        if top_context:
            ctx_q = top_context.get("question") or ""
            ctx_a = top_context.get("answer") or ""
            context_block = f"Relevant context from FAQ (top match):\nQ: {ctx_q}\nA: {ctx_a}"

            msgs.append({"role": "system", "content": context_block})
        msgs.append({"role": "user", "content": user_message})
        return msgs

    def _to_gemini_contents(self, messages: List[Dict[str, str]]) -> Tuple[List[types.Content], str]:
        """Convert OpenAI-style messages to Gemini contents and extract system instruction."""
        gemini_msgs = []
        system_instruction = ""
        
        for m in messages:
            role = m.get("role")
            content = m.get("content") or ""
            
            if role == "system":
                # Concatenate system messages
                if system_instruction:
                    system_instruction += "\n" + content
                else:
                    system_instruction = content
            elif role == "user":
                gemini_msgs.append(types.Content(role="user", parts=[types.Part.from_text(text=content)]))
            elif role == "assistant":
                gemini_msgs.append(types.Content(role="model", parts=[types.Part.from_text(text=content)]))
                
        return gemini_msgs, system_instruction

    def ask(self, user_message: str) -> str:
        # Maintain conversation history
        self.history.append({"role": "user", "content": user_message})

        # Rewrite query using the recent conversation window
        rewritten_query = self.rewriter.rewrite(self.history)
        if not rewritten_query:
            rewritten_query = user_message
        print("rewritten query => ", rewritten_query)
        # Retrieve top-1 context from Weaviate
        query_vec = get_embedding(rewritten_query)
        hits = search_near_vector(query_vec, limit=3)
        print("hits from weaviate => ", hits)
        top_hit = hits[0] if hits else None
        print(top_hit)
        

        messages_for_turn = list(self.history[:-1]) # All except last user message
        
        # Add context if exists
        if top_hit:
            ctx_q = top_hit.get("question") or ""
            ctx_a = top_hit.get("answer") or ""
            context_block = f"Relevant context from FAQ (top match):\nQ: {ctx_q}\nA: {ctx_a}"
            messages_for_turn.append({"role": "system", "content": context_block})
            
        # Add the user message
        messages_for_turn.append(self.history[-1])

        gemini_contents, system_instr = self._to_gemini_contents(messages_for_turn)

        response = self._client.models.generate_content(
            model=self.llm_model,
            contents=gemini_contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instr,
                temperature=0.2,
                max_output_tokens=400,
            )
        )
        answer = (response.text or "").strip()
        self.history.append({"role": "assistant", "content": answer})
        return answer


def _repl() -> None:
    bot = AgreementSpecialistChatbot()
    print("Agreement Specialist Chatbot. Type 'exit' to quit.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break
        response = bot.ask(user_input)
        print(f"Assistant: {response}\n")


if __name__ == "__main__":
    _repl()

