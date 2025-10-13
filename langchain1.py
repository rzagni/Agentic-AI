from typing import TypedDict, Annotated, Optional, Dict, Any, List

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages


# ---------- 1) Define state ----------
class State(TypedDict):
    order: Optional[Dict[str, Any]]
    messages: Annotated[List[Any], add_messages]  # list of BaseMessage


# ---------- 2) Define our single business tool ----------
@tool
def cancel_order(order_id: str) -> str:
    """Cancel an order that hasn't shipped."""
    # (Here you'd call your real backend API)
    return f"Order {order_id} has been cancelled."


# ---------- 3) The agent "brain": invoke LLM, run tool, then invoke LLM again ----------
# Prepare LLMs (bind tools for the first pass)

llm_with_tools = base_llm.bind_tools([cancel_order])

def call_model(state: State) -> dict:
    msgs = state["messages"]
    order = state.get("order") or {"order_id": "UNKNOWN"}

    # System prompt
    prompt = (
        "You are an ecommerce support agent.\n"
        f"ORDER ID: {order['order_id']}\n"
        "If the customer asks to cancel, call cancel_order(order_id) and then send a simple confirmation.\n"
        "Otherwise, just respond normally.\n"
    )

    conversation = [SystemMessage(content=prompt)] + msgs

    # ---- 1st LLM pass (tool decision) ----
    first: AIMessage = llm_with_tools.invoke(conversation)
    outputs = [first]

    # If the model called a tool, run it and do a 2nd pass
    if getattr(first, "tool_calls", None):
        tc = first.tool_calls[0]  # assume at most one here
        # tc has keys: id, name, args
        result = cancel_order.invoke(tc["args"])  # or cancel_order(**tc["args"])

        tool_msg = ToolMessage(
            content=result,
            tool_call_id=tc["id"],
            name=tc["name"],
        )
        outputs.append(tool_msg)

        # ---- 2nd LLM pass (final answer) ----
        second: AIMessage = base_llm.invoke(conversation + outputs)
        outputs.append(second)

    # Return messages to be merged into graph state
    return {"messages": outputs}


# ---------- 4) Wire it all up in a StateGraph ----------
def construct_graph():
    g = StateGraph(State)
    g.add_node("assistant", call_model)
    g.set_entry_point("assistant")
    g.add_edge("assistant", END)  # important!
    return g.compile()

graph = construct_graph()


# ---------- 5) Example run ----------
if __name__ == "__main__":
    example_order = {"order_id": "A12345"}
    convo = [HumanMessage(content="Please cancel my order A12345.")]
    result = graph.invoke({"order": example_order, "messages": convo})
    for msg in result["messages"]:
        # msg could be HumanMessage/AIMessage/ToolMessage; guard for attribute names
        role = getattr(msg, "type", msg.__class__.__name__)
        print(f"{role}: {msg.content}")
