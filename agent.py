print(">>> LOADING RESEARCH AGENT <<<")

import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from tools import web_search, fetch_page

load_dotenv()


@tool
def search_tool(query: str) -> str:
    """Search the web for a query."""
    results = web_search(query)

    if not results:
        return "No results found."

    return "\n".join(
        f"[{i+1}] {r['title']} - {r['url']}\n{r['snippet']}"
        for i, r in enumerate(results)
    )


@tool
def fetch_tool(url: str) -> str:
    """Fetch and return webpage text."""
    text = fetch_page(url)

    if not text:
        return "Could not fetch page."

    return text


tools = [search_tool, fetch_tool]
tool_map = {t.name: t for t in tools}

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY")
).bind_tools(tools)


MAX_STEPS = 6


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    steps: int


def agent_node(state: AgentState):
    response = llm.invoke(state["messages"])

    return {
        "messages": [response],
        "steps": state["steps"] + 1
    }


def tool_node(state: AgentState):
    last = state["messages"][-1]
    results = []

    for call in last.tool_calls:
        fn = tool_map[call["name"]]

        try:
            output = fn.invoke(call["args"])
        except Exception as e:
            output = f"Tool error: {e}"

        results.append(
            ToolMessage(
                content=output,
                tool_call_id=call["id"]
            )
        )

    return {"messages": results}


def should_continue(state: AgentState):
    if state["steps"] >= MAX_STEPS:
        return "end"

    last = state["messages"][-1]

    if hasattr(last, "tool_calls") and last.tool_calls:
        return "continue"

    return "end"


graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

graph.set_entry_point("agent")

graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

graph.add_edge("tools", "agent")

app = graph.compile()


def run(question: str):
    system = SystemMessage(
        content="""
You are a research assistant.

Use search_tool to search the web.
Use fetch_tool to read relevant webpages.

Research the question before answering.
Only use information supported by the sources.
Do not invent facts.
Cite factual claims using the source URL.

Format citations like:
[source: URL]
"""
    )

    state = {
        "messages": [
            system,
            HumanMessage(content=question)
        ],
        "steps": 0
    }

    final = app.invoke(state)

    return final["messages"][-1].content


if __name__ == "__main__":
    print("Research Agent ready.")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("Research Agent: Goodbye!")
            break

        if not question:
            continue

        try:
            answer = run(question)
            print("\nResearch Agent:", answer, "\n")
        except Exception as e:
            print(f"\nError: {e}\n")