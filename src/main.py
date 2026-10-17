import subprocess

from typing import Annotated
from typing_extensions import TypedDict

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

@tool
def run_command(command: str) -> str:
    """Runs a Windows terminal command and returns the output."""
    if input(f"\n[⚠️ Allow command: {command}] (y/n): ").lower() != 'y':
        return "Command aborted."
    try:
        return subprocess.run(command, shell=True, capture_output=True, text=True).stdout
    except Exception as e:
        return str(e)

class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatOllama(model="llama3.1:8b", temperature=0).bind_tools([run_command])

def agent_node(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode([run_command]))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)
graph.add_edge("tools", "agent")
jarvis = graph.compile()

if __name__ == "__main__":
    print("🤖 Jarvis Online.")
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["quit", "exit"]: break
        
        for event in jarvis.stream({"messages": [("user", user_input)]}):
            if "agent" in event and event["agent"]["messages"][-1].content:
                print(f"Jarvis: {event['agent']['messages'][-1].content}")