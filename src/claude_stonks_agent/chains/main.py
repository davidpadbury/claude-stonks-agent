from claude_stonks_agent.claude import create_llm, build_tools_description
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_core.tools import StructuredTool
from langchain_core.runnables import RunnablePassthrough, RunnableAssign, RunnableBinding
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from textwrap import dedent

system_message = SystemMessagePromptTemplate.from_template(
    dedent('''
    You are a helpful agent that helps answer questions about the stock market.

    In this environment you have access to a set of tools you can use to answer the user's question.
    
    Rules:
     - You must always use tools to determine the current date if answering a question related to the current date.
     - If the search_for_symbol tool returns many results, ask the user to clarify which one they meant.

    You may call tools like this:
    <function_calls>
    <invoke>
    <tool_name>$TOOL_NAME</tool_name>
    <parameters>
    <$PARAMETER_NAME>$PARAMETER_VALUE</$PARAMETER_NAME>
    ...
    </parameters>
    </invoke>
    ...
    </function_calls>

    Here are the tools available:
    {tools}
    ''').strip()
)

prompt = ChatPromptTemplate.from_messages([
    system_message,
    MessagesPlaceholder('messages'),
])


def fix_prompt_for_claude(params: dict):
    return params


def create_chain(tools: list[StructuredTool]):
    llm = create_llm()

    return (
        RunnablePassthrough.assign(tools=lambda _: build_tools_description(tools))
        | prompt
        | RunnableLambda(fix_prompt_for_claude)
        | llm
    )
