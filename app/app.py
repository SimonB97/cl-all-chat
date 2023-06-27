# Running the app:
# chainlit run app.py -w --no-cache  (-w for watch mode (auto-reload), --no-cache to disable cache)

import os
import langchain
from langchain import SerpAPIWrapper
from langchain.callbacks.manager import Callbacks
# from langchain.prompts import ChatPromptTemplate
from langchain.agents import Tool   # , AgentType, initialize_agent
from langchain.agents.agent import AgentExecutor
from langchain.agents.openai_functions_multi_agent.base import OpenAIMultiFunctionsAgent
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.chat_models import ChatOpenAI
# from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.tools.file_management import ReadFileTool, CopyFileTool, MoveFileTool, WriteFileTool, DeleteFileTool, FileSearchTool, ListDirectoryTool
from langchain.tools.shell import ShellTool
from langchain.tools.python.tool import PythonAstREPLTool
# from langchain.tools.human.tool import HumanInputRun
# from langchain.agents.tools import InvalidTool
from langchain.memory import ConversationTokenBufferMemory
#from langchain.memory.chat_message_histories import FileChatMessageHistory
from langchain.prompts import MessagesPlaceholder   # , HumanMessagePromptTemplate
# from langchain.vectorstores import FAISS
# from langchain.docstore import InMemoryDocstore
# from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import BaseMessage, AIMessage, AgentAction, AgentFinish   # , SystemMessage
from langchain.agents.openai_functions_multi_agent.base import _FunctionsAgentAction, _format_intermediate_steps
from typing import Union, List, Any, Tuple
from json import JSONDecodeError
import json
# import faiss
import chainlit as cl
import dotenv

# enable langchain tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.environ["LANGCHAIN_API_KEY"] = "9172786fbfbc4ab8ae1ca5dedbe717c9"
os.environ["LANGCHAIN_SESSION"] = "chainlit-app"

# ------- Prerequisites ------- #
dotenv.load_dotenv()
langchain.debug = True
langchain.verbose = True
llms = {
    "0" : "gpt-4-0613", 
    "1" : "gpt-3.5-turbo-0613", 
    "2" : "gpt-3.5-turbo-16k",
    "3" : "gpt-4",
    "4" : "gpt-3.5-turbo",
}
use_model = 1
# ------------------------------ #


@cl.langchain_factory(use_async=False)
def factory():
    # Initialize the OpenAI language model
    model = llms[str(use_model)]
    llm = ChatOpenAI(
        temperature=0, 
        model=model,
        streaming=True, 
        client="openai",
        # callbacks=[cl.ChainlitCallbackHandler()]
    )

    # Initialize the SerpAPIWrapper for search functionality
    search = SerpAPIWrapper(search_engine="google")

    # Define a list of tools offered by the agent
    tools = [
        Tool(
            name="Search",
            func=search.run,       # previously: search.run
            description="Useful when you need to answer questions about current events or if you have to search the web. You should ask targeted questions like for google."
        ),
        # Tool(
        #     name="PythonREPL",
        #     func=python_repl.run,
        #     description=(
        #         "A Python shell. Use this to execute python commands. "
        #         # "Input should be a valid python command. "
        #         "The input must be an object as follows: "
        #         "{'__arg1': 'a valid python command.'} "
        #         "If you want to see the output of a value, you should print it out "
        #         "with `print(...)`."
        #         "Don't add comments to your python code."
        #     )
        # ),
        CustomPythonAstREPLTool(),
        WriteFileTool(),
        ReadFileTool(),
        CopyFileTool(),
        MoveFileTool(),
        DeleteFileTool(),
        FileSearchTool(),
        ListDirectoryTool(),
        ShellTool(),
        # HumanInputRun(),
    ]

    # check if all tools have function _arun (async run)
    for tool in tools:
        if not hasattr(tool, "_arun"):
            raise Exception(f"Tool {tool['name']} has no async run function. Please add it.")

    # # Define your embedding model
    # embeddings_model = OpenAIEmbeddings()

    # # Initialize the vectorstore as empty
    # embedding_size = 1536
    # index = faiss.IndexFlatL2(embedding_size)
    # vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

    memory = ConversationTokenBufferMemory(
        memory_key="memory", 
        return_messages=True,
        max_token_limit=2000,
        llm=llm
    )

    # needed for memory with openai functions agent
    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    }

    # mrkl = initialize_agent(
    #     tools=tools,
    #     llm=llm, 
    #     agent=AgentType.OPENAI_MULTI_FUNCTIONS,
    #     verbose=True, 
    #     agent_kwargs=agent_kwargs, 
    #     memory=memory,
    #     # return_intermediate_steps=True,
    # )

    prompt = OpenAIFunctionsAgent.create_prompt(
        extra_prompt_messages=[MessagesPlaceholder(variable_name="memory")],
    ),

    print("Prompt: ", prompt[0])

    cust_agent = CustomOpenAIMultiFunctionsAgent(
        tools=tools,
        llm=llm,
        prompt=prompt[0],
        # kwargs=agent_kwargs,
        # return_intermediate_steps=True,
    )

    mrkl = AgentExecutor.from_agent_and_tools(
        agent=cust_agent,
        tools=tools,
        memory=memory,
        # kwargs=agent_kwargs,
        # return_intermediate_steps=True,
    )

    return mrkl



# ----- Custom classes and functions ----- #

class CustomPythonAstREPLTool(PythonAstREPLTool):
    name = "python"
    description = (
        "A Python shell. Use this to execute python commands. "
        "The input must be an object as follows: "
        "{'__arg1': 'a valid python command.'} "
        "When using this tool, sometimes output is abbreviated - "
        "Make sure it does not look abbreviated before using it in your answer. "
        "Don't add comments to your python code."
    )

def _parse_ai_message(message: BaseMessage) -> Union[AgentAction, AgentFinish]:
    """Parse an AI message."""
    if not isinstance(message, AIMessage):
        raise TypeError(f"Expected an AI message got {type(message)}")

    function_call = message.additional_kwargs.get("function_call", {})

    if function_call:
        function_call = message.additional_kwargs["function_call"]
        function_name = function_call["name"]
        try:
            _tool_input = json.loads(function_call["arguments"])
        except JSONDecodeError:
            print(
                f"Could not parse tool input: {function_call} because "
                f"the `arguments` is not valid JSON."
            )
            _tool_input = function_call["arguments"]

        # HACK HACK HACK:
        # The code that encodes tool input into Open AI uses a special variable
        # name called `__arg1` to handle old style tools that do not expose a
        # schema and expect a single string argument as an input.
        # We unpack the argument here if it exists.
        # Open AI does not support passing in a JSON array as an argument.
        if "__arg1" in _tool_input:
            tool_input = _tool_input["__arg1"]
        else:
            tool_input = _tool_input

        content_msg = "responded: {content}\n" if message.content else "\n"

        return _FunctionsAgentAction(
            tool=function_name,
            tool_input=tool_input,
            log=f"\nInvoking: `{function_name}` with `{tool_input}`\n{content_msg}\n",
            message_log=[message],
        )

    return AgentFinish(return_values={"output": message.content}, log=message.content)

class CustomOpenAIMultiFunctionsAgent(OpenAIMultiFunctionsAgent):
    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.
        Args:
            intermediate_steps: Steps the LLM has taken to date, along with observations
            **kwargs: User inputs.
        Returns:
            Action specifying what tool to use.
        """
        user_input = kwargs["input"]
        agent_scratchpad = _format_intermediate_steps(intermediate_steps)
        memory = kwargs["memory"]
        prompt = self.prompt.format_prompt(
            input=user_input, agent_scratchpad=agent_scratchpad, memory=memory
        )
        messages = prompt.to_messages()
        predicted_message = self.llm.predict_messages(
            messages, functions=self.functions, callbacks=callbacks
        )
        agent_decision = _parse_ai_message(predicted_message)
        return agent_decision
    


# @cl.langchain_run
# async def run(agent, input):
#     # Since the agent is sync, we need to make it async
#     res = await cl.make_async(agent.run)(
#         input, 
#         # callbacks=[cl.ChainlitCallbackHandler()]
#     )
#     await cl.Message(content=res).send()

# @cl.action_callback("action_button")
# async def on_action(action):
#     await cl.Message(content=f"Executed {action.name}").send()
#     # Optionally remove the action button from the chatbot user interface
#     await action.remove()

# @cl.on_chat_start
# async def start():
#     # Sending an action button within a chatbot message
#     actions = [
#         cl.Action(name="action_button", value="example_value", description="Click me!")
#     ]

#     await cl.Message(content="Interact with this action button:", actions=actions).send()