# Running the app:
# chainlit run app.py -w --no-cache  (-w for watch mode (auto-reload), --no-cache to disable cache)

# Imports
from _utils import (
    get_config,
    init_langchain,
)
from prompts import (
    cust_pyrepl_desc,
)
import os
import langchain
from langchain import SerpAPIWrapper
from langchain.callbacks.manager import Callbacks, AsyncCallbackManagerForChainRun
# from langchain.prompts import ChatPromptTemplate
from langchain.agents import Tool, AgentType, initialize_agent
from langchain.agents.agent import AgentExecutor, ExceptionTool
from langchain.agents.tools import InvalidTool
from langchain.agents.agent_toolkits import PlayWrightBrowserToolkit
from langchain.tools.playwright.utils import (
    create_async_playwright_browser,
    create_sync_playwright_browser,  # A synchronous browser is available, though it isn't compatible with jupyter.
)
from langchain.tools.base import BaseTool
from langchain.agents.openai_functions_multi_agent.base import OpenAIMultiFunctionsAgent
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.chat_models import ChatOpenAI
# from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.tools.file_management import ReadFileTool, CopyFileTool, MoveFileTool, WriteFileTool, DeleteFileTool, FileSearchTool, ListDirectoryTool
from langchain.tools.shell import ShellTool
from langchain.tools.python.tool import PythonAstREPLTool
from langchain.tools.human.tool import HumanInputRun
from langchain.memory import ConversationTokenBufferMemory
#from langchain.memory.chat_message_histories import FileChatMessageHistory
from langchain.prompts import MessagesPlaceholder   # , HumanMessagePromptTemplate
# from langchain.vectorstores import FAISS
# from langchain.docstore import InMemoryDocstore
# from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import BaseMessage, AIMessage, AgentAction, AgentFinish, OutputParserException
from langchain.agents.openai_functions_multi_agent.base import _FunctionsAgentAction, _format_intermediate_steps
from typing import Union, List, Any, Tuple, Dict, Optional
from json import JSONDecodeError
import asyncio
import json
# import faiss
import chainlit as cl
import dotenv
# import nest_asyncio
# nest_asyncio.apply()


# ------- Prerequisites
dotenv.load_dotenv()
config = get_config("config.yaml")
init_langchain(config)


# Factory
use_async = False
@cl.langchain_factory(use_async=use_async)
def factory():
    # Initialize the OpenAI language model
    llm = ChatOpenAI(
        model=config["base_llm"]["model"],
        temperature=config["base_llm"]["temperature"],
        streaming=config["base_llm"]["streaming"],
        client="openai",
        # callbacks=[cl.ChainlitCallbackHandler()]
    )

    # Initialize the SerpAPIWrapper for search functionality
    search = SerpAPIWrapper(search_engine="google")

    # Define a list of tools offered by the agent
    tools = [
        Tool(
            name="search",
            func=search.run,       # previously: search.run
            description="Useful when you need to answer questions about current events or if you have to search the web. You should ask targeted questions like for google."
        ),
        PythonAstREPLTool(
            # description = cust_pyrepl_desc,
        ),
        WriteFileTool(),
        ReadFileTool(),
        CopyFileTool(),
        MoveFileTool(),
        DeleteFileTool(),
        FileSearchTool(),
        ListDirectoryTool(),
        ShellTool(),
        HumanInputRun(),
    ]
    # playwright_browser = create_async_playwright_browser()

    # playwright_tools = PlayWrightBrowserToolkit.from_browser(async_browser=playwright_browser).get_tools()
    # tools.extend(playwright_tools)

    # check if all tools have an async run function
    if use_async:
        for tool in tools:
            if not hasattr(tool, "_arun"):
                raise Exception(f"Tool {tool['name']} has no async run function.")

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

    agent_exec = initialize_agent(
        tools=tools,
        llm=llm, 
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        # verbose=True, 
        agent_kwargs=agent_kwargs, 
        memory=memory,
        # return_intermediate_steps=True,
    )

    return agent_exec



# ----- Custom classes and functions ----- #


    
    
# async def _atake_next_step(
#         self,
#         name_to_tool_map: Dict[str, BaseTool],
#         color_mapping: Dict[str, str],
#         inputs: Dict[str, str],
#         intermediate_steps: List[Tuple[AgentAction, str]],
#         run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
#     ) -> Union[AgentFinish, List[Tuple[AgentAction, str]]]:
#         """Take a single step in the thought-action-observation loop.

#         Override this to take control of how the agent makes and acts on choices.
#         """
#         try:
#             # Call the LLM to see what to do.
#             output = await self.agent.aplan(
#                 intermediate_steps,
#                 callbacks=run_manager.get_child() if run_manager else None,
#                 **inputs,
#             )
#         except OutputParserException as e:
#             if isinstance(self.handle_parsing_errors, bool):
#                 raise_error = not self.handle_parsing_errors
#             else:
#                 raise_error = False
#             if raise_error:
#                 raise e
#             text = str(e)
#             if isinstance(self.handle_parsing_errors, bool):
#                 if e.send_to_llm:
#                     observation = str(e.observation)
#                     text = str(e.llm_output)
#                 else:
#                     observation = "Invalid or incomplete response"
#             elif isinstance(self.handle_parsing_errors, str):
#                 observation = self.handle_parsing_errors
#             elif callable(self.handle_parsing_errors):
#                 observation = self.handle_parsing_errors(e)
#             else:
#                 raise ValueError("Got unexpected type of `handle_parsing_errors`")
#             output = AgentAction("_Exception", observation, text)
#             tool_run_kwargs = self.agent.tool_run_logging_kwargs()
#             observation = await ExceptionTool().arun(
#                 output.tool_input,
#                 verbose=self.verbose,
#                 color=None,
#                 callbacks=run_manager.get_child() if run_manager else None,
#                 **tool_run_kwargs,
#             )
#             return [(output, observation)]
#         # If the tool chosen is the finishing tool, then we end and return.
#         if isinstance(output, AgentFinish):
#             return output
#         actions: List[AgentAction]
#         if isinstance(output, AgentAction):
#             actions = [output]
#         else:
#             actions = output

#         async def _aperform_agent_action(
#             agent_action: AgentAction,
#         ) -> Tuple[AgentAction, str]:
#             if run_manager:
#                 await run_manager.on_agent_action(
#                     agent_action, verbose=self.verbose, color="green"
#                 )
#             # Otherwise we lookup the tool
#             if agent_action.tool in name_to_tool_map:
#                 tool = name_to_tool_map[agent_action.tool]
#                 return_direct = tool.return_direct
#                 color = color_mapping[agent_action.tool]
#                 tool_run_kwargs = self.agent.tool_run_logging_kwargs()
#                 if return_direct:
#                     tool_run_kwargs["llm_prefix"] = ""
#                 # We then call the tool on the tool input to get an observation
#                 observation = await tool.arun(
#                     agent_action.tool_input,
#                     verbose=self.verbose,
#                     color=color,
#                     callbacks=run_manager.get_child() if run_manager else None,
#                     **tool_run_kwargs,
#                 )
#             else:
#                 tool_run_kwargs = self.agent.tool_run_logging_kwargs()
#                 observation = await InvalidTool().arun(
#                     agent_action.tool,
#                     verbose=self.verbose,
#                     color=None,
#                     callbacks=run_manager.get_child() if run_manager else None,
#                     **tool_run_kwargs,
#                 )
#             return agent_action, observation

#         # Use asyncio.gather to run multiple tool.arun() calls concurrently
#         result = await asyncio.gather(
#             *[_aperform_agent_action(agent_action) for agent_action in actions]
#         )

#         return list(result)


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