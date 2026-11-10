import os
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import ToolSelection, ToolOutput
from llama_index.core.workflow import Event
from typing import Any, List

from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools.types import BaseTool
from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
)
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from src.memory_manger import BasicMemory
from dotenv import load_dotenv
import time

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# workflows events 
# handles new messages
class InputEvent(Event):
    input: list[ChatMessage]

# handle streaming response
class StreamEvent(Event):
    delta: str

# triger tool calls
class ToolCallEvent(Event):
    tool_calls:list[ToolSelection]

# handle result of tool call
class FunctionOutputEvent(Event):
    output: ToolOutput



class FunctionCallAgent(Workflow):
    def __init__(self,*args:Any
                 ,llm: FunctionCallingLLM | None = None
                 ,tools: List[BaseTool] | None = None
                 ,memory_cls = BasicMemory
                 ,**kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.tools = tools or []
        self.llm = llm or OpenAI()
        # to make sure that the used llm support tool calling
        assert self.llm.metadata.is_function_calling_model
        self.memory = memory_cls


    @step
    async def prepare_chat_history(self,ctx: Context,ev: StartEvent) -> InputEvent:
        
        memory = ev.get("memory", None)
        if memory is None:
            memory = await ctx.store.get("memory", default=None)
        if memory is None:
            memory = self.memory_cls()

        # get user query and save it in memory
        await memory.save_to_memory([ChatMessage(role="user", content=ev.input)])

        # retrieve the history 
        chat_history = await memory.get_history()
        # print("\n\n😀😀😀here is the history ",chat_history,"\n\n")

        # update context
        await ctx.store.set("memory", memory)

        return InputEvent(input=chat_history)


    @step
    async def handle_llm_input(self,ctx:Context,ev:InputEvent) -> ToolCallEvent | StopEvent:
        # stream reponse
        chat_history = ev.input
        response_stream = await self.llm.astream_chat_with_tools(
            self.tools,chat_history = chat_history
        )
        async for response in response_stream:
            ctx.write_event_to_stream(StreamEvent(delta=response.delta or ""))
        
        # save llm's final response into memory
        memory = await ctx.store.get("memory")
        await memory.save_to_memory([response.message])
        await ctx.store.set("memory",memory)
        
        # get tool calls
        tool_calls = self.llm.get_tool_calls_from_response(
            response, error_on_no_tool_call=False
        )


        if not tool_calls:
            sources = await ctx.store.get("sources", default=[])
            return StopEvent(
                result={"response":response,"sources": [*sources]}
            )
        else:
            return ToolCallEvent(tool_calls=tool_calls)

    @step
    async def handle_tool_calls(
        self, ctx: Context, ev: ToolCallEvent
    ) -> InputEvent:
        tool_calls = ev.tool_calls
        tools_by_name = {tool.metadata.get_name(): tool for tool in self.tools}

        tool_msgs = []
        sources = await ctx.store.get("sources", default=[])

        # call tools -- safely!
        for tool_call in tool_calls:
            tool = tools_by_name.get(tool_call.tool_name)
            additional_kwargs = {
                "tool_call_id": tool_call.tool_id,
                "name": tool.metadata.get_name(),
            }
            if not tool:
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=f"Tool {tool_call.tool_name} does not exist",
                        additional_kwargs=additional_kwargs,
                    )
                )
                continue

            try:
                tool_output = tool(**tool_call.tool_kwargs)
                sources.append(tool_output)
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=tool_output.content,
                        additional_kwargs=additional_kwargs,
                    )
                )
            except Exception as e:
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=f"Encountered error in tool call: {e}",
                        additional_kwargs=additional_kwargs,
                    )
                )

        # update memory
        memory = await ctx.store.get("memory")
        for msg in tool_msgs:
            await memory.save_to_memory([msg])

        await ctx.store.set("sources", sources)
        await ctx.store.set("memory", memory)

        chat_history = await memory.get_history()
        return InputEvent(input=chat_history)
    
# ---------------------------------------------------------------
