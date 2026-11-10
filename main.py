from src.agent import FunctionCallAgent
from src.memory_manger import BasicMemory,SummaryMemory

from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.core.workflow import Context
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core import Settings
import tiktoken
import time
from llama_index.core.llms import ChatMessage, MessageRole
import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

PERSIST_DIR = "./storage"
DATA_PATH = "./data"

def build_index():
    if not os.path.exists(PERSIST_DIR):
        documents = SimpleDirectoryReader(DATA_PATH).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
    return index 

def add(x: int, y: int) -> int:
    """Useful function to add two numbers."""
    return x + y


def multiply(x: int, y: int) -> int:
    """Useful function to multiply two numbers."""
    return x * y

def make_rag_tool(index):
    def rag_search(query: str) -> str:
        """Useful for answering questions about robotics study materials,
        including topics like localization, SLAM, kinematics, sensors, and
        control. Always prefer this tool over general knowledge for any
        robotics-related question, since the study materials may define
        concepts differently or with more specific detail."""
        query_engine = index.as_query_engine()
        response = query_engine.query(query)
        return str(response)
    return rag_search
index = build_index()
rag_search = make_rag_tool(index)


tools = [
    FunctionTool.from_defaults(add),
    FunctionTool.from_defaults(multiply),
    FunctionTool.from_defaults(rag_search)
]
# -------------------------------------




async def chat(agent,ctx,memory,token_counter,constat_converation=None):

    for user_query in constat_converation:
        token_counter.reset_counts()
        
        # run the agent
        start = time.perf_counter()
        result = await agent.run(input = user_query,ctx=ctx,memory=memory)
        end = time.perf_counter()

        response = result["response"]
        # total time the agent took to answer
        total_time = end - start
        # token counts for current question
        prompt_tokens = token_counter.prompt_llm_token_count
        print(f"🕛Latency:      {total_time:.2f} seconds")
        print(f"🧐Tokens Used:  {prompt_tokens} (PromptContext)")
        # print(response,"\n") 
        print("*"*50,"\n")
        
        
    # print(response,"\n")
    print(f"\n📊 Total Tokens consumed by {memory}: {token_counter.total_llm_token_count}")

    return await memory.get_history()
    

async def main():
    # chat setup
    # token counter setup
    token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("gpt-4o-mini").encode
    )
    callback_manager = CallbackManager([token_counter])

    Settings.callback_manager = callback_manager

    llm = OpenAI(
        model="gpt-4o-mini",
        callback_manager=callback_manager,
        temperature=0.0,
    )
    agent = FunctionCallAgent(
        llm=llm, tools=tools, timeout=120, verbose=True
    )
    memory = SummaryMemory()
    ctx = Context(agent)
    
    # constaant conversation to test prompt summary
    benchmark_conversation = [
       
    # Message 1: Setting up the case and context (Mixture of dialect and legal terminology)
    "أنا شغال على قضية طرد مستأجر لعدم سداد الأجرة. العين المؤجرة شقة سكنية في الدقي، والمالك (المؤكل بتاعي) اسمه أحمد محمد حسن، والمستأجر اسمه جرجس فؤاد. العقد تاريخه 1 مارس 2021 والمسألة معروضة قدام محكمة شمال الجيزة الابتدائية، دائرة 4 إيجارات. وعايز أتأكد من الإجراءات.",
    
    # Message 2: Requesting theoretical/procedural breakdown (Standard legal question)
    "ممكن تشرح لي باختصار شروط قبول دعوى الإخلاء للتكرار في قانون الإيجار القديم والجديد؟ وإيه المواعيد القانونية لإنذار التكليف بالوفاء؟",
    
    # Message 3: Drafting/Detailed explanation request (Deep technical task)
    "تمام، اكتب لي صيغة إنذار بالتكليف بالوفاء بالأجرة المتأخرة يكون مستوفي كل الشروط القانونية عشان ما ياخدش بطلان، مع توضيح الدفوع اللي ممكن المستأجر يرفعها زي الدفع بالتنازل الضمني عن الأجرة.",
    
    # Message 4: The Memory Test (Checking if the LLM retains the initial constraints)
    "إيه هي المحكمة المختصة والدائرة اللي أنا ذكرتها في أول رسالة خالص؟ ومين هو المدعي والمدعى عليه في الحالة بتاعتنا؟"
]

    # test memory method

    history = await chat(agent,ctx,memory,token_counter,benchmark_conversation)
    print(f"Final token counts of the fistory after 4 turns {memory._count_tokens(history)}")
    # print(history)
    # turns = memory._build_turns(history)
    # old_turns, recent_turns = memory._select_recent_turns(turns)
    # print("Old turns")
    # for t in old_turns:
    #     print("----------------")
    #     for msg in t:
    #         print(msg.role, msg.content)

    # print("\nRecent turns")
    # for t in recent_turns:
    #     print("----------------")
    #     for msg in t:
    #         print(msg.role, msg.content)
    
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
    