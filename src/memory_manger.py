from llama_index.core.memory import Memory
from typing import Any, List,  Tuple
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.openai import OpenAI
from abc import ABC,abstractmethod
from llama_index.core.utils import get_tokenizer


class BaseMemory(ABC):
    """
    Abstract interface that every memory strategy must implement.
    """
    @abstractmethod
    async def save_to_memory(self,msg:List[ChatMessage]) -> None:
        pass

    @abstractmethod
    async def get_history(self) -> List[ChatMessage]:
        pass
    
    @abstractmethod
    async def reset(self) -> None:
        pass
class BasicMemory(BaseMemory):

    def __init__(self,session_id="my_session",
        token_limit=10000,
        chat_history_token_ratio=0.99,
        token_flush_size=100):
        
        self.memory = Memory.from_defaults(
            session_id=session_id,
            token_limit=token_limit,
            chat_history_token_ratio=chat_history_token_ratio,
            token_flush_size=token_flush_size
        )
        

    async def save_to_memory(self,msg:List[ChatMessage]):
        """save chat into memory"""
        await self.memory.aput_messages(msg)

    async def get_history(self):
        """retrieve chat history"""
        return await self.memory.aget()
    
    async def reset(self) -> None:
        """Reset the memory."""
        await self.memory.areset()
# keeps the last n turns 
class old_SummaryMemory(BaseMemory):
    def __init__(self,session_id="my_session",
        token_limit=10000,
        chat_history_token_ratio=1,
        token_flush_size=1000):
        
        self.memory = Memory.from_defaults(
            session_id=session_id,
            token_limit=token_limit,
            chat_history_token_ratio=chat_history_token_ratio,
            token_flush_size=token_flush_size
        )
        self.llm = OpenAI(
            model = "gpt-4o-mini",    
            system_prompt="You are a summrizer that summrize the chat history for a legal memory agent that is used in law firm by lawyers to aquire about legal documents," \
            "you should focus on the important legal stuff when summrizing the chat history," \
            "you must write the summary in the same language that is used by the user in the provided chat history.")
        
        self.short_term_token_limit = int(token_limit * chat_history_token_ratio)

        self.tokenizer = get_tokenizer()

        # keep the last 4 messages as they are and summrize the previous msgs
        self.n_msgs_to_keep= 4
        self.threshold = self.short_term_token_limit * 0.5


    async def get_current_token_count(self,msgs: List[ChatMessage]):
        """count token in current history"""
        return sum(len(self.tokenizer(msg.content)) for msg in msgs if msg.content)
    
    async def summrize_history(self):
        print("☠️☠️☠️Memory reached its Limit , start summrizing history☠️☠️☠️")
        history = await self.get_history()
        print(f"😀😀😀length of history before summarizing :{len(history)}")
        
        if len(history) <= self.n_msgs_to_keep:
            return
        
        msgs_to_summarize = history[:-self.n_msgs_to_keep]
        msgs_to_keep = history[-self.n_msgs_to_keep:]
        print(f"\n😀😀😀messages that will be summrized :\n {msgs_to_summarize}, length: {len(msgs_to_summarize)}\n messages that will be keept :\n{msgs_to_keep}, length: {len(msgs_to_keep)}\n")

        # extract existed summary , so it won't be summrized again, it will be just updated
        existing_summary = ""
        if msgs_to_summarize and msgs_to_summarize[0].role == MessageRole.SYSTEM:
            existing_summary = msgs_to_summarize[0].content
            msgs_to_summarize = msgs_to_summarize[1:]
        
        dialogue_to_summarize = []
        verbatim_tool_results = []
        for msg in msgs_to_summarize:
            if msg.role == MessageRole.TOOL or msg.role == MessageRole.FUNCTION:
                verbatim_tool_results.append(msg)
            else:
                dialogue_to_summarize.append(msg)

        chat_text = "\n".join([f"{msg.role}:{msg.content}" for msg in dialogue_to_summarize])

        if existing_summary:
            prompt = (
                f"Here is the current summary of the older conversation:\n{existing_summary}\n\n"
                f"Here are the new messages to incorporate:\n{chat_text}\n\n"
                "Please seamlessly update the current summary with the new information. "
                "Keep it concise and focus on the important legal context."
            )
            content = ""
        else:
            prompt = (
                f"Here is the chat history that you should summarize:\n{chat_text}\n\n"
                "Create a concise summary focusing on the important legal context."
            )
            content = "Summary of previous conversation: "

        response = await self.llm.acomplete(prompt)
        summary_text = response.text
        print("😀😀😀Here is the memory summary",summary_text)
        # wrap the summrized history into ChatMessage
        summary = ChatMessage(
            role=MessageRole.SYSTEM, 
            content=content + summary_text
        )

        # clear the full memory
        await self.reset()

        # save the new history 
        new_chat_history = [summary] + verbatim_tool_results + msgs_to_keep 
        await self.memory.aput_messages(new_chat_history)
        

    async def save_to_memory(self,msg:List[ChatMessage]):
        """save chat into memory"""
        # token count of history and new message
        history = await self.get_history()
        history_token_count = await self.get_current_token_count(history)
        new_msg_token_count = await self.get_current_token_count(msg)

        total_future_token = history_token_count + new_msg_token_count

        # trigger summary if total future token count crossed the threshold
        if total_future_token > self.threshold :
            print(f"🧐🧐🧐Current history tokens : {history_token_count} \
                \nfuture token count if we inserted the new message {total_future_token}\
                \nmemory tokens : {self.short_term_token_limit}\
                \nthreshold : {self.threshold}")
            await self.summrize_history()

        await self.memory.aput_messages(msg)

    async def get_history(self):
        """retrieve chat history"""
        return await self.memory.aget()
    
    async def reset(self) -> None:
        """Reset the memory."""
        await self.memory.areset()
# keep the last n tokens count , except the last turn whatever its size
# -----------------------------------------------------
class SummaryMemory(BaseMemory):
    def __init__(self,session_id="my_session",
        token_limit=20000,
        chat_history_token_ratio=1,
        token_flush_size=1000):
        
        self.memory = Memory.from_defaults(
            session_id=session_id,
            token_limit=token_limit,
            chat_history_token_ratio=chat_history_token_ratio,
            token_flush_size=token_flush_size
        )
        self.llm = OpenAI(
            model = "gpt-4o-mini",    
            system_prompt="""
            Role: You are an expert Egyptian Legal Archivist AI. Your task is to summarize the older conversation history between an Egyptian lawyer and an AI assistant to save token space, WITHOUT losing any critical legal metadata.

            Input Language: The conversation will be in Arabic (Egyptian dialect or Standard Arabic).
            Output Language: Your summary MUST be in Arabic, maintaining exact Egyptian legal terminology.

            Objective: Compress the conversation length while strictly preserving the factual, procedural, and legal context required for future advice.

            CRITICAL ENTITIES TO EXTRACT & PRESERVE:
            1. Case & Court Data: Case numbers, judicial years (السنة القضائية), court names, degrees of litigation (جزئي، ابتدائي، استئناف، نقض), and circuit numbers.
            2. Parties & Capacities: Names of plaintiffs (المدعي), defendants (المدعى عليه), appellants (المستأنف), and their exact legal capacities. Include original and chosen domiciles (الموطن المختار) if mentioned.
            3. Dates & Timelines: Session dates (تاريخ الجلسة), filing dates, notification dates (تاريخ إعلان صحيفة الدعوى), and statutory deadlines.
            4. Core Legal Matter: 
            - The exact final requests (الطلبات الختامية).
            - Legal grounds and referenced laws/articles (الأسانيد القانونية ومواد القانون).
            - Formal and substantive defenses raised (الدفوع الشكلية والموضوعية).
            5. Documents & Contracts: Details of promissory notes (سند لأمر), contracts, and specific clauses (e.g., penalty clauses, jurisdiction).

            Rules:
            - DO NOT translate Egyptian legal terms into English or general Arabic. Keep terms like (صحيفة دعوى, إنذار على يد محضر, حافظة مستندات) exactly as they appeared.
            - DO NOT hallucinate or infer missing details. If a case number is not mentioned, do not invent one.
            - Compress greetings, brainstorming, and conversational filler aggressively.
            - Format the output as a concise overview followed by bullet points for the preserved metadata.
            """
            ,temprature=0)
        self.short_term_token_limit = int(token_limit * chat_history_token_ratio)
        self.tokenizer = get_tokenizer()
        self.threshold = int(self.short_term_token_limit * 0.75)
        self.keep_budget = 5000
        self.max_recent_turns = 2
        

    def _count_tokens(self,msgs: List[ChatMessage]):
        """count token in current history"""
        return sum(len(self.tokenizer(msg.content)) for msg in msgs if msg.content)
    

    def _build_turns(self, history: List[ChatMessage]) -> List[List[ChatMessage]]:
        """Convert a flat chat history into conversation turns."""
        history_turns = []
        current_turn = []

        for msg in history:
            # ignore system messages 
            if msg.role == MessageRole.SYSTEM:
                continue
            # start new turn
            if msg.role == MessageRole.USER:
                # save previous turn if exist
                if current_turn:
                    history_turns.append(current_turn)
                # start new turn
                current_turn = [msg]
            else:
                if current_turn:
                    current_turn.append(msg)
        # last turn
        if current_turn:
            history_turns.append(current_turn)

        return history_turns
    

    def _select_recent_turns(
        self,
        history_turns: List[List[ChatMessage]],
    ) -> Tuple[List[List[ChatMessage]], List[List[ChatMessage]]]:
        """split the turns into recent turns and old turns"""
        # check if history turns empty
        if not history_turns:
            return [],[]
        
        recent_turns = []
        current_tokens = 0

        # keep the last newst turn
        newest_turn = history_turns[-1]
        recent_turns.append(newest_turn)
        current_tokens += self._count_tokens(newest_turn)
        kept_turns = 1

        # walk backwards
        for turn in reversed(history_turns[:-1]):
            turn_tokens = self._count_tokens(turn)

            if kept_turns >= self.max_recent_turns:
                break

            if current_tokens + turn_tokens > self.keep_budget:
                break

            recent_turns.append(turn)
            current_tokens += turn_tokens
            kept_turns += 1

        # return recent turns to normal oreder
        recent_turns.reverse()
        num_recent = len(recent_turns)
        old_turns = history_turns[:-num_recent]

        return old_turns, recent_turns

    def _flatten_turns(
        self,
        turns: List[List[ChatMessage]],
    ) -> List[ChatMessage]:
        """flatten the turns lists"""

        flattened = []
        for turn in turns:
            flattened.extend(turn)

        return flattened
    
    def _prepare_history_for_summary(self,history:List[ChatMessage]
        ) -> Tuple[str, List[ChatMessage], List[ChatMessage]] :
        """prepares and splits the history for summarization"""
       
        print(f"😀😀😀length of history before summarizing :{len(history)}")

        # extract existed summary , so it won't be summrized again, it will be just updated
        existing_summary = ""
        if history  and history [0].role == MessageRole.SYSTEM:
            existing_summary = history[0].content
            history  = history [1:]
        
        # build turns
        history_turns = self._build_turns(history)
        # split turns
        old_turns, recent_turns = self._select_recent_turns(history_turns)
        # Convert back into ChatMessage lists
        msgs_to_summarize = self._flatten_turns(old_turns)
        msgs_to_keep = self._flatten_turns(recent_turns)
        
        print(f"\n😀😀😀length of messages that will be summrized : {len(msgs_to_summarize)}\nlength of messages that will be keept :{len(msgs_to_keep)}\n")

        return existing_summary,msgs_to_summarize,msgs_to_keep
    
    async def _generate_summary(self,dialogue_messages,existing_summary):
        """llm generates the text summary"""
        
        chat_text = "\n".join([f"{msg.role}:{msg.content}" for msg in dialogue_messages])

        if existing_summary:
            prompt = (
                f"Here is the current summary of the older conversation:\n{existing_summary}\n\n"
                f"Here are the new messages to incorporate:\n{chat_text}\n\n"
                "Please seamlessly update the current summary with the new information. "
                "Keep it concise and focus on the important legal context."
            )
            content = ""
        else:
            prompt = (
                f"Here is the chat history that you should summarize:\n{chat_text}\n\n"
                "Create a concise summary focusing on the important legal context."
            )
            content = "Summary of previous conversation: "

        response = await self.llm.acomplete(prompt)

        summary_text = response.text

        summary_text = content + summary_text
        print("😀😀😀Here is the memory summary",summary_text)
        return summary_text

    async def _rebuild_memory(self,summary_text,msgs_to_keep,verbatim_tool_results=None):
        """prepare summary chatmessage and rebuild the memory"""
        # wrap the summrized history into ChatMessage
        summary = ChatMessage(
            role=MessageRole.SYSTEM, 
            content=summary_text
        )

         # clear the full memory
        await self.reset()

        # save the new history 
        new_chat_history = [summary] + msgs_to_keep 

        await self.memory.aput_messages(new_chat_history)

    async def summarize_history(self):
        """make summary for history"""
        print("☠️☠️☠️Memory reached its Limit , start summrizing history☠️☠️☠️")

        history = await self.get_history()
        existing_summary, msgs_to_summarize, msgs_to_keep = (self._prepare_history_for_summary(history))

        if not msgs_to_summarize:
            return

        # dialogue, tool_messages = self._split_dialogue_and_tools(msgs_to_summarize)

        summary = await self._generate_summary(msgs_to_summarize,existing_summary)

        await self._rebuild_memory(summary,msgs_to_keep)

    async def check_if_summarize_is_needed(self,msg):
        # token count of history and new message
        history = await self.get_history()
        history_token_count = self._count_tokens(history)
        
        # new_msg_token_count = self._count_tokens(msg)
        # total_future_token = history_token_count + new_msg_token_count

        # trigger summary if total future token count crossed the threshold
        if history_token_count > self.threshold :
            print(f"🧐🧐🧐Current history tokens : {history_token_count} \
                \nmemory tokens : {self.short_term_token_limit}\
                \nthreshold : {self.threshold}")
            await self.summarize_history()
            
    async def save_to_memory(self,msg:List[ChatMessage]):
        """save chat into memory"""
        
        await self.memory.aput_messages(msg)
        await self.check_if_summarize_is_needed(msg)

    

    async def get_history(self):
        """retrieve chat history"""
        return await self.memory.aget()
    
    async def reset(self) -> None:
        """Reset the memory."""
        await self.memory.areset()

# not important for now
    def _split_dialogue_and_tools(self,msgs_to_summarize:List[ChatMessage]
        ) -> tuple[List[ChatMessage], List[ChatMessage]]:
        dialogue_to_summarize = []
        verbatim_tool_results = []
        for msg in msgs_to_summarize:
            if msg.role in (MessageRole.TOOL, MessageRole.FUNCTION):
                verbatim_tool_results.append(msg)
            else:
                dialogue_to_summarize.append(msg)

        return dialogue_to_summarize,verbatim_tool_results
     