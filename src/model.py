import os
import sys
import contextlib
import gc
import torch

# vLLM imports for model loading and inference
from vllm import LLM, SamplingParams
from vllm.distributed import destroy_distributed_environment, destroy_model_parallel

# Project-specific configuration and helpers
from prompts import *
from utils   import *
from config  import *

# Prevent Python from generating .pyc (bytecode cache) files
# Useful for cleaner environments and containerized deployments
sys.dont_write_bytecode = True

# Ensure CUDA devices are enumerated by PCI bus ID
# This guarantees consistent GPU ordering across runs
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["VLLM_TARGET_DEVICE"] = "cuda"

# Restrict visible GPUs to those specified in config (e.g. "0,1")
if len(gpu_id) > 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

class Model:
    """
    Wrapper class around vLLM to manage:
    - Message histories (chat state)
    - Prompt formatting for different model families
    - Text generation
    - Cleanup of distributed GPU resources
    """

    # Stores multiple independent conversation histories
    messageHistories = []

    # vLLM-related objects
    llm = None
    sampling_params = None
    model = None

    def __init__(self, model : str = modelID):
        """
        Initialize the LLM and sampling parameters.
        """
        self.model = model

        log(f"Available: {torch.cuda.is_available()}")
        log(f"Device count: {torch.cuda.device_count()}")
        log(f"Max Num batched Tokens: {max_num_batched_tokens}")
        log(f"Max new Tokens: {max_tokens}")
        log(f"Temperature: {temperature}")
        log(f"Max Model Length: {max_model_len}")
        log(f"Few-Shot (Zero-Shot): {fewShot}")
        log(f"Chain-Of-Thoughts: {chainOfThoughts}")
        log(f"Add Definitions to Prompt: {addDefinition}")
        log(f"Add Comment to Prompt: {addComment}")
        log(f"Add Parents to Prompt: {addParents}")
        log(f"Add Children to Prompt: {addChildren}")
        joinString = '\', \''
        log(f"GPUs: '{joinString.join(gpu_id.split(','))}'")

        # Initialize vLLM engine
        self.llm = LLM(
            model=model,
            # Number of GPUs used for tensor parallelism
            tensor_parallel_size=len(gpu_id.split(",")),
            max_model_len=max_model_len,
            # Enable expert parallelism only for specific models
            enable_expert_parallel=(
                modelID == "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
            )
        )

        # Sampling configuration for text generation
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )


    def addPrompt(self, role : str = userRole, message : list = []) -> int:
        """
        Add a message (or messages) to the conversation histories.

        Supports:
        - Broadcasting a single message to all histories
        - Appending one message per history
        - Creating new histories if none exist

        Returns:
            Number of active message histories
        """
        ret = 0

        # Case 1: Single message broadcast to all histories
        if len(message) == 1 and len(self.messageHistories) > 0:
            for index in range(0, len(self.messageHistories)):
                self.messageHistories[index].append({
                    messageRoleElement : role,
                    messageTextElement : message[0]
                })
            ret = len(self.messageHistories)
        else:
            # Case 2: One message per history
            if len(message) == len(self.messageHistories):
                for index in range(0, len(self.messageHistories)):
                    self.messageHistories[index].append({
                        messageRoleElement : role,
                        messageTextElement : message[index]
                    })
                ret = len(self.messageHistories)
            else:
                # Case 3: Initialize histories when none exist
                if len(self.messageHistories) == 0 and len(message) > 0:
                    for index in range(0, len(message)):
                        history = []
                        history.append({
                            messageRoleElement : role,
                            messageTextElement : message[index]
                        })
                        self.messageHistories.append(history)
                    ret = len(self.messageHistories)

        return ret
    
    def generateGemma(self, logging : bool = False) -> None:
        """
        Generate responses using Gemma-style prompt formatting.
        """
        inputs = []
        
        # Build a prompt for each conversation history
        for messageHistory in self.messageHistories:
            prompt = ""

            s = False # Tracks whether a system message was just processed
            for message in messageHistory:
                if message[messageRoleElement] == systemRole:

                    # System messages are converted into user turns
                    prompt += f"{startTurn}{userRole}\n" \
                            f"{message[messageTextElement]}\n\n"
                    s = True
                else:
                    if s:
                        # Close system-injected user message
                        prompt +=  f"{message[messageTextElement]}{endTurn}\n"
                        s = False
                    else:
                        # Standard user/assistant turn
                        prompt += f"{startTurn}{message[messageRoleElement]}\n" \
                            f"{message[messageTextElement]}{endTurn}\n"

            # Prepare model to generate the next assistant response
            prompt += f"{startTurn}{modelRole}"

            if logging:
                log(prompt)

            inputs.append(prompt)

        # Run inference  
        generatedText = self.llm.generate(inputs, self.sampling_params, use_tqdm=False)
        outputs = []

        # Extract and clean generated text
        for text in generatedText:
            outputs.append(formatGeneratedText(text.outputs[0].text))

        # Append model responses to histories
        self.addPrompt(role = modelRole, message = outputs)

    def generateLlama(self, logging : bool = False) -> None:
        """
        Generate responses using LLaMA-style prompt formatting.
        """
        inputs = []

        # Add each message with LLaMA headers
        for messageHistory in self.messageHistories:
            prompt = f"{beginOfText}"
            
            for message in messageHistory:
                prompt += f"{startHeader}{message[messageRoleElement]}" \
                    f"{endHeader}{message[messageTextElement]}{endOfText}"

            # Signal model to generate assistant response
            prompt += f"{startHeader}{modelRole}{endHeader}"

            if logging:
                log(prompt, cmdline = False)

            inputs.append(prompt)

        # Run inference 
        generatedText = self.llm.generate(inputs, self.sampling_params)
        outputs = []

        for text in generatedText:
            outputs.append(formatGeneratedText(text.outputs[0].text))

        self.addPrompt(role = modelRole, message = outputs)

    def generate(self, logging : bool = False) -> None:
        """
        Dispatch generation based on model type.
        Currently defaults to Gemma formatting.
        """
        # if "gemma" in self.model.lower():
        self.generateGemma(logging)
        # else:
        #     self.generateLlama(logging)

    def getMessageHistories(self) -> list[list[object]]:
        """
        Return all stored conversation histories.
        """
        return self.messageHistories

    def reset(self) -> None:
        """
        Clear all conversation histories.
        """
        self.messageHistories = []

    def logPrompts(self, file : str = logFilePrompts) -> None:
        """
        Log every prompt from every history to a file.
        """
        for index in range(0, len(self.messageHistories)):
            for index2 in range(0, len(self.messageHistories[index])):
                self.logPrompt(file, index, index2)

    def logPrompt(
            self, 
            file: str = logFilePrompts, 
            indexHistory : int = 0, 
            indexPrompt : int = -1
    )-> None:
        """
        Log a single prompt entry to file.
        """
        message = self.messageHistories[indexHistory][indexPrompt]
        log(f"Role: {message[messageRoleElement]}", False, file)
        log("Message: ", False, file)
        log(message[messageTextElement], False, file)

    def __del__(self) -> None:
        """
        Destructor to aggressively clean up GPU and distributed resources.
        Prevents memory leaks and CUDA context issues.
        """
        try:
            del self.llm
        except:
            pass
        try:
            destroy_model_parallel()
        except:
            pass
        try:
            destroy_distributed_environment()
        except:
            pass
        try:
            with contextlib.suppress(AssertionError):
                torch.distributed.destroy_process_group()
        except:
            pass
        try:
            gc.collect()
        except:
            pass
        try:
            torch.cuda.empty_cache()
        except:    
           pass
            
def formatGeneratedText(text : str) -> str:
    """
    Remove model-specific tokens and role markers
    from generated text before returning it.
    """
    ret = text.replace(startHeader, "").replace(endHeader, 
        "").replace(endOfText, "").replace(beginOfText, "").replace(modelRole, 
        "").replace(userRole, ""). replace(systemRole, "").strip()
    return ret