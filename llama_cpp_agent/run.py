from llama_cpp import Llama
from llama_cpp_agent import LlamaCppAgent, FunctionCallingAgent
from llama_cpp_agent.providers import LlamaCppPythonProvider
from llama_cpp_agent import MessagesFormatterType, LlamaCppFunctionTool
from pydantic import BaseModel, Field
import subprocess  # Keep for safe tools

llm = Llama(model_path="C:/models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf", n_ctx=4096, n_threads=8)
provider = LlamaCppPythonProvider(llm)

# Define safe tool (your subprocess logic here)
def is_safe_command(cmd: str) -> bool:
    """Rudimentary safety check to allow only a small whitelist of harmless commands."""
    allowed = ["dir", "ls", "echo", "type", "where", "Get-ChildItem"]
    stripped = cmd.strip()
    return any(stripped.startswith(a) for a in allowed)


def safe_execute(cmd: str):
    """Execute safe shell command."""
    if not is_safe_command(cmd):
        return {"error": "unsafe"}
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return {"stdout": result.stdout, "stderr": result.stderr, "code": result.returncode}

# Pydantic tool for agent
class CommandTool(BaseModel):
    command: str = Field(..., description="Safe shell command")
    def run(self):
        return safe_execute(self.command)

agent = FunctionCallingAgent(
    provider,
    llama_cpp_function_tools=[LlamaCppFunctionTool(CommandTool)],
    system_prompt="Plan steps, use tools when needed, respond in natural language.",
    messages_formatter_type=MessagesFormatterType.CHATML,
    allow_parallel_function_calling=True
)

response = agent.generate_response("List files in current directory")  # Handles planning + execution
print(response)
