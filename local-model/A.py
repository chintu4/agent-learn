from llama_cpp import Llama
import json
import re
import subprocess

# =========================
# Model init
# =========================
llm = Llama(
    model_path="C:/models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf",
    n_ctx=4096,
    n_threads=8
)

SYSTEM_PROMPT = (
    "JSON ONLY.\n"
    "Return ONE JSON object.\n"
    "No text outside JSON.\n"
    "\n"
    "You MUST return this structure:\n"
    "{\n"
    "  \"plan\": [\"step1\", \"step2\"],\n"
    "  \"action\": \"answer\" | \"command\",\n"
    "  \"expose_to_user\": true | false,\n"
    "  \"data\": {...} | null,\n"
    "  \"command\": \"...\" | null\n"
    "}\n"
    "\n"
    "Rules:\n"
    "- plan is always required\n"
    "- expose_to_user decides visibility (prefer true unless asked to hide)\n"
    "- command is used only if action == command; if exist:command response will be provided\n"
    "- If unsure, return {\"error\":\"unknown\"}\n"
)

# =========================
# Helpers
# =========================
def extract_json_from_text(s: str):
    starts = [m.start() for m in re.finditer(r"[\{\[]", s)]
    pairs = {"{": "}", "[": "]"}
    for start in starts:
        stack = []
        for i in range(start, len(s)):
            c = s[i]
            if c in "{[":
                stack.append(pairs[c])
            elif stack and c == stack[-1]:
                stack.pop()
                if not stack:
                    try:
                        return json.loads(s[start:i+1])
                    except Exception:
                        break
    return None


def is_safe_command(command: str) -> bool:
    banned = ["&&", "|", ";", ">", "<", "$", "`", "rm ", "del ", "shutdown", "format"]
    if not command or len(command) > 200:
        return False
    return not any(b in command for b in banned)

# =========================
# Agent core
# =========================
def ask_model(prompt: str, max_retries: int = 3) -> dict:
    last_error = None

    for _ in range(max_retries):
        out = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            stream=False,
        )

        raw = out["choices"][0]["message"]["content"]

        try:
            return json.loads(raw)
        except Exception as e:
            extracted = extract_json_from_text(raw)
            if extracted is not None:
                return extracted
            last_error = str(e)

    return {"error": "invalid_json", "detail": last_error}


def _coerce_agent_json(agent_json: dict) -> dict:
    """Normalize loosely-valid JSON into the required agent schema.

    This prevents hard failures like {"error":"missing_plan"} when the model
    returns JSON that doesn't perfectly match the schema.
    """
    if not isinstance(agent_json, dict):
        return {"error": "invalid_json_type", "detail": str(type(agent_json))}
    if "error" in agent_json:
        return agent_json

    schema_keys = {"plan", "action", "expose_to_user", "data", "command"}
    coerced = dict(agent_json)

    if not isinstance(coerced.get("plan"), list):
        coerced["plan"] = []

    action = coerced.get("action")
    if action not in ("answer", "command"):
        # Infer action from presence of a command
        cmd = coerced.get("command")
        if isinstance(cmd, str) and cmd.strip():
            coerced["action"] = "command"
        else:
            coerced["action"] = "answer"

    if not isinstance(coerced.get("expose_to_user"), bool):
        coerced["expose_to_user"] = True

    if coerced["action"] == "command":
        if not isinstance(coerced.get("command"), str):
            coerced["command"] = None
        # data is irrelevant for command mode; keep if present
        coerced.setdefault("data", None)
    else:
        # Ensure data is present; if absent, use the non-schema fields as payload
        if "data" not in coerced or coerced.get("data") is None:
            payload = {k: v for k, v in agent_json.items() if k not in schema_keys}
            coerced["data"] = payload if payload else coerced.get("data")
        coerced.setdefault("command", None)

    return coerced


def execute_command(command: str) -> dict:
    if not is_safe_command(command):
        return {"status": "rejected", "reason": "unsafe command"}

    if not isinstance(command, str) or not command.strip():
        return {"status": "rejected", "reason": "missing command"}

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True
        )
        return {
            "status": "executed",
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def process_agent_output(agent_json: dict) -> dict:
    """
    Enforces planning + exposure rules
    """
    agent_json = _coerce_agent_json(agent_json)
    if "error" in agent_json:
        return agent_json

    action = agent_json.get("action")
    expose = agent_json.get("expose_to_user", False)

    result = {
        "plan": agent_json["plan"],
        "expose_to_user": expose
    }

    if action == "command":
        cmd = agent_json.get("command")
        result["execution"] = execute_command(cmd)

    elif action == "answer":
        result["data"] = agent_json.get("data")

    return result


def ask(prompt: str) -> dict:
    """Convenience API: run the agent once and return processed output."""
    agent_output = ask_model(prompt)
    return process_agent_output(agent_output)

# =========================
# Entry point
# =========================
def main():
    user_prompt = input(">> ")

    agent_output = ask_model(user_prompt)
    final = process_agent_output(agent_output)

    # Always log full agent output internally
    print("\n[INTERNAL]")
    print(json.dumps(final, indent=2))

    # Only expose when allowed
    if final.get("expose_to_user"):
        print("\n[USER]")
        if "data" in final:
            print(json.dumps(final["data"], indent=2))
        elif "execution" in final:
            print(json.dumps(final["execution"], indent=2))
    else:
        print("\n[USER]")
        print("(no user-visible response)")

if __name__ == "__main__":
    main()
