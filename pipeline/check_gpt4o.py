"""One-call check that GPT-4o works with the pipeline's settings (expansion prompt, JSON output)."""
from config import azure_client, chat_options
from llm_queries import PROMPTS

spec = PROMPTS["et_expand.v2"]
message = "Termin: Munasarjatsüsti punktsioon\nNCSP: Naissuguorganid > Munasari > Munasarja punktsioon"
response = azure_client().chat.completions.create(
    messages=[{"role": "system", "content": spec["system"]}, {"role": "user", "content": message}],
    max_completion_tokens=4000, **chat_options("gpt4o", "low"),
)
print(response.model, "| tokens in/out:", response.usage.prompt_tokens, response.usage.completion_tokens)
print(response.choices[0].message.content)
