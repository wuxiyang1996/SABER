"""Cold-start trajectory collection using GPT-5 Mini.

Uses an external LLM (GPT-5 Mini via OpenAI API) to drive the attack
tool_sets against Pi0.5 in LIBERO, collecting successful attack
trajectories without any GRPO training.  The saved trajectories can
later warm-start the local attack agent's training.
"""
