"""Scratch module for testing prompt variants.

Edit USER_TEMPLATE freely and re-run the pipeline with `--prompt-style experimental`
(or `prompt_style: experimental` in the YAML config) — no code changes needed.
"""

SYSTEM_PROMPT = ""

USER_TEMPLATE = """\
Design a reward function for a quadcopter hover-and-navigation task in IsaacLab DirectRLEnv.
Steer the quadcopter to a 3D target position while discouraging aggressive motion and keeping the reward bounded.

Available attributes:
self._robot.data.root_pos_w 
self._desired_pos_w 
self.step_dt 

Write the complete reward function for the from_scratch variant.
Return ONLY the raw Python code. Do not include explanations or markdown.
"""
