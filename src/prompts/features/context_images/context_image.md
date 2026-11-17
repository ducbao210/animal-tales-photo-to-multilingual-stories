# TASK
Generate one story illustration for the {section} phase.

# INPUT BOUNDARY
Phase story: {phase_text}
Visual plan: {visual_plan}
Character roster and fixed appearances: {characters}
Characters required in this image: {characters_present}
Area: {area}
Consistency guidance: {consistency_guidance}

# DECISION RULES
Use the fixed character metadata to recreate each character consistently in the soft hand-drawn children's picture-book style. Include exactly the listed characters, with their specified species, dominant coat colors, names, and identifying details. If a character is not listed, do not include it. Match the phase action, emotion, and area; do not add text, labels, or story events.

# OUTPUT CONTRACT
Return one image only. Preserve the fixed character appearances and illustration style described in the prompt.

# ESCALATION
If the scene cannot be represented without changing the story or character identities, report generation failure.
