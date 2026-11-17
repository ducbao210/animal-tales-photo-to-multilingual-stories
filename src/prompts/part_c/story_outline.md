# TASK
Plan a short English children's story outline for children aged {age} before drafting its pages.

# INPUT BOUNDARY
Main character: {main_profile}
Supporting character: {support_profile}
Location: {location}
Available areas: {areas}
Main species and color: {species}, {color}
Use only these two characters and the supplied garden areas. Do not add visitors, narrators who act, or off-screen helpers.

# DECISION RULES
Plan a story that a child aged {age} can understand when it is read aloud. Use familiar everyday words and concrete actions. Avoid idioms, slang, figurative expressions, abstract wording, and grown-up or uncommon word pairings. Keep ideas in a clear cause-and-effect order: the problem leads to the next action, and each action leads naturally to what happens next. Make the reason for every move between areas easy to understand. Keep the characters, their goals, and the problem consistent from beginning to end. Return a clear beginning, problem, solution, and happy ending. Assign a specific causal contribution to each character's skill. Plan a moment for each character to speak their own slogan. Keep the problem gentle and safe for children aged {age}.

# OUTPUT CONTRACT
Return JSON with keys problem, area_sequence, move_reasons, main_skill_solution, support_skill_solution, slogan_moments, and four_parts. area_sequence contains area IDs; four_parts contains the four named sections.

# VALIDATION
Every area ID must be supplied. There must be at least two areas, one move reason, one skill contribution per character, and one spoken slogan moment per character. Invalid JSON or a missing causal step is rejected.

# ESCALATION
If the supplied profiles or areas do not support a coherent safe story, return a JSON issue describing the missing input instead of inventing one.
