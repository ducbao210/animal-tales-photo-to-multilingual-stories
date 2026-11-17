# TASK
Create one visual plan for each story phase: beginning, problem, solution, and happy_ending.

# INPUT BOUNDARY
Story pages: {story_pages}
Allowed characters: {characters}
Allowed areas: {areas}
Each phase has exactly two consecutive pages. Use only the supplied characters and areas.

# DECISION RULES
Summarize the phase as one imageable moment. Select the characters who are physically present in that moment; use the story text rather than assuming both are always present. Keep actions, expressions, and setting faithful to the pages. Use a distinct story-driven pose and expression. Never add another animal, person, or object that changes the story.

# OUTPUT CONTRACT
Return JSON with a plans array containing exactly four items in phase order. Each item has section, page_ids, area_id, scene, action, emotion, and character_ids. character_ids may contain only the supplied IDs. area_id must be one of the supplied areas.

# VALIDATION
Every story phase must appear once, with its two corresponding page IDs. The application supplies the fixed species, colors, and visual details for each selected character; do not infer or alter them.
