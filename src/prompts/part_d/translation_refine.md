# TASK
Correct a translation using review issues while preserving the English source meaning.

# INPUT BOUNDARY
Translation: {translation}
English source: {source_story}
Issues: {issues}
Fixed glossary and locale rules: {glossary}
Target age: {age} years
Do not add facts or change the source story.

# DECISION RULES
Fix the listed consistency, faithfulness, naturalness, or child-friendly issue with the smallest edit. Keep all eight page IDs, names, actions, dialogue, and direction aligned with the source. Use natural language for children aged {age}: familiar everyday words, concrete wording, and short sentences that are easy to follow aloud. Avoid advanced or abstract vocabulary, idioms, slang, figurative expressions, uncommon phrasal verbs, and grown-up or unnatural word pairings. Connect each page clearly to the one before it and keep names or pronouns easy to follow. Prefer a natural child-friendly phrase over a stiff literal translation, while keeping the same meaning and facts.

# OUTPUT CONTRACT
Return the complete replacement JSON with eight pages. Do not include commentary or Markdown fences.

# VALIDATION
The caller reruns page alignment, glossary, locale, banned-word, and semantic checks. A failed replacement uses one correction round.

# ESCALATION
If a source phrase cannot be translated confidently, preserve its closest supported meaning and leave a review issue; the caller stops after the configured round limit.
