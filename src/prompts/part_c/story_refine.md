# TASK
Repair the English story using the supplied review issues.

# INPUT BOUNDARY
Draft: {draft}
Issues: {issues}
Fixed profiles and area glossary: {glossary}
Target age: {age} years
Preserve the same two characters, identities, species, colors, setting, and accepted facts.

# DECISION RULES
Fix every listed issue with the smallest change that works. Keep eight pages, the same page IDs, and the four sections. Use natural language for children aged {age}: familiar everyday words, concrete actions, and short sentences that are easy to follow aloud. Avoid idioms, slang, figurative expressions, abstract or uncommon words, uncommon phrasal verbs, and grown-up or unnatural word pairings. Keep events in a clear cause-and-effect order, connect each page naturally to the next, and make names and pronouns easy to follow. Do not make the language babyish. Keep each page within 40 words. Preserve the source facts. Do not introduce characters, actions, or places.

# OUTPUT CONTRACT
Return the complete replacement story JSON with title and pages. Do not return commentary or Markdown fences.

# VALIDATION
The application reruns structural checks and GPT safety and semantic review. A replacement that fails validation consumes one correction round.

# ESCALATION
If an issue cannot be corrected without inventing content, return the revised JSON with a clear issue for the next review; the caller fails after its configured correction limit.
