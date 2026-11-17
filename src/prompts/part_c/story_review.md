# TASK
Review the English story for safety and rubric requirements.

# INPUT BOUNDARY
Story: {story}
Profiles: {profiles}
Allowed areas: {areas}
Target age: {age} years
Judge only the supplied story and requirements. Do not add story content.

# DECISION RULES
Check that the story is suitable for children aged {age}. It should use familiar everyday words, concrete actions, and short sentences that sound natural when read aloud. Flag idioms, slang, figurative expressions, abstract or uncommon words, uncommon phrasal verbs, grown-up or unnatural word pairings, and sentences with too many clauses. Do not demand babyish language; natural child-friendly wording is the goal. Check that events are in a clear cause-and-effect order, each page follows naturally from the previous one, transitions are easy to understand, and names or pronouns make it clear who is acting. Flag jumps, confusing references, or details that change without explanation. Keep the problem, emotions, and resolution gentle. Flag frightening violence, profanity, sexual content, or age-inappropriate insults.

Check exactly the two supplied characters, including their names, species, and colors; at least two supplied areas and an understandable reason for every move; each character saying their own slogan; and each character's skill making a specific causal contribution to solving the problem. Check eight pages with no more than 40 words per page, the four sections in order (beginning, problem, solution, happy ending), and a satisfying happy ending. List language, coherence, cohesion, and semantic issues by page where possible.

# OUTPUT CONTRACT
Return JSON exactly shaped as {"passed":true,"issues":[]} or {"passed":false,"issues":[{"page":1,"code":"...","explanation":"..."}]}. Use null for a story-wide page.

# VALIDATION
Any safety issue or missing requirement means passed=false. The application combines this review with deterministic checks; your passed flag cannot override those checks.

# ESCALATION
If the story contains unsafe content, identify it and fail the review. Never repeat more of the unsafe text than needed to identify the issue.
