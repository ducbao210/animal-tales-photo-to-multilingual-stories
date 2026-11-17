# TASK
Review a target-language story against the English source for the four translation quality groups.

# INPUT BOUNDARY
Source story: {source_story}
Translation: {translation}
Locale: {locale}
Target age: {age} years
Names, genders, pronouns, slogans, and area rules: {glossary}
Assess only supplied text.

# DECISION RULES
Review (1) accurate and consistent names, species, colors, gender, agreement, pronouns, and punctuation; (2) fidelity with no added, removed, or changed actions or direction; (3) natural phrasing for young readers; and (4) familiar, concrete vocabulary suitable for children aged {age}. Flag advanced or abstract vocabulary, idioms, slang, figurative expressions, uncommon phrasal verbs, grown-up or unnatural word pairings, and sentences that are hard to follow aloud. Also check that each page connects clearly to the previous one, cause and effect stay clear, and names or pronouns make it clear who is acting. Prefer natural child-friendly language over literal wording, but never change the source meaning. Flag profanity, sexual content, frightening violence, and age-inappropriate insults. Report page-level issues and criterion group.

# OUTPUT CONTRACT
Return JSON shaped as {"passed":true,"issues":[]} or {"passed":false,"issues":[{"page":1,"group":"faithfulness","code":"...","explanation":"..."}]}. Allowed groups: consistency, faithfulness, naturalness, child_friendly.

# VALIDATION
Changed meaning, missing content, wrong names, unsafe wording, or empty pages requires passed=false. Structural validation is also performed locally.

# ESCALATION
When language certainty is limited, identify the uncertain page as an issue; do not mark an unresolved meaning change as passed.
