# TASK
Translate the complete English story into {locale} for children aged {age}.

# INPUT BOUNDARY
Source story: {source_story}
Name list: {names}
Character genders: {genders}
Pronouns: {pronouns}
Adapted slogans: {slogans}
Area names: {areas}
Locale punctuation notes: {punctuation_notes}
Examples: {examples}
Translate only the supplied pages. Keep all names exactly as listed and preserve the source's facts and actions. Use safe, natural language for children aged {age}. Choose familiar everyday words and concrete wording that a young child can understand when read aloud. Avoid advanced or abstract vocabulary, idioms, slang, figurative expressions, uncommon phrasal verbs, and grown-up or unnatural word pairings. Do not translate word-for-word if that makes the result sound stiff; use a natural child-friendly phrase that keeps the same meaning. Avoid profanity, sexual content, frightening violence, and age-inappropriate insults.

# DECISION RULES
Keep the same eight page IDs and translate every page. Use natural, simple language a child aged {age} would hear in a story read aloud. Keep sentences short and clear, with one main idea at a time. Make each page connect clearly to the previous one; preserve the story's order, cause and effect, and clear references to who is acting. Preserve both characters' species and colors, dialogue, and gender. Use each character's locale slogan where its English slogan is spoken. In Vietnamese, put the species noun before a character name where natural. French adjectives, verbs, pronouns, and number must agree; leave a narrow space before ! ? : ; when typography supports it. Spanish questions and exclamations use opening ¡ ¿ marks as appropriate. Keep Vietnamese pronouns stable.

# OUTPUT CONTRACT
Return JSON with exactly {"pages":[{"page":1,"text":"..."}]}; page IDs must match the source. No extra facts or translator notes.

# VALIDATION
The application checks page alignment, names, area mappings, empty pages, and selected punctuation patterns. GPT semantic review checks safety and that no action, direction, species, or color changed.

# ESCALATION
If a source passage is ambiguous, preserve the narrowest supported meaning and report the uncertainty in the review issues. Never resolve ambiguity by adding a fact.
