# TASK
Back-translate the completed {locale} story into English for a diagnostic comparison.

# INPUT BOUNDARY
Translation: {translation}
Original English source: {source_story}
Compare meaning only; do not rewrite the source.

# DECISION RULES
Return one English page per source page. Preserve names, actions, dialogue, direction, and emphasis as expressed by the translation. List differences by page, including omissions and additions.

# OUTPUT CONTRACT
Return JSON shaped as {"pages":[{"page":1,"text":"..."}],"differences":[{"page":1,"type":"...","explanation":"..."}]}. No additional story content.

# VALIDATION
Page IDs must match the translation. A difference is diagnostic and never silently edits the accepted English source.

# ESCALATION
If the target text is ambiguous, note that in differences and keep the back-translation conservative.
