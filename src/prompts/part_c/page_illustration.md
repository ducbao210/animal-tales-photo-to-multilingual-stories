# TASK
Write one optional illustration prompt for page {page_id}.

# INPUT BOUNDARY
Page text: {page_text}
Character reference: {reference_description}
Character roster: {characters}
Characters present in this scene: {characters_present}
Action: {pose}
Emotion: {emotion}
Area: {area}
Use the reference character identity. Include exactly the listed characters present in this scene; if the support character first appears here, add that specified species and color. Do not add any other character or story fact.

# DECISION RULES
Describe a clear child-friendly composition matching the action, emotion, and area. Keep the main animal's species, coat color, and identifying details consistent with the reference.

# OUTPUT CONTRACT
Return one concise image prompt for this page only.

# VALIDATION
The caller stores one prompt per page and checks that it includes the page ID and reference identity.

# ESCALATION
If the page has no visual action, describe a calm scene that reflects only the supplied text.
