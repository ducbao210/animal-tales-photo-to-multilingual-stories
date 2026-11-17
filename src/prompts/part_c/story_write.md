# TASK
Write the eight-page English story from the approved outline for children aged {age}.

# INPUT BOUNDARY
Outline: {outline}
Main profile: {main_profile}
Supporting profile: {support_profile}
Character image metadata: {character_metadata}
Area names and descriptions: {area_names}
Main species and color: {species}, {color}
Page limit: {page_limit} words per page
Target age: {age}
Only the two named characters may act or speak. Do not introduce other animals, people, voices, or new story facts.

# DECISION RULES
Write for children aged {age}, using familiar everyday words, concrete actions, and short sentences that sound natural when read aloud. Use one clear idea at a time. Avoid idioms, slang, figurative expressions, abstract words, uncommon phrasal verbs, and grown-up or unusual word pairings. Avoid long sentences with several clauses. Do not make the language babyish; make it warm, natural, and easy to understand.

Keep the story easy to follow across all eight pages. Put events in a clear order. Make each page connect to the page before it, using simple links such as what happened next or why a character acted. Keep names and pronouns clear, and do not make the reader guess who is acting. Make the problem cause the characters' actions, and make those actions lead to the solution. Keep each character, place, and story fact consistent. Give a clear reason for every move between areas.

Make each character say their own slogan in dialogue and use their stated skill to solve part of the problem. Name both characters' species and colors in the story. Use at least two selected areas. Keep each page to at most 40 whitespace-separated words.

# OUTPUT CONTRACT
Return JSON: {"title":"...","pages":[{"page":1,"section":"beginning","area_id":"...","text":"..."}]}. Include pages 1–8, exactly two pages in each of beginning, problem, solution, happy_ending.

# VALIDATION
The application checks page IDs, section order, word counts, selected area IDs, both names, species/colors, slogans, and required fields. GPT reviews child safety. Do not wrap JSON in Markdown.

# ESCALATION
If the outline is incomplete, return a concise validation issue. Do not silently repair it by adding characters or facts.
