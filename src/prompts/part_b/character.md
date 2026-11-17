# TASK
Create one clear full-body children's picture-book character image.

# INPUT BOUNDARY
Species: {species}
Coat color: {color}
Recognizing detail: {visual_detail}
Default pose: {pose}
Use only these requested animal and visual details. Do not add another animal, a human, writing, logos, or copyrighted character styling.

# DECISION RULES
Use a soft hand-drawn storybook illustration style. Keep the animal recognizable as the requested species, with the requested color as its dominant coat or fur color. Use a pale, simple, uncluttered background and centered composition. Show the complete animal and all limbs clearly. Avoid patterns or accessories that obscure the species or color.

# NEGATIVE PROMPT
Exclude: any other species, extra animals, humans, text, letters, logos, watermarks, copyrighted character designs, busy backgrounds, cropped bodies, hidden limbs, accessories or patterns that obscure the requested species or dominant coat color, and colors that change the animal's requested main coat color.

# OUTPUT CONTRACT
Return a single image only. Requested aspect ratio: {aspect_ratio}. Requested image size: {image_size}.

# VALIDATION
The caller checks that the returned bytes decode as a non-empty PNG and records actual dimensions, model, parameters, and prompt hash. Reject missing, blocked, or text-only output.

# ESCALATION
If the requested species or color cannot be followed, report generation failure. Never substitute a different species or color.
