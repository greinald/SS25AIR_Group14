import json

# Load the input JSON from a file
with open('./data/batch4_out.json', 'r') as f:
    data = json.load(f)

# Add an empty 'snippet' field to each question
for question in data.get('questions', []):
    question['snippets'] = []

# Save the modified JSON to a new file
with open('./data/batch4_out_snippets.json', 'w') as f:
    json.dump(data, f, indent=2)

print("Added empty 'snippets' field to all questions.")
