import re
with open('src/App.tsx', 'r') as f:
    text = f.read()

text = re.sub(r'</FormModal>\s*</div>\s*\)\s*:\s*\(', r'</FormModal>\n                            </>\n                        ) : (', text)
with open('src/App.tsx', 'w') as f:
    f.write(text)
