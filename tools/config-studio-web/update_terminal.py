import re

file_path = "/home/yunzechen/Code/QQbot/tools/config-studio-web/src/TerminalPanel.tsx"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Fix the broken quotes
content = content.replace("buffer.split('\n');", "buffer.split('\\n');")
content = content.replace("lines.pop() || '';", "lines.pop() || '';") # this was broken if missing quotes
content = content.replace("...prev, '\n---", "...prev, '\\n---")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

