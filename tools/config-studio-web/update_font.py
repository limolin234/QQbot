import re

file_path = "/home/yunzechen/Code/QQbot/tools/config-studio-web/src/TerminalPanel.tsx"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

content = content.replace('fontFamily: "Menlo, Monaco, Consolas, "Courier New", monospace",', 'fontFamily: "Menlo, Monaco, Consolas, \'Courier New\', monospace",')

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
