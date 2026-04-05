import re
with open('src/App.tsx', 'r') as f:
    text = f.read()

# Add dida_config back to FixedAgentKey
text = text.replace("type FixedAgentKey = 'summary_config' | 'forward_config' | 'auto_reply_config' | 'dida_agent_config';", "type FixedAgentKey = 'summary_config' | 'forward_config' | 'auto_reply_config' | 'dida_agent_config' | 'dida_config';")

# fix duplicate openSchedulerHelp
text = re.sub(r'const \[openSchedulerHelp, setOpenSchedulerHelp\] = useState\(false\);\n?\s*const \[openSchedulerHelp, setOpenSchedulerHelp\] = useState\(false\);', 'const [openSchedulerHelp, setOpenSchedulerHelp] = useState(false);', text)

with open('src/App.tsx', 'w') as f:
    f.write(text)
