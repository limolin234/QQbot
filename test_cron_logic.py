
import aiocron
import asyncio
from zoneinfo import ZoneInfo
from datetime import datetime

def _normalize_cron_expression(raw: str) -> str:
    expression = (raw or "").strip()
    if not expression:
        return ""
    parts = expression.split()
    if len(parts) >= 6:
        return " ".join(parts[1:6])
    return expression

async def test_cron():
    expression = "00 08 * * *"
    norm_expression = _normalize_cron_expression(expression)
    tz = ZoneInfo("Asia/Shanghai")
    
    # 我们不运行它，只看它的逻辑
    print(f"Original expression: '{expression}'")
    print(f"Normalized expression: '{norm_expression}'")
    
    # 模拟 aiocron 初始化逻辑
    cron = aiocron.crontab(norm_expression, tz=tz)
    # 获取下一次运行时间
    next_run = await cron.next()
    print(f"Next run (timestamp): {next_run}")
    print(f"Next run (local time in Shanghai): {datetime.fromtimestamp(next_run, tz=tz)}")

if __name__ == "__main__":
    asyncio.run(test_cron())
