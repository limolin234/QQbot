#!/usr/bin/env python3
"""
dida_agent intent_type 完整链路测试。

测试两条路径：
1. task_management: "提醒我明天开会" -> decision -> action -> reply
2. normal_conversation: "你好呀" -> decision -> reply (跳过 action)

Usage:
    python tools/replay_dida_agent.py
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path, override=True)

# ============================================================================
# Mock 外部依赖模块
# ============================================================================


# Mock ncatbot (Not needed for core logic testing, but kept for imports)
class _MockNcatbotCore:
    GroupMessage = MagicMock()
    PrivateMessage = MagicMock()


class _MockNcatbot:
    core = _MockNcatbotCore()
    utils = MagicMock()


sys.modules["ncatbot"] = _MockNcatbot()
sys.modules["ncatbot.core"] = _MockNcatbotCore()
sys.modules["ncatbot.utils"] = MagicMock()

# Mock bot (To bypass bot.api check)
mock_bot = MagicMock()
mock_bot.api = MagicMock()
sys.modules["bot"] = MagicMock()
sys.modules["bot"].bot = mock_bot

# Mock agent_pool
sys.modules["agent_pool"] = MagicMock()
sys.modules["agent_pool"].submit_agent_job = MagicMock(
    side_effect=lambda func, *args, **kwargs: (
        func(*args, **kwargs) if callable(func) else None
    )
)

# Mock dida_scheduler
mock_scheduler = MagicMock()


async def mock_execute_action_structured(*args, **kwargs):
    return {"ok": True, "message": "模拟执行成功"}


mock_scheduler.execute_action_structured = mock_execute_action_structured
sys.modules["workflows.dida.dida_scheduler"] = MagicMock()
sys.modules["workflows.dida.dida_scheduler"].dida_scheduler = mock_scheduler

# ============================================================================
# 现在可以安全导入了
# ============================================================================

from workflows.agent_config_loader import load_current_agent_config
from workflows.dida.dida_agent import (
    DidaAgentDecisionEngine,
    DidaAgentMessageContext,
)


def load_config():
    import yaml

    config_path = project_root / "workflows" / "agent_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        full_config = yaml.safe_load(f)

    if (
        "dida_agent_config" in full_config
        and "config" in full_config["dida_agent_config"]
    ):
        return full_config["dida_agent_config"]["config"]
    return full_config


async def test_task_management():
    """测试 1: task_management 链路"""
    print("\n" + "=" * 70)
    print("测试 1: task_management - 提醒我明天开会")
    print("=" * 70)

    config = load_config()
    engine = DidaAgentDecisionEngine(config)

    context = DidaAgentMessageContext(
        chat_type="group",
        group_id="499616852",
        user_id="2667221906",
        user_name="陈昀泽",
        ts=datetime.now().isoformat(),
        raw_message="[CQ:at,qq=3637401634] 提醒我明天开会",
        cleaned_message="提醒我明天开会",
        history_messages=[],
        run_id="test_task_management",
    )

    print(f"\n[Step 1] 调用 should_reply()...")
    result = engine.should_reply(context)

    print(f"\n[Step 1 结果]")
    print(f"  should_reply: {result.get('should_reply')}")
    print(f"  reason: {result.get('reason')}")
    print(f"  intent_type: {result.get('intent_type')}")

    if not result.get("should_reply"):
        print(f"\n[FAIL] should_reply=False，决策模块判定不需要回复")
        return False

    if result.get("intent_type") != "task_management":
        print(
            f"\n[FAIL] intent_type 不匹配，期望 task_management，实际 {result.get('intent_type')}"
        )
        return False

    rule = result.get("rule", {})

    print(f"\n[Step 2] intent_type=task_management，调用 generate_action()...")
    action_prompt = str(rule.get("action_prompt", "")).strip()
    action_result = engine.generate_action(
        action_prompt=action_prompt,
        context=context,
        rule=rule,
    )

    print(f"\n[Step 2 结果]")
    print(f"  action_result.need_clarification: {action_result.need_clarification}")
    print(
        f"  action_result.dida_actions: {len(action_result.dida_actions) if action_result.dida_actions else 0}"
    )

    execution_result = None
    if action_result.dida_actions:
        for i, act in enumerate(action_result.dida_actions):
            print(f"    action[{i}]: type={act.action_type}, title={act.title}")

        # 模拟执行阶段
        execution_result = {
            "ok": True,
            "message": "批量执行完成：成功 1 条，失败 0 条\n1. 任务 [明天开会] 已创建。",
            "success_count": 1,
            "failed_count": 0,
        }

    print(f"\n[Step 3] 调用 generate_final_reply()...")
    reply_prompt = str(rule.get("reply_prompt", "")).strip()
    reply_result = engine.generate_final_reply(
        reply_prompt=reply_prompt,
        context=context,
        action_result=action_result,
        execution_result=execution_result,
        rule=rule,
    )

    print(f"\n[Step 3 结果]")
    reply_preview = reply_result.reply_text[:150] if reply_result.reply_text else ""
    print(f"  reply_text: {reply_preview}")

    print(f"\n[PASS] task_management 链路完成")
    return True


async def test_normal_conversation():
    """测试 2: normal_conversation 链路"""
    print("\n" + "=" * 70)
    print("测试 2: normal_conversation - 你好呀")
    print("=" * 70)

    config = load_config()
    engine = DidaAgentDecisionEngine(config)

    context = DidaAgentMessageContext(
        chat_type="group",
        group_id="499616852",
        user_id="2667221906",
        user_name="陈昀泽",
        ts=datetime.now().isoformat(),
        raw_message="[CQ:at,qq=3637401634] 你好呀",
        cleaned_message="你好呀",
        history_messages=[],
        run_id="test_normal_conversation",
    )

    print(f"\n[Step 1] 调用 should_reply()...")
    result = engine.should_reply(context)

    if not result.get("should_reply"):
        print(f"  [DEBUG] rules count: {len(engine.rules)}")
        for i, rule in enumerate(engine.rules):
            print(
                f"  [DEBUG] rule {i}: chat_type={rule.get('chat_type')}, number={rule.get('number')}, enabled={rule.get('enabled')}"
            )
        print(
            f"  [DEBUG] context: chat_type={context.chat_type}, group_id={context.group_id}, user_id={context.user_id}"
        )

    print(f"\n[Step 1 结果]")
    print(f"  should_reply: {result.get('should_reply')}")
    print(f"  reason: {result.get('reason')}")
    print(f"  intent_type: {result.get('intent_type')}")

    if not result.get("should_reply"):
        print(f"\n[FAIL] should_reply=False，决策模块判定不需要回复")
        return False

    if result.get("intent_type") != "normal_conversation":
        print(
            f"\n[FAIL] intent_type 不匹配，期望 normal_conversation，实际 {result.get('intent_type')}"
        )
        return False

    rule = result.get("rule", {})

    # normal_conversation 跳过 action，直接调用 generate_final_reply
    print(
        f"\n[Step 2] intent_type=normal_conversation，跳过 action，直接调用 generate_final_reply()..."
    )
    normal_reply_prompt = str(rule.get("normal_conversation_reply_prompt", "")).strip()

    if not normal_reply_prompt:
        print(f"\n[FAIL] normal_conversation_reply_prompt 未配置")
        return False

    reply_result = engine.generate_final_reply(
        reply_prompt=normal_reply_prompt,
        context=context,
        action_result=None,
        execution_result=None,
        rule=rule,
    )

    print(f"\n[Step 2 结果]")
    reply_preview = reply_result.reply_text[:150] if reply_result.reply_text else ""
    print(f"  reply_text: {reply_preview}")

    print(f"\n[PASS] normal_conversation 链路完成")
    return True


async def main():
    print("=" * 70)
    print("dida_agent intent_type 完整链路测试")
    print("=" * 70)

    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        print("[ERROR] LLM_API_KEY not found in .env")
        return False

    base_url = os.getenv("LLM_API_BASE_URL", "https://api.minimaxi.com/v1")
    print(f"\n环境: API_BASE_URL={base_url}")
    print(f"      API_KEY={api_key[:10]}...")

    results = []

    passed1 = await test_task_management()
    results.append(("task_management", passed1))

    passed2 = await test_normal_conversation()
    results.append(("normal_conversation", passed2))

    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)

    all_passed = True
    for name, passed in results:
        print(f"  {'[PASS]' if passed else '[FAIL]'} {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n所有测试通过！")
    else:
        print("\n存在测试失败。")

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
