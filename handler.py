import asyncio
from time import perf_counter

from scheduler import PriorityTask, TaskType
from bot import bot, QQnumber
from workflows.agent_observe import generate_run_id, observe_agent_event
from workflows.auto_reply import run_auto_reply_pipeline
from workflows.forward import run_forward_graph
from workflows.summary import (
    format_grouped_summary_message,
    format_grouped_summary_messages,
    format_summary_message,
    get_summary_send_mode,
    preprocess_summary_chunk,
    run_grouped_summary_graph,
    run_summary_graph,
)


async def handle_task(task: PriorityTask):
    if task.type == TaskType.SUMMARY:
        group_jobs = getattr(task.msg, "group_jobs", None)
        raw_message = getattr(task.msg, "raw_message", "") or ""

        try:
            if isinstance(group_jobs, list):
                grouped_result = await asyncio.to_thread(run_grouped_summary_graph, group_jobs)
                send_mode = get_summary_send_mode()
                if send_mode == "multi_message":
                    for message_text in format_grouped_summary_messages(grouped_result):
                        await bot.api.post_private_msg(QQnumber, text=message_text)
                    text = ""
                else:
                    text = format_grouped_summary_message(grouped_result)
                print(
                    "[SUMMARY] "
                    f"groups={len(grouped_result.group_results)} | "
                    f"chunks={grouped_result.chunk_count} | "
                    f"messages={grouped_result.message_count} | "
                    f"elapsed_ms={grouped_result.elapsed_ms:.2f}"
                )
            else:
                final_result = await asyncio.to_thread(run_summary_graph, raw_message)
                text = format_summary_message(final_result)
                print(
                    "[SUMMARY] "
                    f"overview={final_result.overview[:80]} | "
                    f"highlights={len(final_result.highlights)} | "
                    f"risks={len(final_result.risks)} | "
                    f"todos={len(final_result.todos)} | "
                    f"sources={len(final_result.sources)} | "
                    f"elapsed_ms={final_result.elapsed_ms:.2f}"
                )
            if text:
                await bot.api.post_private_msg(QQnumber, text=text)
        except Exception as error:
            prepared = preprocess_summary_chunk(raw_message)
            preview_lines = prepared.texts[:5]
            preview_text = "\n".join(preview_lines) if preview_lines else "(无可用文本)"
            debug_msg = (
                "[SUMMARY-FALLBACK]\n"
                f"reason={error}\n"
                f"input_lines={prepared.total_lines}, non_empty={prepared.non_empty_lines}, "
                f"unique={prepared.unique_lines}, blocks={prepared.block_count}, "
                f"output_chars={prepared.output_chars}, elapsed_ms={prepared.elapsed_ms:.2f}\n"
                "preview:\n"
                f"{preview_text}"
            )
            await bot.api.post_private_msg(QQnumber, text=debug_msg)
            print(debug_msg)

    elif task.type == TaskType.FROWARD:
        ts = str(getattr(task.msg, "ts", ""))
        group_id = str(getattr(task.msg, "group_id", ""))
        user_id = str(getattr(task.msg, "user_id", ""))
        user_name = str(getattr(task.msg, "user_name", ""))
        cleaned_message = str(
            getattr(task.msg, "cleaned_message", getattr(task.msg, "raw_message", ""))
        )
        try:
            result = await asyncio.to_thread(
                run_forward_graph,
                ts=ts,
                group_id=group_id,
                user_id=user_id,
                user_name=user_name,
                cleaned_message=cleaned_message,
            )
            if result.get("should_forward"):
                await bot.api.post_private_msg(QQnumber, text=str(result.get("forward_text", cleaned_message)))
            print(
                "[FORWARD] "
                f"group={group_id} user={user_id} should_forward={bool(result.get('should_forward'))} "
                f"reason={result.get('reason', '')}"
            )
        except Exception as error:
            print(f"[FORWARD-ERROR] group={group_id} user={user_id} error={error}")
    
    elif task.type == TaskType.AUTO_REPLY:
        ts = str(getattr(task.msg, "ts", ""))
        chat_type = str(getattr(task.msg, "chat_type", ""))
        group_id = str(getattr(task.msg, "group_id", ""))
        user_id = str(getattr(task.msg, "user_id", ""))
        user_name = str(getattr(task.msg, "user_name", ""))
        raw_message = str(getattr(task.msg, "raw_message", ""))
        cleaned_message = str(getattr(task.msg, "cleaned_message", raw_message))

        run_id = generate_run_id()
        started = perf_counter()
        observe_agent_event(
            agent_name="auto_reply",
            task_type="AUTO_REPLY",
            stage="start",
            run_id=run_id,
            chat_type=chat_type,
            group_id=group_id,
            user_id=user_id,
            user_name=user_name,
            ts=ts,
        )

        try:
            result = await asyncio.to_thread(
                run_auto_reply_pipeline,
                chat_type=chat_type,
                group_id=group_id,
                user_id=user_id,
                user_name=user_name,
                ts=ts,
                raw_message=raw_message,
                cleaned_message=cleaned_message,
                run_id=run_id,
            )
            elapsed_ms = (perf_counter() - started) * 1000
            observe_agent_event(
                agent_name="auto_reply",
                task_type="AUTO_REPLY",
                stage="end",
                run_id=run_id,
                chat_type=chat_type,
                group_id=group_id,
                user_id=user_id,
                user_name=user_name,
                ts=ts,
                latency_ms=elapsed_ms,
                decision={
                    "should_reply": bool(result.get("should_reply", False)),
                    "reason": str(result.get("reason", "")),
                    "matched_rule": result.get("matched_rule"),
                    "trigger_mode": result.get("trigger_mode", ""),
                    "reply_length": len(str(result.get("reply_text", "") or "")),
                },
            )
            should_reply_flag = bool(result.get("should_reply", False))
            reason_text = str(result.get("reason", ""))
            reply_text = str(result.get("reply_text", "") or "").strip()

            if should_reply_flag and reply_text:
                observe_agent_event(
                    agent_name="auto_reply",
                    task_type="AUTO_REPLY",
                    stage="send_start",
                    run_id=run_id,
                    chat_type=chat_type,
                    group_id=group_id,
                    user_id=user_id,
                    user_name=user_name,
                    ts=ts,
                    extra={"reply_length": len(reply_text)},
                )
                try:
                    if chat_type == "group":
                        await bot.api.post_group_msg(group_id, text=reply_text)
                    else:
                        await bot.api.post_private_msg(user_id, text=reply_text)
                    observe_agent_event(
                        agent_name="auto_reply",
                        task_type="AUTO_REPLY",
                        stage="send_end",
                        run_id=run_id,
                        chat_type=chat_type,
                        group_id=group_id,
                        user_id=user_id,
                        user_name=user_name,
                        ts=ts,
                        decision={"sent": True, "reply_length": len(reply_text)},
                    )
                except Exception as send_error:
                    observe_agent_event(
                        agent_name="auto_reply",
                        task_type="AUTO_REPLY",
                        stage="send_error",
                        run_id=run_id,
                        chat_type=chat_type,
                        group_id=group_id,
                        user_id=user_id,
                        user_name=user_name,
                        ts=ts,
                        error=str(send_error),
                    )
                    print(
                        "[AUTO_REPLY-SEND-ERROR] "
                        f"chat={chat_type} group={group_id} user={user_id} error={send_error}"
                    )
            elif should_reply_flag:
                observe_agent_event(
                    agent_name="auto_reply",
                    task_type="AUTO_REPLY",
                    stage="send_skip",
                    run_id=run_id,
                    chat_type=chat_type,
                    group_id=group_id,
                    user_id=user_id,
                    user_name=user_name,
                    ts=ts,
                    decision={"sent": False, "reason": "reply_text_empty"},
                )

            print(
                "[AUTO_REPLY] "
                f"chat={chat_type} group={group_id} user={user_name}({user_id}) "
                f"should_reply={should_reply_flag} "
                f"reason={reason_text} reply_len={len(reply_text)}"
            )
        except Exception as error:
            elapsed_ms = (perf_counter() - started) * 1000
            observe_agent_event(
                agent_name="auto_reply",
                task_type="AUTO_REPLY",
                stage="error",
                run_id=run_id,
                chat_type=chat_type,
                group_id=group_id,
                user_id=user_id,
                user_name=user_name,
                ts=ts,
                latency_ms=elapsed_ms,
                error=str(error),
            )
            print(
                "[AUTO_REPLY-ERROR] "
                f"chat={chat_type} group={group_id} user={user_id} error={error}"
            )
    elif task.type == TaskType.GROUPNOTE:
        print("回复")
    else:
        return
