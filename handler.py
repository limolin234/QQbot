import asyncio

from scheduler import PriorityTask, TaskType
from bot import bot, QQnumber
from workflows.summary import (
    format_summary_message,
    preprocess_summary_chunk,
    run_summary_graph,
)


async def handle_task(task: PriorityTask):
    if task.type == TaskType.SUMMARY:
        raw_message = getattr(task.msg, "raw_message", "") or ""

        try:
            final_result = await asyncio.to_thread(run_summary_graph, raw_message)
            text = format_summary_message(final_result)
            await bot.api.post_private_msg(QQnumber, text=text)
            print(
                "[SUMMARY] "
                f"overview={final_result.overview[:80]} | "
                f"highlights={len(final_result.highlights)} | "
                f"risks={len(final_result.risks)} | "
                f"todos={len(final_result.todos)} | "
                f"sources={len(final_result.sources)} | "
                f"elapsed_ms={final_result.elapsed_ms:.2f}"
            )
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
        await bot.api.post_private_msg(QQnumber, text=task.msg.raw_message)
        print(task.msg.raw_message)
    elif task.type == TaskType.GROUPNOTE:
        print("回复")
    else:
        return
