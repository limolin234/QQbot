"""进程管理模块 - 检测和管理 Bot 进程"""
import os
import sys
import psutil
from pathlib import Path
from loguru import logger


class ProcessManager:
    """Bot 进程管理器"""

    def __init__(self, pid_file: str = ".bot.pid"):
        """
        初始化进程管理器

        Args:
            pid_file: PID 文件路径
        """
        self.pid_file = Path(pid_file)

    def get_running_processes(self):
        """
        获取所有正在运行的 Bot 进程

        Returns:
            list: 进程列表 [(pid, cmdline, create_time), ...]
        """
        current_pid = os.getpid()
        running_processes = []

        for proc in psutil.process_iter(['pid', 'cmdline', 'create_time']):
            try:
                cmdline = proc.info['cmdline']
                if not cmdline:
                    continue

                # 检查是否是 Python 进程运行 main.py
                if len(cmdline) >= 2 and 'python' in cmdline[0].lower() and 'main.py' in cmdline[1]:
                    pid = proc.info['pid']
                    if pid != current_pid:  # 排除当前进程
                        running_processes.append((
                            pid,
                            ' '.join(cmdline),
                            proc.info['create_time']
                        ))
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

        return running_processes

    def kill_process(self, pid: int) -> bool:
        """
        杀死指定进程

        Args:
            pid: 进程 ID

        Returns:
            bool: 是否成功
        """
        try:
            proc = psutil.Process(pid)
            proc.terminate()  # 先尝试优雅关闭
            try:
                proc.wait(timeout=5)  # 等待 5 秒
            except psutil.TimeoutExpired:
                proc.kill()  # 强制杀死
            logger.info(f"已关闭进程: {pid}")
            return True
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.error(f"无法关闭进程 {pid}: {e}")
            return False

    def check_and_handle_duplicates(self) -> bool:
        """
        检查并处理重复进程

        Returns:
            bool: True 继续运行，False 退出
        """
        running_processes = self.get_running_processes()

        if not running_processes:
            logger.info("没有检测到其他 Bot 进程")
            return True

        # 显示检测到的进程
        print("\n" + "=" * 60)
        print("⚠️  检测到已有 Bot 进程正在运行：")
        print("=" * 60)

        for i, (pid, cmdline, create_time) in enumerate(running_processes, 1):
            from datetime import datetime
            start_time = datetime.fromtimestamp(create_time).strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n进程 {i}:")
            print(f"  PID: {pid}")
            print(f"  启动时间: {start_time}")
            print(f"  命令: {cmdline[:80]}...")

        print("\n" + "=" * 60)
        print("请选择操作：")
        print("  1. 关闭旧进程，启动新进程")
        print("  2. 保留旧进程，退出当前启动")
        print("  3. 全部关闭，启动新进程")
        print("  4. 忽略，继续启动（不推荐）")
        print("=" * 60)

        while True:
            try:
                choice = input("\n请输入选项 (1-4): ").strip()

                if choice == "1":
                    # 关闭第一个旧进程
                    pid, _, _ = running_processes[0]
                    if self.kill_process(pid):
                        print(f"✅ 已关闭旧进程 {pid}")
                        return True
                    else:
                        print(f"❌ 无法关闭进程 {pid}，请手动处理")
                        return False

                elif choice == "2":
                    # 退出当前启动
                    print("👋 保留旧进程，退出当前启动")
                    return False

                elif choice == "3":
                    # 关闭所有旧进程
                    success = True
                    for pid, _, _ in running_processes:
                        if not self.kill_process(pid):
                            success = False
                    if success:
                        print("✅ 已关闭所有旧进程")
                        return True
                    else:
                        print("❌ 部分进程无法关闭，请手动处理")
                        return False

                elif choice == "4":
                    # 忽略，继续启动
                    print("⚠️  忽略重复进程检查，继续启动（可能导致冲突）")
                    return True

                else:
                    print("❌ 无效选项，请输入 1-4")

            except KeyboardInterrupt:
                print("\n\n👋 用户取消，退出启动")
                return False
            except Exception as e:
                logger.error(f"处理用户输入异常: {e}")
                return False

    def save_pid(self):
        """保存当前进程 PID"""
        try:
            self.pid_file.write_text(str(os.getpid()))
            logger.debug(f"PID 已保存: {os.getpid()}")
        except Exception as e:
            logger.warning(f"无法保存 PID 文件: {e}")

    def remove_pid(self):
        """删除 PID 文件"""
        try:
            if self.pid_file.exists():
                self.pid_file.unlink()
                logger.debug("PID 文件已删除")
        except Exception as e:
            logger.warning(f"无法删除 PID 文件: {e}")
