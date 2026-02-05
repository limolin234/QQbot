"""NapCat WebSocket 客户端"""
import json
import asyncio
import websockets
from typing import Optional, Dict, Any, AsyncIterator
from loguru import logger
from config.napcat_config import NAPCAT_WS_URL


class NapCatClient:
    """NapCat WebSocket 客户端"""

    def __init__(self, ws_url: str = NAPCAT_WS_URL):
        """
        初始化 NapCat 客户端

        Args:
            ws_url: WebSocket 服务器地址
        """
        self.ws_url = ws_url
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.bot_qq: Optional[int] = None
        self.connected = False

        # API 响应队列（用于匹配 echo）
        self.api_responses: Dict[str, asyncio.Future] = {}

        # 事件消息队列（在初始化时创建，避免消息丢失）
        self._event_queue: asyncio.Queue = asyncio.Queue()

        # 消息接收任务
        self.receive_task: Optional[asyncio.Task] = None

    async def connect(self) -> bool:
        """
        连接到 NapCat WebSocket 服务器

        Returns:
            True 如果连接成功
        """
        try:
            logger.info(f"正在连接到 NapCat WebSocket: {self.ws_url}")
            self.websocket = await websockets.connect(self.ws_url)
            self.connected = True
            logger.success("WebSocket 连接成功")

            # 启动消息接收任务
            self.receive_task = asyncio.create_task(self._receive_messages())

            # 等待一下让接收任务启动
            await asyncio.sleep(0.1)

            # 获取 bot 的 QQ 号
            await self._fetch_bot_qq()

            return True
        except Exception as e:
            logger.error(f"WebSocket 连接失败: {e}")
            self.connected = False
            return False

    async def _receive_messages(self):
        """后台接收消息任务"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    logger.debug(f"收到消息: {data}")

                    # 如果是 API 响应（包含 echo 字段）
                    if "echo" in data:
                        echo = data["echo"]
                        if echo in self.api_responses:
                            # 将响应传递给等待的 Future
                            self.api_responses[echo].set_result(data)
                    # 如果是事件消息（包含 post_type）
                    elif "post_type" in data:
                        # 事件消息会在 listen() 中处理
                        logger.debug(f"收到事件消息: post_type={data.get('post_type')}, message_type={data.get('message_type')}")
                        await self._event_queue.put(data)
                        logger.debug(f"事件消息已加入队列，当前队列大小: {self._event_queue.qsize()}")

                except json.JSONDecodeError as e:
                    logger.error(f"JSON 解析失败: {e}")
                except Exception as e:
                    logger.error(f"处理消息异常: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket 连接已关闭")
            self.connected = False
        except Exception as e:
            logger.error(f"接收消息异常: {e}")
            self.connected = False

    async def _fetch_bot_qq(self):
        """获取 bot 的 QQ 号"""
        try:
            login_info = await self.call_api("get_login_info")
            if login_info and "user_id" in login_info:
                self.bot_qq = login_info["user_id"]
                logger.info(f"Bot QQ 号: {self.bot_qq}")
            else:
                logger.warning("无法获取 Bot QQ 号")
        except Exception as e:
            logger.error(f"获取 Bot QQ 号失败: {e}")

    async def call_api(self, action: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        调用 NapCat API

        Args:
            action: API 动作名称
            params: API 参数

        Returns:
            API 响应数据
        """
        if not self.websocket:
            logger.error("WebSocket 未连接")
            return None

        try:
            # 生成唯一的 echo
            echo = f"{action}_{asyncio.get_event_loop().time()}"

            # 创建 Future 用于等待响应
            future = asyncio.Future()
            self.api_responses[echo] = future

            # 构造请求
            request = {
                "action": action,
                "params": params or {},
                "echo": echo
            }

            # 发送请求
            await self.websocket.send(json.dumps(request))
            logger.debug(f"发送 API 请求: {action}")

            # 等待响应（最多 10 秒）
            try:
                response = await asyncio.wait_for(future, timeout=10.0)
            finally:
                # 清理
                if echo in self.api_responses:
                    del self.api_responses[echo]

            # 检查响应状态
            if response.get("status") == "ok":
                logger.debug(f"API 响应成功: {action}")
                return response.get("data")
            else:
                logger.error(f"API 响应失败: {response}")
                return None

        except asyncio.TimeoutError:
            logger.error(f"API 调用超时: {action}")
            if echo in self.api_responses:
                del self.api_responses[echo]
            return None
        except Exception as e:
            logger.error(f"API 调用失败: {action}, 错误: {e}")
            return None

    async def send_group_msg(self, group_id: int, message: str) -> bool:
        """
        发送群消息

        Args:
            group_id: 群号
            message: 消息内容

        Returns:
            True 如果发送成功
        """
        try:
            result = await self.call_api("send_group_msg", {
                "group_id": group_id,
                "message": message
            })

            if result:
                logger.info(f"消息发送成功 -> 群 {group_id}: {message[:50]}")
                return True
            else:
                logger.error(f"消息发送失败 -> 群 {group_id}")
                return False

        except Exception as e:
            logger.error(f"发送群消息异常: {e}")
            return False

    async def send_private_msg(self, user_id: int, message: str) -> bool:
        """
        发送私聊消息

        Args:
            user_id: 接收者 QQ 号
            message: 消息内容

        Returns:
            True 如果发送成功
        """
        try:
            result = await self.call_api("send_private_msg", {
                "user_id": user_id,
                "message": message
            })

            if result:
                logger.info(f"私聊消息发送成功 -> 用户 {user_id}: {message[:50]}")
                return True
            else:
                logger.error(f"私聊消息发送失败 -> 用户 {user_id}")
                return False

        except Exception as e:
            logger.error(f"发送私聊消息异常: {e}")
            return False

    async def listen(self) -> AsyncIterator[Dict[str, Any]]:
        """
        监听消息事件

        Yields:
            消息事件数据
        """
        if not self.websocket:
            logger.error("WebSocket 未连接")
            return

        logger.info("开始监听消息...")

        try:
            while self.connected:
                # 从队列中获取事件
                try:
                    event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                    yield event
                except asyncio.TimeoutError:
                    # 超时继续循环
                    continue

        except Exception as e:
            logger.error(f"监听消息异常: {e}")
            self.connected = False

    async def close(self):
        """关闭 WebSocket 连接"""
        self.connected = False

        # 取消接收任务
        if self.receive_task:
            self.receive_task.cancel()
            try:
                await self.receive_task
            except asyncio.CancelledError:
                pass

        # 关闭 WebSocket
        if self.websocket:
            await self.websocket.close()
            logger.info("WebSocket 连接已关闭")

    async def reconnect(self, max_retries: int = 5, delay: int = 5) -> bool:
        """
        重连 WebSocket

        Args:
            max_retries: 最大重试次数
            delay: 重试间隔（秒）

        Returns:
            True 如果重连成功
        """
        for i in range(max_retries):
            logger.info(f"尝试重连 ({i + 1}/{max_retries})...")
            if await self.connect():
                return True
            await asyncio.sleep(delay)

        logger.error("重连失败")
        return False
