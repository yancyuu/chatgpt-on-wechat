# access LazyAI knowledge base platform
# docs: https://doc.fastgpt.in/docs/development/openapi/auth/

import re
import os
import time
import requests
import config
from bot.bot import Bot
from bot.chatgpt.chat_gpt_session import ChatGPTSession
from bot.session_manager import SessionManager
from bridge.context import Context, ContextType
from bridge.reply import Reply, ReplyType
from common.log import logger
from config import conf, pconf
import threading
from common import memory, utils
import base64


class LazyAIBot(Bot):
    # authentication failed
    AUTH_FAILED_CODE = 401
    NO_QUOTA_CODE = 406

    def __init__(self):
        super().__init__()
        self.args = {}
        self.sessions = LazyAISessionManager(LazyAISession, model=conf().get("model") or "gpt-3.5-turbo")


    def reply(self, query, context: Context = None) -> Reply:
        if context.type == ContextType.TEXT:
            return self._chat(query, context)
        elif context.type == ContextType.IMAGE:
            # 文件处理
            context.get("msg").prepare()
            file_path = context.content
            if not self.check_file(file_path, self.sum_config):
                return
            return self.summary_pic(file_path)
        elif context.type == ContextType.IMAGE_CREATE:
            if not conf().get("text_to_image"):
                logger.warn("[LazyAI] text_to_image is not enabled, ignore the IMAGE_CREATE request")
                return Reply(ReplyType.TEXT, "")
            ok, res = self.create_img(query, 0)
            if ok:
                reply = Reply(ReplyType.IMAGE_URL, res)
            else:
                reply = Reply(ReplyType.ERROR, res)
            return reply
        else:
            reply = Reply(ReplyType.ERROR, "Bot不支持处理{}类型的消息".format(context.type))
            return reply
        
    def check_file(self, file_path: str, sum_config: dict) -> bool:
        file_size = os.path.getsize(file_path) // 1000
        if (sum_config.get("max_file_size") and file_size > sum_config.get("max_file_size")) or file_size > 15000:
            logger.warn(f"[LinkSum] file size exceeds limit, No processing, file_size={file_size}KB")
            return False

        suffix = file_path.split(".")[-1]
        support_list = ["txt", "csv", "docx", "pdf", "md", "jpg", "jpeg", "png"]
        if suffix not in support_list:
            logger.warn(f"[LinkSum] unsupported file, suffix={suffix}, support_list={support_list}")
            return False

        return True
    
    def summary_pic(self, file_path, content):
        
        file_body = f'img-block
        {"src":"{file_path}"}'
        return self._chat(file_body, content)

    def _chat(self, query, context, retry_count=0) -> Reply:
        """
        发起对话请求
        :param query: 请求提示词
        :param context: 对话上下文
        :param retry_count: 当前递归重试次数
        :return: 回复
        """
        if retry_count > 2:
            # exit from retry 2 times
            logger.warn("[LazyAI] failed after maximum number of retry times")
            return Reply(ReplyType.TEXT, "请再问我一次吧")

        try:
            lazyai_api_key = conf().get("lazyai_api_key")
            session_id = context["session_id"]
            session_message = self.sessions.session_msg_query(query, session_id)
            logger.debug(f"[LazyAI] session={session_message}, session_id={session_id}")

            # image process
            img_cache = memory.USER_IMAGE_CACHE.get(session_id)
            if img_cache:
                messages = self._process_image_msg(session_id=session_id, query=query, img_cache=img_cache)
                if messages:
                    session_message = messages

            model = conf().get("model")
            # remove system message
            if session_message[0].get("role") == "system":
                if model == "wenxin":
                    session_message.pop(0)
            logger.debug(f"[LazyAI] session_id={session_id}")
            body = {
                "chatId" : session_id,
                "messages": [
                    {
                        "content": query,
                        "role": "user"
                    },
                ],
                "detail": True,
            }
            logger.info(f"[LazyAI] query={query}")
            headers = {"Authorization": "Bearer " + lazyai_api_key}

            # do http request
            base_url = conf().get("lazyai_api_base", "https://api.Lazy-ai.chat")
            res = requests.post(url=base_url + "/v1/chat/completions", json=body, headers=headers,
                                timeout=conf().get("request_timeout", 180))
            response = res.json()
            logger.info(f"[LazyAI] response={response}")
            if res.status_code == 200:
                # execute success
                reply_content = response["choices"][0]["message"]["content"]
                total_tokens = response["usage"]["total_tokens"]
                logger.info(f"[LazyAI] reply={reply_content}, total_tokens={total_tokens}")
                reply_content = self._process_url(reply_content)
                return Reply(ReplyType.TEXT, reply_content)
            else:
                error = response.get("error")
                logger.error(f"[LazyAI] chat failed, status_code={res.status_code}, "
                             f"msg={error.get('message')}, type={error.get('type')}")

                if res.status_code >= 500:
                    # server error, need retry
                    time.sleep(2)
                    logger.warn(f"[LazyAI] do retry, times={retry_count}")
                    return self._chat(query, content, retry_count + 1)

                return Reply(ReplyType.TEXT, "提问太快啦，请休息一下再问我吧")

        except Exception as e:
            logger.exception(e)
            # retry
            time.sleep(2)
            logger.warn(f"[LazyAI] do retry, times={retry_count}")
            return self._chat(query, content, retry_count + 1)

    def _build_vision_msg(self, query: str, path: str):
        try:
            suffix = utils.get_path_suffix(path)
            with open(path, "rb") as file:
                base64_str = base64.b64encode(file.read()).decode('utf-8')
                messages = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": query
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{suffix};base64,{base64_str}"
                            }
                        }
                    ]
                }]
                return messages
        except Exception as e:
            logger.exception(e)

    def reply_text(self, session: ChatGPTSession, app_code="", retry_count=0) -> dict:
        if retry_count >= 2:
            # exit from retry 2 times
            logger.warn("[LazyAI] failed after maximum number of retry times")
            return {
                "total_tokens": 0,
                "completion_tokens": 0,
                "content": "请再问我一次吧"
            }

        try:
            body = {
                "app_code": app_code,
                "messages": session.messages,
                "model": conf().get("model") or "gpt-3.5-turbo",  # 对话模型的名称, 支持 gpt-3.5-turbo, gpt-3.5-turbo-16k, gpt-4, wenxin, xunfei
                "temperature": conf().get("temperature"),
                "top_p": conf().get("top_p", 1),
                "frequency_penalty": conf().get("frequency_penalty", 0.0),  # [-2,2]之间，该值越大则更倾向于产生不同的内容
                "presence_penalty": conf().get("presence_penalty", 0.0),  # [-2,2]之间，该值越大则更倾向于产生不同的内容
            }
            if self.args.get("max_tokens"):
                body["max_tokens"] = self.args.get("max_tokens")
            headers = {"Authorization": "Bearer " +  conf().get("lazyai_api_key")}

            # do http request
            base_url = conf().get("lazyai_api_base", "https://api.lazygpt.cn")
            res = requests.post(url=base_url + "/v1/chat/completions", json=body, headers=headers,
                                timeout=conf().get("request_timeout", 180))
            if res.status_code == 200:
                # execute success
                response = res.json()
                reply_content = response["choices"][0]["message"]["content"]
                total_tokens = response["usage"]["total_tokens"]
                logger.info(f"[LazyAI] reply={reply_content}, total_tokens={total_tokens}")
                return {
                    "total_tokens": total_tokens,
                    "completion_tokens": response["usage"]["completion_tokens"],
                    "content": reply_content,
                }

            else:
                response = res.json()
                error = response.get("error")
                logger.error(f"[LazyAI] chat failed, status_code={res.status_code}, "
                             f"msg={error.get('message')}, type={error.get('type')}")

                if res.status_code >= 500:
                    # server error, need retry
                    time.sleep(2)
                    logger.warn(f"[LazyAI] do retry, times={retry_count}")
                    return self.reply_text(session, app_code, retry_count + 1)

                return {
                    "total_tokens": 0,
                    "completion_tokens": 0,
                    "content": "提问太快啦，请休息一下再问我吧"
                }

        except Exception as e:
            logger.exception(e)
            # retry
            time.sleep(2)
            logger.warn(f"[LazyAI] do retry, times={retry_count}")
            return self.reply_text(session, app_code, retry_count + 1)

    def create_img(self, query, retry_count=0, api_key=None):
        '''画画还是用LazyAI'''
        try:
            logger.info("[LazyImage] image_query={}".format(query))
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {conf().get('LazyAI_api_key')}"
            }
            data = {
                "prompt": query,
                "n": 1,
                "model": conf().get("text_to_image") or "dall-e-2",
                "response_format": "url",
                "img_proxy": conf().get("image_proxy")
            }
            url = conf().get("link_api_base", "https://api.link-ai.chat") + "/v1/images/generations"
            res = requests.post(url, headers=headers, json=data, timeout=(5, 90))
            t2 = time.time()
            image_url = res.json()["data"][0]["url"]
            logger.info("[OPEN_AI] image_url={}".format(image_url))
            return True, image_url

        except Exception as e:
            logger.error(format(e))
            return False, "画图出现问题，请休息一下再问我吧"

    def _process_url(self, text):
        try:
            url_pattern = re.compile(r'\[(.*?)\]\((http[s]?://.*?)\)')
            def replace_markdown_url(match):
                return f"{match.group(2)}"
            return url_pattern.sub(replace_markdown_url, text)
        except Exception as e:
            logger.error(e)
    
    def _process_image_msg(self, session_id: str, query:str, img_cache: dict):
        try:
            msg = img_cache.get("msg")
            path = img_cache.get("path")
            msg.prepare()
            logger.info(f"[LazyAI] query with images, path={path}")
            messages = self._build_vision_msg(query, path)
            memory.USER_IMAGE_CACHE[session_id] = None
            return messages
        except Exception as e:
            logger.exception(e)

class LazyAISessionManager(SessionManager):
    def session_msg_query(self, query, session_id):
        session = self.build_session(session_id)
        messages = session.messages + [{"role": "user", "content": query}]
        return messages

    def session_reply(self, reply, session_id, total_tokens=None, query=None):
        session = self.build_session(session_id)
        if query:
            session.add_query(query)
        session.add_reply(reply)
        try:
            max_tokens = conf().get("conversation_max_tokens", 2500)
            tokens_cnt = session.discard_exceeding(max_tokens, total_tokens)
            logger.debug(f"[LazyAI] chat history, before tokens={total_tokens}, now tokens={tokens_cnt}")
        except Exception as e:
            logger.warning("Exception when counting tokens precisely for session: {}".format(str(e)))
        return session

class LazyAISessionManager(SessionManager):
    def session_msg_query(self, query, session_id):
        session = self.build_session(session_id)
        messages = session.messages + [{"role": "user", "content": query}]
        return messages

    def session_reply(self, reply, session_id, total_tokens=None, query=None):
        session = self.build_session(session_id)
        if query:
            session.add_query(query)
        session.add_reply(reply)
        try:
            max_tokens = conf().get("conversation_max_tokens", 2500)
            tokens_cnt = session.discard_exceeding(max_tokens, total_tokens)
            logger.debug(f"[LazyAI] chat history, before tokens={total_tokens}, now tokens={tokens_cnt}")
        except Exception as e:
            logger.warning("Exception when counting tokens precisely for session: {}".format(str(e)))
        return session


class LazyAISession(ChatGPTSession):
    def calc_tokens(self):
        if not self.messages:
            return 0
        return len(str(self.messages))

    def discard_exceeding(self, max_tokens, cur_tokens=None):
        cur_tokens = self.calc_tokens()
        if cur_tokens > max_tokens:
            for i in range(0, len(self.messages)):
                if i > 0 and self.messages[i].get("role") == "assistant" and self.messages[i - 1].get("role") == "user":
                    self.messages.pop(i)
                    self.messages.pop(i - 1)
                    return self.calc_tokens()
        return cur_tokens
