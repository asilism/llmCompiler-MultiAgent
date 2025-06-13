from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI


################################################################################
# TODO: for debug
################################################################################
_DEBUG = False
_DEBUG_STREAM_OUTPUT = False


def _debug_inputs(func_name, config, input_, messages):
    print(f'\n@@@@@@@@@@@@@@@@@@@@ BaseChatModel.{func_name}() @@@@@@@@@@@@@@@@@@@@')
    print(f'@@ config: {config}')
    for n, msg in enumerate(messages, 1):
        print(f'@@ ==================== input[{n}/{len(messages)}]-{msg.__class__.__name__} ========================================\n'
              f'{msg.content}\n'
              f'@@ --------------------------------------------------------------------------------\n'
              f'@@ additional_kwargs={msg.additional_kwargs}\n'
              f'@@ response_metadata={msg.response_metadata}\n'
              f'@@ ================================================================================')


def _debug(func_name, key, value):
    print(f'@@ BaseChatModel.{func_name}(): {key}={value}')


class DebugChatOpenAI(ChatOpenAI):
    def invoke(self, input_, config=None, *args, **kwargs):
        _debug_inputs("invoke", config, input_, self._convert_input(input_).to_messages())
        output = super().invoke(input_, config, *args, **kwargs)
        _debug("invoke", "output", output)
        return output

    async def ainvoke(self, input_, config=None, *args, **kwargs):
        _debug_inputs("ainvoke", config, input_, self._convert_input(input_).to_messages())
        output = super().invoke(input_, config, *args, **kwargs)
        _debug("ainvoke", "output", output)
        return output

    def stream(self, input_, config=None, *args, **kwargs):
        _debug_inputs("stream", config, input_, self._convert_input(input_).to_messages())
        _debug_index = 0
        for output in super().stream(input_, config, *args, **kwargs):
            if _DEBUG_STREAM_OUTPUT:
                _debug("stream", f'output[{_debug_index}]', output)
                _debug_index += 1
            yield output

    async def astream(self, input_, config=None, *args, **kwargs):
        _debug_inputs("astream", config, input_, self._convert_input(input_).to_messages())
        _debug_index = 0
        async for output in super().astream(input_, config, *args, **kwargs):
            if _DEBUG_STREAM_OUTPUT:
                _debug("astream", f'output[{_debug_index}]', output)
                _debug_index += 1
            yield output


class DebugChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    def invoke(self, input_, config=None, *args, **kwargs):
        _debug_inputs("invoke", config, input_, self._convert_input(input_).to_messages())
        output = super().invoke(input_, config, *args, **kwargs)
        _debug("invoke", "output", output)
        return output

    async def ainvoke(self, input_, config=None, *args, **kwargs):
        _debug_inputs("ainvoke", config, input_, self._convert_input(input_).to_messages())
        output = super().invoke(input_, config, *args, **kwargs)
        _debug("ainvoke", "output", output)
        return output

    def stream(self, input_, config=None, *args, **kwargs):
        _debug_inputs("stream", config, input_, self._convert_input(input_).to_messages())
        _debug_index = 0
        for output in super().stream(input_, config, *args, **kwargs):
            if _DEBUG_STREAM_OUTPUT:
                _debug("stream", f'output[{_debug_index}]', output)
                _debug_index += 1
            yield output

    async def astream(self, input_, config=None, *args, **kwargs):
        _debug_inputs("astream", config, input_, self._convert_input(input_).to_messages())
        _debug_index = 0
        async for output in super().astream(input_, config, *args, **kwargs):
            if _DEBUG_STREAM_OUTPUT:
                _debug("astream", f'output[{_debug_index}]', output)
                _debug_index += 1
            yield output


if _DEBUG:
    ChatOpenAI = DebugChatOpenAI
    ChatGoogleGenerativeAI = DebugChatGoogleGenerativeAI
################################################################################


_CLIENTS = {
    "OPENAI": ChatOpenAI,
    "GOOGLE": ChatGoogleGenerativeAI,
}


def get_llm_client(llm_type: str, config: dict, **kwargs):
    for required in ("api_key", "model", "base_url"):
        assert required in config
    config.update(kwargs)
    return _CLIENTS[llm_type.upper()](**config)
