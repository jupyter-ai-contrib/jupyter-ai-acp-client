# import json

from tornado.httpclient import HTTPClientError

async def test_slash_commands_route_no_chat(jp_fetch):
    """
    Expects that the /ai/acp/slash_commands route returns a 400 when no ?chat_id
    URL query argument is given.
    """
    try:
        await jp_fetch("ai", "acp", "slash_commands")
    except HTTPClientError as e:
        assert e.code == 400
