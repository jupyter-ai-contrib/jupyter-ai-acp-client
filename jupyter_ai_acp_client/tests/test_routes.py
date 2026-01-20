import json


async def test_hello(jp_fetch):
    # When
    response = await jp_fetch("jupyter-ai-acp-client", "hello")

    # Then
    assert response.code == 200
    payload = json.loads(response.body)
    assert payload == {
            "data": (
                "Hello, world!"
                " This is the '/jupyter-ai-acp-client/hello' endpoint."
                " Try visiting me in your browser!"
            ),
        }
