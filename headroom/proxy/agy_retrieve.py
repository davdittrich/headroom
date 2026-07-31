"""In-process hypercorn PLAIN-HTTP retrieve server for agy.

The proxy compresses tool_result payloads and emits ``[Retrieve more:
hash=…]`` markers.  For agy those markers are produced on the decrypted
stream inside the HTTPS dispatch server (:mod:`headroom.proxy.agy_dispatch`).
To resolve a marker the agent runs the ``headroom mcp serve`` stdio child,
which calls the proxy's retrieve HTTP endpoint via ``HEADROOM_PROXY_URL``.

The dispatch server is HTTPS with a Cloud-Code-SNI leaf only, so a stdio
retrieve child cannot reach it over loopback.  This module stands up a
SECOND loopback listener — PLAIN HTTP, no TLS — serving the same FastAPI
app on an ephemeral port for the session.  The compression/marker cache is
a process-global singleton (:func:`headroom.cache.compression_store.get_compression_store`),
so this second ``create_app()`` shares the exact cache the dispatch server
populates: a marker minted on the HTTPS side resolves over plain HTTP here.

Why plain HTTP is safe: the listener binds ``127.0.0.1`` only, serves the
retrieve endpoints to a stdio child in the *same* trust boundary, and never
carries upstream credentials (it only reads the in-memory marker cache).

The hypercorn plumbing (lifespan, TCPServer, socket options, lifecycle) is
:class:`headroom.proxy.agy_dispatch.AgyDispatchServer`'s — this listener is
that same server in its ``plain_http`` mode: no SSL context, no CA touched,
no Host allowlist guard.
"""

from __future__ import annotations

from headroom.proxy.agy_dispatch import AgyDispatchServer


class AgyRetrieveServer(AgyDispatchServer):
    """PLAIN-HTTP loopback listener serving the headroom FastAPI app.

    Serves the process-global compression cache via ``create_app()`` so
    ``GET /v1/retrieve/{hash}`` resolves markers the HTTPS dispatch server
    populated.

    Usage::

        server = AgyRetrieveServer()
        await server.start()
        # server.address → ("127.0.0.1", <ephemeral-port>)
        await server.stop()

    Or as an async context manager::

        async with AgyRetrieveServer() as srv:
            host, port = srv.address
    """

    def __init__(self, port: int = 0) -> None:
        super().__init__(port=port, plain_http=True)
