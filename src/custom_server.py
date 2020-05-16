import asyncio
import logging
import ssl
import websockets
import binascii
import syft as sy
from syft.workers.websocket_server import WebsocketServerWorker

class CustomWebsocketServerWorker(WebsocketServerWorker):
    def start(self):
        """Start the server"""
        # Secure behavior: adds a secure layer applying cryptography and authentication
        if not (self.cert_path is None) and not (self.key_path is None):
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_context.load_cert_chain(self.cert_path, self.key_path)
            start_server = websockets.serve(
                self._handler,
                self.host,
                self.port,
                ssl=ssl_context,
                max_size=None,
                ping_interval=None,
                ping_timeout=None,
                close_timeout=None,
            )
        else:
            # Insecure
            start_server = websockets.serve(
                self._handler,
                self.host,
                self.port,
                max_size=None,
                ping_interval=None,
                ping_timeout=None,
                close_timeout=None,
            )

        asyncio.get_event_loop().run_until_complete(start_server)
        # print("Serving. Press CTRL-C to stop.")
        # try:
        asyncio.get_event_loop().run_forever()
        # except KeyboardInterrupt:
            # logging.info("Websocket server stopped.")
