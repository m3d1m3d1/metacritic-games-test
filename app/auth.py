import base64
import binascii
import hmac

from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from app.config import settings


class DemoBasicAuthMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(
        self, scope: Scope, receive: Receive, send: Send
    ) -> None:
        if scope["type"] != "http" or scope.get("path") == "/health":
            await self.app(scope, receive, send)
            return

        username = settings.demo_auth_username
        password = settings.demo_auth_password
        if username is None or password is None:
            await self.app(scope, receive, send)
            return

        authorization = next(
            (
                value
                for key, value in scope.get("headers", [])
                if key.lower() == b"authorization"
            ),
            b"",
        )
        try:
            scheme, encoded = authorization.split(None, 1)
            if scheme.lower() != b"basic":
                raise ValueError("Unsupported authorization scheme")
            decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
            supplied_username, supplied_password = decoded.split(":", 1)
        except (ValueError, UnicodeDecodeError, binascii.Error):
            supplied_username, supplied_password = "", ""

        username_matches = hmac.compare_digest(
            supplied_username.encode("utf-8"), username.encode("utf-8")
        )
        password_matches = hmac.compare_digest(
            supplied_password.encode("utf-8"),
            password.get_secret_value().encode("utf-8"),
        )
        if not (username_matches & password_matches):
            response = PlainTextResponse(
                "Authentication required",
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="Metacritic Games"'},
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
