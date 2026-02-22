import base64
import json
import urllib.parse
import urllib.request
from time import perf_counter
from typing import Any


class DidaService:
    def __init__(self, *, client_id: str, client_secret: str, redirect_uri: str) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.base_url = "https://api.dida365.com/open/v1"
        self.oauth_url = "https://dida365.com/oauth/authorize"
        self.token_url = "https://dida365.com/oauth/token"

    def get_oauth_url(self, *, state: str) -> str:
        params = {
            "scope": "tasks:write tasks:read",
            "client_id": self.client_id,
            "state": state,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
        }
        return f"{self.oauth_url}?{urllib.parse.urlencode(params)}"

    def exchange_code(self, *, code: str) -> dict[str, Any]:
        auth = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode("utf-8")).decode("utf-8")
        headers = {
            "Authorization": f"Basic {auth}",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = urllib.parse.urlencode(
            {
                "code": code,
                "grant_type": "authorization_code",
                "scope": "tasks:write tasks:read",
                "redirect_uri": self.redirect_uri,
            }
        ).encode("utf-8")
        return self._request_raw("POST", self.token_url, headers=headers, data=data)

    def get_projects(self, *, access_token: str) -> list[dict[str, Any]]:
        return self._request_openapi("GET", "/project", access_token=access_token)

    def get_project_data(self, *, access_token: str, project_id: str) -> dict[str, Any]:
        return self._request_openapi("GET", f"/project/{project_id}/data", access_token=access_token)

    def create_task(self, *, access_token: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request_openapi("POST", "/task", access_token=access_token, data=payload)

    def update_task(self, *, access_token: str, task_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request_openapi("POST", f"/task/{task_id}", access_token=access_token, data=payload)

    def delete_task(self, *, access_token: str, project_id: str, task_id: str) -> dict[str, Any]:
        return self._request_openapi(
            "DELETE",
            f"/project/{project_id}/task/{task_id}",
            access_token=access_token,
            headers={"Content-Type": "application/json"},
        )

    def complete_task(self, *, access_token: str, project_id: str, task_id: str) -> dict[str, Any]:
        return self._request_openapi(
            "POST",
            f"/project/{project_id}/task/{task_id}/complete",
            access_token=access_token,
        )

    def _request_openapi(
        self,
        method: str,
        path: str,
        *,
        access_token: str,
        data: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        final_headers = {"Authorization": f"Bearer {access_token}"}
        if headers:
            final_headers.update(headers)
        return self._request_json(method, url, headers=final_headers, data=data)

    def _request_json(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = None
        final_headers = dict(headers or {})
        if data is not None:
            body = json.dumps(data).encode("utf-8")
            final_headers["Content-Type"] = "application/json"
        return self._request_raw(method, url, headers=final_headers, data=body)

    def _request_raw(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        data: bytes | None = None,
    ) -> dict[str, Any]:
        request = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
        started = perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                status = response.getcode()
                text = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            text = error.read().decode("utf-8") if hasattr(error, "read") else ""
            elapsed_ms = (perf_counter() - started) * 1000
            print(f"[DIDA] {method} {url} status={error.code} elapsed_ms={elapsed_ms:.2f} error=http")
            raise RuntimeError(f"Dida API error {error.code}: {text}") from error
        except urllib.error.URLError as error:
            elapsed_ms = (perf_counter() - started) * 1000
            print(f"[DIDA] {method} {url} elapsed_ms={elapsed_ms:.2f} error=network detail={error}")
            raise RuntimeError(f"Dida API request failed: {error}") from error
        elapsed_ms = (perf_counter() - started) * 1000
        print(f"[DIDA] {method} {url} status={status} elapsed_ms={elapsed_ms:.2f} bytes={len(text)}")
        if status not in (200, 201):
            raise RuntimeError(f"Dida API error {status}: {text}")
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw": text}
