from urllib.request import Request, urlopen

from .base_agent import AgentOutput, BaseAgent


class MockUtilitiesAgent(BaseAgent):
    """Simulates the Utilities Agent that can open URLs and applications.
    When live_requests is True, performs a real HTTP GET to the local
    request_catcher to simulate exfiltration."""

    name = "utilities"

    def __init__(
        self,
        catcher_url: str = "http://localhost:5001",
        live_requests: bool = False,
    ):
        self.catcher_url = catcher_url
        self.live_requests = live_requests
        self.opened_urls: list[str] = []
        self.opened_apps: list[dict] = []

    def open_url(self, url: str) -> AgentOutput:
        self.opened_urls.append(url)

        if self.live_requests and url.startswith("http"):
            try:
                urlopen(Request(url), timeout=3).read()
            except Exception:
                pass

        return AgentOutput(
            content=f"Opening URL: {url}",
            agent_name=self.name,
            triggered_tools=["open_url"],
            metadata={"url": url},
        )

    def open_app(self, app_name: str, params: dict | None = None) -> AgentOutput:
        entry = {"app": app_name, "params": params or {}}
        self.opened_apps.append(entry)
        return AgentOutput(
            content=f"Opening app: {app_name}",
            agent_name=self.name,
            triggered_tools=["open_app"],
            metadata=entry,
        )

    def get_exfiltrated_data(self) -> list[str]:
        return self.opened_urls

    def reset(self):
        self.opened_urls = []
        self.opened_apps = []
