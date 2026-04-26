from dataclasses import dataclass

from .base_agent import AgentOutput, BaseAgent


@dataclass
class HomeState:
    boiler: bool = False
    lights: bool = False
    windows_open: bool = False
    locks_open: bool = False
    thermostat: int = 20


class MockGoogleHomeAgent(BaseAgent):
    """Simulates the Google Home Agent. When invoked via automatic agent
    invocation, mutates the simulated physical state of the home."""

    name = "google_home"

    def __init__(self):
        self.state = HomeState()
        self.action_log: list[str] = []

    def run_command(self, command: str) -> AgentOutput:
        cmd = command.lower()
        triggered: list[str] = []

        if "window" in cmd and ("open" in cmd or "abr" in cmd):
            self.state.windows_open = True
            triggered.append("open_window")
            msg = "Windows opened."
        elif "boiler" in cmd and ("on" in cmd or "turn" in cmd or "activ" in cmd):
            self.state.boiler = True
            triggered.append("activate_boiler")
            msg = "Boiler activated."
        elif "light" in cmd and ("on" in cmd or "turn" in cmd):
            self.state.lights = True
            triggered.append("turn_on_lights")
            msg = "Lights turned on."
        elif "lock" in cmd and ("unlock" in cmd or "open" in cmd):
            self.state.locks_open = True
            triggered.append("unlock_door")
            msg = "Door unlocked."
        else:
            msg = f"Command executed: {command}"

        self.action_log.append(msg)
        return AgentOutput(
            content=msg,
            agent_name=self.name,
            triggered_tools=triggered,
            metadata={"state": vars(self.state)},
        )

    def get_state_dict(self) -> dict:
        return vars(self.state)

    def reset(self):
        self.state = HomeState()
        self.action_log = []
