"""Step 10: the production runtime loop --

    RobotBackend.get_observation()
        -> Observation
        -> policy.predict(observation, instruction)
        -> RobotAction
        -> SafetySupervisor.process(...)
        -> safe RobotAction
        -> RobotBackend.execute_action(...)

with structured per-tick telemetry recorded alongside
(``telemetry.recorder.EpisodeTelemetryRecorder``). This is the canonical
direct-mode runtime path (README "Final system architecture") used by
both ``demo/run.py`` and any future benchmark/showcase tooling.
"""

from runtime.run_episode import run_episode

__all__ = ["run_episode"]
