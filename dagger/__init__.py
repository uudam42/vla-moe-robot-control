"""Step 8: DAgger / corrective on-policy data collection for the Temporal
Dense VLA (README "Step 8").

Nothing in this package is on the final policy's runtime path -- the
scripted/corrective expert is teacher supervision used only to LABEL states
visited by the learned policy during offline data collection (see
``dagger/corrective_expert.py`` and ``dagger/collector.py``), never to
choose which action is actually executed in the simulator.
"""
