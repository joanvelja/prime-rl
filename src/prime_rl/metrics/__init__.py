"""Step-level metric aggregators for training observability.

Pure functions over ``list[vf.RolloutOutput]``; no I/O, no orchestrator
state. Called once per training step from the orchestrator's save path.
"""
