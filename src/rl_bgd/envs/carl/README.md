# CARL adapter

RL-BGD targets CARL 1.1.1 for the first controlled-nonstationarity adapter.

CARL 1.1.1 currently constrains Gymnasium to versions below 1.0, whereas the
general RL development environment may use newer Gymnasium releases. Keep CARL
as an optional benchmark environment rather than importing it from the base
package.

The adapter removes CARL's context observation and context ID in strict
task-agnostic mode. The context schedule is generated inside the environment
wrapper from global environment steps; no task switch callback reaches the
agent.

CARL 1.1.1 re-selects its active context through `context_selector` during
`reset()`. The adapter therefore synchronizes the scheduled context into the
selector's active slot and applies it before reset, so both reset-time simulator
state and subsequent steps use the same hidden context.

Supported schedule primitives are abrupt, smooth interpolation, periodic,
random-walk, and recurring trajectories. Multidimensional contexts are
supported because schedules operate on context dictionaries.
