"""The risk network (plan 2026-11-16 Task 15): a network in place of the danger tables.

``data`` reads the rows the Rust ``risk_log`` example writes, ``model`` is the network
and its MJNN export, ``fit`` trains it, ``check`` measures it against the tables, exact
counting and the belief model on held-out deals.
"""
