#!/usr/bin/env python3
"""Emergency force-kill utility; never called by normal suite shutdown."""

from simple_adp.runner.process_cleanup import kill_owned_processes


if __name__ == "__main__":
    count = kill_owned_processes()
    print(f"force-killed {count} owned simple-adp process(es)")
