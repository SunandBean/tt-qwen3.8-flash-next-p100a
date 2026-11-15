"""Self-kill for card jobs: after P100A_WALL seconds (default 900) dump every thread's Python stack to stderr and
_exit(1) from faulthandler's own C thread, so a job stuck inside a device op still ends (and says where).
Import first thing in every card script. """
import faulthandler
import os
import sys

_wall = int(os.environ.get("P100A_WALL", "900"))
if _wall > 0:                                   # 0 = no self-exit (the interactive server)
    faulthandler.dump_traceback_later(_wall, exit=True, file=sys.stderr)
