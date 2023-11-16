import os
from importlib.machinery import SourceFileLoader
from pathlib import Path

from fgsim.config import conf

env = os.environ


def send_error(s):
    fn = Path("~/.bin/sendmsg.py").expanduser()
    if fn.exists():
        for k in ["HOSTNAME"]:
            s = f"{k}: {env[k]}\n" + s
        for k in ["tag", "command"][::-1]:
            s = f"{k}: {conf[k]}\n" + s
        SourceFileLoader("send_msg", str(fn)).load_module().send_message(s)


def send_exit():
    fn = Path("~/.bin/sendmsg.py").expanduser()
    if fn.exists():
        s = ""
        for k in ["HOSTNAME"]:
            s = f"{k}: {env[k]}\n" + s
        for k in ["tag", "command"][::-1]:
            s = f"{k}: {conf[k]}\n" + s
        s += "Succesfull exit"
        SourceFileLoader("send_msg", str(fn)).load_module().send_message(s)
