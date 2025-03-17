import io
import sys
from functools import wraps
from traceback import print_exception

import psutil
from slack import WebClient
from slack.errors import SlackApiError
from termcolor import cprint


def print_warn(text, *args, **kwargs):
    cprint(text, "yellow", *args, **kwargs)


def print_error(text, *args, **kwargs):
    cprint(text, "red", *args, **kwargs)


def print_success(text, *args, **kwargs):
    cprint(text, "green", *args, **kwargs)


def print_code(text, *args, **kwargs):
    cprint(text, "cyan", *args, **kwargs)


def print_now(text, *args, **kwargs):
    print(text, *args, **kwargs)
    sys.stdout.flush()


def send_slack_msg(msg="", success=True):
    token = "token here"
    client = WebClient(token=token)
    try:
        proc_name = " ".join(psutil.Process().cmdline()[1:])
        if success:
            msg = f"Your job {proc_name} succeeded! :tada:\n```{msg}```"
        else:
            msg = f"Your job {proc_name} failed! :collision:\n```{msg}```"
        client.chat_postMessage(channel="remote-job-notifs", text=msg)
        print("Slack message sent")
    except SlackApiError as err:
        errmsg = err.response["error"]
        print(f"Unable to send Slack message. {errmsg}")


def notify(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        output = io.StringIO()
        try:
            result = func(*args, **kwargs)
            send_slack_msg(result)
        except:
            exp_type, exp_val, tb = sys.exc_info()
            print_exception(exp_type, exp_val, tb, limit=1, file=output)
            send_slack_msg(output.getvalue(), success=False)
            raise

    return wrapper
