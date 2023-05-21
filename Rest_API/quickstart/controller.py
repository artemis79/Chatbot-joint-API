from django.http import JsonResponse
from django.template import loader
import json
# from Bot import ChatBot as bot
from time import gmtime, strftime

import os,sys

# PACKAGE_PARENT = '..\..\..'
# SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
# sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from django.views.decorators.csrf import csrf_exempt

from Chatbot import ChatBot as bot


@csrf_exempt
def index(request):
    if request.method == 'POST':
        jsonData = json.loads(request.body.decode('utf-8'))
        msg = jsonData["msg"]

        code, domain, intent, tokens, labels, slots, res = bot.getBot().response(msg)
        time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        return JsonResponse({
            "desc": code,
            "domain": domain,
            "intent": intent,
            "tokens": tokens,
            "labels": labels,
            "slots": slots,
            "ques": msg,
            "res": res,
            "time": time
        })
    else:
        return JsonResponse({"desc": "Bad request"}, status=400)
