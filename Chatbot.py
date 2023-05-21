import numpy as np
import torch
import requests
from geopy.geocoders import Nominatim
import json
import random
# import translators as ts
from deep_translator import GoogleTranslator
from langdetect import detect
import finpy_tse as fpy
from jdatetime import timedelta
from hazm import *
from word2number import w2n
import difflib

from transformers import AutoConfig, AutoTokenizer, AutoModel, TFAutoModel
from transformers import BertTokenizer, BertModel

import os, sys

# PACKAGE_PARENT = '..'
# SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
# sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import project_statics
from utils import load_obj
from models.transformer.seqTagger import transformertagger
from models.transformer.BERT_Joint_IDSF import BertIDSF
from answer_utils import *
from post_processing import PostProcessing

class ChatBot(object):
    instance = None

    @classmethod
    def getBot(cls):
        if cls.instance is None:
            cls.instance = ChatBot()
        return cls.instance

    def __init__(self):
        print("Init")
        if self.instance is not None:
            raise ValueError("Did you forgot to call getBot function ? ")

        self.sentence = ""
        self.slots = {}
        self.domain = ""
        self.intent = ""
        self.tokens = []
        self.labels = []
        self.question_slot = ""
        self.turn = 0

        self.recipe_data = {}
        self.num_recipe = 0

        self.default_slots = {"DATE": "فردا", "LOC": "تهران", "CURR": "دلار", "FLOW": "صعودی", "TOPIC": "سیاسی"}

        f_slots = open('dataset/slots.json', encoding="utf-8")
        self.all_slots = json.load(f_slots)

        f_questions = open('dataset/incomplete_questions.json', encoding="utf-8")
        self.questions = json.load(f_questions)

        # Load Bert model
        model_name_or_path = "HooshvareLab/bert-fa-zwnj-base"
        config = AutoConfig.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModel.from_pretrained(model_name_or_path)
        model_bert = BertModel.from_pretrained("HooshvareLab/bert-fa-zwnj-base",
                                               output_hidden_states=True,
                                               )
        model_bert.eval()
        self.tokenizer = tokenizer
        self.bert_model = model_bert

        self.normalizer = Normalizer()

        # Load slot filling model
        save_path = 'test_IDSF_bert'
        data_path = project_statics.SFID_pickle_files
        Data = load_obj(data_path + '/Data')
        dict2 = load_obj(data_path + '/dict2')
        inte2 = load_obj(data_path + '/inte2')
        tagger_obj = transformertagger(save_path, BertIDSF, dict2, inte2, device=torch.device("cpu"))
        self.slot_model = tagger_obj

        # Request URLs
        self.weather_url = "https://api.open-meteo.com/v1/forecast?"
        self.oghat_url = "https://api.keybit.ir/owghat?"
        self.currency_url = "http://api.navasan.tech/latest/?"
        self.recipe_title_url = "http://127.0.0.1:5000/title?"
        self.recipe_ingredient_url = "http://127.0.0.1:5000/ingredients?"

        # Geolocator for weather conditions
        self.geoLocator = Nominatim(user_agent='myapplication')

        # Load templates answer
        f = open('dataset/Answers.json', encoding="utf-8")
        self.answers_template = json.load(f)

        # Define currency codes currency domain
        self.currencies = {'نیم': 'nim', 'ربع': 'rob', 'امامی': 'sekkeh', 'دلار': 'usd_buy', 'سکه': 'bahar'}

        # Define language codes for translation domain
        self.lang_codes = {'انگلیسی':'en', 'فارسی':'fa', 'ایرانی':'fa'}

    def clean_up_sentence(self, sentence):
        return sentence


    def classify_intent_slot(self, sentence):
        toks = sentence.split(' ')
        toks, predicted_labels, predicted_intents = self.slot_model.get_label([toks], need_tokenization=False)
        domain = predicted_intents[0].split('.')[1]
        intent = predicted_intents[0].split('.')[0]
        return domain, intent, toks[0], predicted_labels[0]


    def get_weather_conditions(self, tokens, labels):
        time = self.slots['DATE']
        date = get_date_from_string(time)
        date = jdatetime.date(date.year, date.month, date.day).togregorian()

        loc = self.slots['LOC']
        loc = GoogleTranslator(source='auto', target='en').translate(loc)

        location = self.geoLocator.geocode(loc)
        latitude = location.raw['lat']
        longtitude = location.raw['lon']

        URL = self.weather_url
        PARAMS = {'latitude': latitude, 'longitude': longtitude, 'start_date': date, 'end_date': date,
                  'current_weather': 'true'}

        r = requests.get(url=URL, params=PARAMS)
        data = r.json()
        weather_code = int(data['current_weather']['weathercode'])
        return weather_code_to_condition(weather_code)


    def get_oghat_times(self, tokens, labels):
        time = self.slots['DATE']
        date = get_date_from_string(time)
        print(date)

        month = str(date).split('-')[1]
        day = str(date).split('-')[2]
        print(month, day)

        loc = self.slots['LOC']

        oghat = self.slots['OGHAT']
        print(oghat)

        URL = self.oghat_url
        PARAMS = {'city': loc, 'month': month, 'day': day}

        r = requests.get(url=URL, params=PARAMS)
        data = r.json()
        print(data)
        oghat_time = get_oghat_from_data(data, oghat)
        if oghat_time:
            return oghat_time

        return "زمان کدام وقت شرعی را میخواهید بدانید"


    def get_currency_value(self, tokens, labels):
        currency = self.slots['CURR']
        item = 'usd_buy'

        for curr in self.currencies.keys():
            if currency in curr or curr in currency:
                item = self.currencies[curr]
                break

        URL = self.currency_url
        PARAMS = {'api_key': 'freeochEmsIcYrfhv9In7r1wEoAmGx6t', 'item': item}

        r = requests.get(url=URL, params=PARAMS)
        data = r.json()
        print(item)
        print(data)
        value = data[item]['value']
        change = data[item]['change']

        return str(value), str(change)

    def math_expression_result(self, toks, labels):
        operand1 = get_first_operand(toks, labels)
        operator = get_slot(toks, labels, 'OPERATOR')
        operand2 = get_second_operand(toks, labels)

        operand1 = w2n.word_to_num(GoogleTranslator(source='auto', target='en').translate(operand1))
        operand2 = w2n.word_to_num(GoogleTranslator(source='auto', target='en').translate(operand2))

        return calculate_string(operand1, operator, operand2)


    def translate_text(self, toks, labels):
        text = self.slots['TRANS']
        src_lang = detect(text)
        if src_lang == 'fa':
            to_lang = 'en'
        else:
            to_lang = 'fa'

        translation = translated = GoogleTranslator(source='auto', target=to_lang).translate(text)
        print(translation)
        return translation


    def get_stock_price(self, tokens, labels):
        stock = get_slot(tokens, labels, 'COMP')
        if stock == '':
            return ''

        data = fpy.Get_Price_History(
            stock=stock,
            start_date=str(jdatetime.date.today() - timedelta(days=100)),
            end_date=str(jdatetime.date.today()),
            ignore_date=False,
            adjust_price=False,
            show_weekday=False,
            double_date=False).iloc[-1]

        price = data['Final']
        company = data['Name']
        return str(price)


    def get_recipe_from_title(self, tokens, labels):
        title = get_slot(tokens, labels, 'FOOD-TITLE')

        URL = self.recipe_title_url
        PARAMS = {'title': title}

        r = requests.get(url = URL, params = PARAMS)
        data = r.json()
        print(len(data))
        self.recipe_data = data
        return recipe_from_api_answer(data, 0)


    def get_recipe_from_ingredients(self, tokens, labels):
        ingredients = get_ingredient_slots(tokens, labels)
        ingredients = ', '.join(ingredients)

        URL = self.recipe_ingredient_url
        PARAMS = {'ingredients': ingredients}

        r = requests.get(url=URL, params=PARAMS)
        data = r.json()
        print(data)
        self.recipe_data = data
        return recipe_from_api_answer(data, 0)

    def fill_slots(self, domain, intent, tokens, labels):
        needed_slots = self.all_slots[domain][intent]
        for needed_slot in needed_slots:
            if needed_slot == 'OPERAND1':
                slot = get_first_operand(tokens, labels)
            elif needed_slot == 'OPERAND2':
                slot = get_second_operand(tokens, labels)
            else:
                slot = get_slot(tokens, labels, needed_slot)

            if not slot:
                self.slots[needed_slot] = ""
            else:
                self.slots[needed_slot] = slot

        print(self.slots)


    def get_empty_slot(self):
        for slot in self.slots.keys():
            if self.slots[slot] == "":
                return slot
        return None

    def reset_slots(self):
        self.sentence = ""
        self.slots = {}
        self.domain = ""
        self.intent = ""
        self.tokens = []
        self.labels = []
        self.question_slot = ""
        self.turn = 0


    def incomplete_question(self, incomplete_slot):
        return "لطفا اطلاعات " + incomplete_slot + " را وارد کنید"


    def generate_complete_answer(self, answer, domain, intent, tokens, labels, slots, additional=""):
        temps = self.answers_template[domain][intent]
        if intent == "تایید" or intent == "مشورت":
            if domain == "آب و هوا":
                condition = self.normalizer.normalize(slots['COND'])
                answer = self.normalizer.normalize(answer)

                print(answer, condition)
                seq = difflib.SequenceMatcher(a=answer, b=condition)

                if seq.ratio() > 0.8:
                    temps = temps["بله"]
                else:
                    temps = temps["خیر"]

            elif domain == "ارز و سکه":
                flow = get_slot(tokens, labels, 'FLOW')
                if 'صعود' in flow or 'بالا' in flow or 'زیاد' in flow:
                    flow = "+"
                elif 'نزول' in flow or 'پایین' in flow or 'کم' in flow:
                    flow = "-"
                else:
                    flow = "0"

                correct_flow = "0"
                if int(answer) > 0:
                    answer = "صعودی"
                    correct_flow = "+"
                elif int(answer) < 0:
                    correct_flow = "-"
                    answer = "نزولی"
                else:
                    answer = "ثابت"

                if correct_flow == flow:
                    temps = temps["بله"]
                else:
                    temps = temps["خیر"]

        temp = random.choice(temps)

        if "ANSWER" in temp:
            temp = temp.replace('ANSWER', answer)

        # empty_slots = []
        # for word in temp.split(' '):
        #     if word.isupper():
        #         empty_slots.append(word)
        #
        # for empty_slot in empty_slots:
        #     slot = self.slots[empty_slot.replace('B-', '')]
        #     print(slot, empty_slot)
        #     if not slot:
        #         if empty_slot == 'B-TIME':
        #             slot = 'امروز'
        #         elif empty_slot == 'B-LOC':
        #             slot = 'تهران'
        #     temp = temp.replace(empty_slot, slot)
        # if additional != "":
        #     return additional + "\n" + temp
        return temp


    def response(self, sentence):
        additional = ""
        answer = ""
        print(self.turn)
        if len(self.recipe_data) != 0:
            if ("بله" in sentence or "آره" in sentence or "یس" in sentence or "اره" in sentence) and self.num_recipe < 3:
                answer = recipe_from_api_answer(self.recipe_data, self.num_recipe)
                self.num_recipe += 1

                if self.num_recipe < 3:
                    return "INCOMPLETE", self.domain, self.intent, self.tokens, self.labels, self.slots, self.generate_complete_answer(
                    answer, self.domain, self.intent, self.tokens, self.labels, self.slots, additional) + "\n" + "آیا مایلید دستورپخت های دیگری را نیز ببینید"

                else:
                    domain = self.domain
                    intent = self.intent
                    tokens = self.tokens,
                    labels = self.labels
                    slots = self.slots
                    self.recipe_data = {}
                    self.num_recipe = 0
                    self.reset_slots()
                    return "SUCCESS", domain, intent, tokens, labels, slots, self.generate_complete_answer(
                        answer, domain, intent, tokens, labels, slots)

            else:
                answer = " خواهشا یک سوال دیگر بپرسید"
                domain = self.domain
                intent = self.intent
                tokens = self.tokens,
                labels = self.labels
                slots = self.slots
                self.recipe_data = {}
                self.num_recipe = 0
                self.reset_slots()

                return "SUCCESS", domain, intent, tokens, labels, slots, answer


        if self.turn == 2:
            self.turn += 1
            if self.question_slot in self.default_slots.keys():
                self.sentence = self.sentence + self.default_slots[self.question_slot]
                self.tokens = self.tokens + [self.default_slots[self.question_slot]]
                self.labels = self.labels + ["B-" + self.question_slot]
                self.slots[self.question_slot] = self.default_slots[self.question_slot]
                additional = self.questions[self.question_slot][str(self.turn)][0]
                self.question_slot = self.get_empty_slot()
                if self.question_slot:
                    self.turn = 0

            else:
                answer = self.questions[self.question_slot][str(self.turn)][0]
                domain = self.domain
                intent = self.intent
                tokens = self.tokens,
                labels = self.labels
                slots = self.slots
                self.reset_slots()
                if additional != "":
                    answer = additional + "\n" + answer
                return "INCOMPLETE", domain, intent, tokens, labels, slots, answer

        if self.sentence == "":
            self.turn += 1
            domain, intent, toks, labels  = self.classify_intent_slot(sentence)
            self.sentence = sentence
            self.domain = domain
            self.intent = intent
            self.tokens = toks
            self.labels = labels
            self.fill_slots(domain, intent, toks, labels)

        else:
            domain, intent, toks, labels = self.classify_intent_slot(self.sentence + " " + sentence)
            self.question_slot = self.get_empty_slot()
            print(self.question_slot)
            if self.question_slot:
                if not get_slot(toks, labels, self.question_slot):
                    self.turn += 1
                    answer = self.questions[self.question_slot][str(self.turn)][0]
                    if additional != "":
                        answer = additional + "\n" + answer
                    return "INCOMPLETE", self.domain, self.intent, self.tokens, self.labels, self.slots, answer

                else:
                    self.turn = 1
                    self.sentence = self.sentence + sentence
                    self.tokens = self.tokens + sentence.split(" ")
                    self.labels = self.labels + ["B-" + self.question_slot]
                    for i in range(len(sentence.split(" "))-1):
                        self.labels = self.labels + ["I-" + self.question_slot]
                    self.slots[self.question_slot] = sentence

        self.domain, self.intent, self.slots = PostProcessing.fix_weather(self.domain, self.intent, self.slots)
        empty_slot = self.get_empty_slot()
        if empty_slot:
            if empty_slot != 'DEST-LANG':
                self.question_slot = empty_slot
                answer = self.questions[self.question_slot][str(self.turn)][0]
                if additional != "":
                    answer = additional + "\n" + answer
                return "INCOMPLETE", self.domain, self.intent, self.tokens, self.labels, self.slots, answer

        if domain == "آب و هوا":
            answer = self.get_weather_conditions(toks, labels)

        elif domain == "اوقات شرعی":
            answer = self.get_oghat_times(toks, labels)

        elif domain == "ارز و سکه":
            value, change = self.get_currency_value(toks, labels)
            if intent == "پرسش":
                answer = value
            elif intent == "تایید":
                answer = change

        elif domain == "ترجمه عبارت":
            answer = self.translate_text(toks, labels)

        elif domain == "ماشین حساب":
            answer = self.math_expression_result(toks, labels)

        elif domain == "اخبار روز":
            answer = ""

        elif domain == "اخبار بورس":
            answer = self.get_stock_price(toks, labels)

        elif domain == "دستورپخت":
            self.num_recipe = 1
            if intent == "دستور":
                slot = get_slot(toks, labels, 'FOOD-TITLE')
                if not slot:
                    return "INCOMPLETE", self.incomplete_question('عنوان غذا')
                answer = self.get_recipe_from_title(toks, labels)

            elif intent == "پیشنهاد":
                print(toks, labels)
                slot = get_slot(toks, labels, 'INGREDIENT')
                if not slot:
                    return "INCOMPLETE", self.incomplete_question('مواد غذایی')

                answer = self.get_recipe_from_ingredients(toks, labels)

            return "SUCCESS", domain, intent, toks, labels, self.slots, self.generate_complete_answer(answer, domain,
                                                                                                       intent, toks,
                                                                                                       labels, self.slots) + "\n" + "آیا مایلید دستورپخت های دیگری را نیز ببینید"

        elif domain == "احوالپرسی":
            answer = ""

        domain = self.domain
        intent = self.intent
        tokens = self.tokens,
        labels = self.labels
        slots = self.slots
        self.reset_slots()

        if additional != '':
            return "SUCCESS", domain, intent, tokens, labels, slots, additional + "\n" +self.generate_complete_answer(answer, domain,
                                                                                                   intent, toks, labels,
                                                                                                   slots, additional)

        return "SUCCESS", domain, intent, tokens, labels, slots, self.generate_complete_answer(answer, domain, intent, toks, labels, slots, additional)
