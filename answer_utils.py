import jdatetime
from dateparser.calendars.jalali import JalaliCalendar


def get_slot(tokens, labels, slot_type):
    start_slot = "B-" + slot_type.upper()
    mid_slot = "I-" + slot_type.upper()
    slot = ''
    flag = False
    for i in range(len(tokens)):
        if flag and labels[i] != mid_slot:
            break

        if labels[i] == start_slot:
            slot = tokens[i]
            flag = True

        elif labels[i] == mid_slot:
            slot = slot + ' ' + tokens[i]

        else:
            if flag:
                return slot

    return slot


def get_date_from_string(date_string):
    date = JalaliCalendar(date_string).get_date().date_obj
    jalili_date = jdatetime.date.fromgregorian(day=date.day, month=date.month, year=date.year)
    return jalili_date


def weather_code_to_condition(weather_code):
    if 0 <= weather_code <= 3:
        condition = 'افتابی'

    elif 4 <= weather_code <= 19:
        condition = 'ابری'

    elif 20 <= weather_code <= 29:
        condition = 'بارانی'

    elif 30 <= weather_code <= 39:
        condition = 'برفی'

    elif 40 <= weather_code <= 49:
        condition = 'مه'

    else:
        condition = 'بارانی'

    return condition


def get_oghat_from_data(data, oghat):
    if not oghat:
        return None

    elif 'صبح' in oghat:
        return str(data['result']['azan_sobh'])

    elif 'طلوع' in oghat:
        return str(data['result']['tolu_aftab'])

    elif 'ظهر' in oghat:
        return str(data['result']['azan_zohr'])

    elif 'غروب' in oghat:
        return data['result']['ghorub_aftab']

    elif 'مغرب' in oghat:
        return str(data['result']['azan_maghreb'])

    elif 'شب' in oghat:
        return str(data['result']['nimeshab'])

    elif 'اوقات' in oghat or 'اذان' in oghat or 'اذون' in oghat:
        return '\n' +'اذان صبح: ' + str(data['result']['azan_sobh']) + '\n' + 'طلوع آفتاب: ' + str(data['result']['tolu_aftab']) + '\n' +'اذان ظهر: ' + str(data['result']['azan_zohr']) + '\n' +  'غروب آفتاب: ' + str(data['result']['ghorub_aftab']) + '\n' +   'اذان مغرب: ' + str(data['result']['azan_maghreb']) + '\n' +  'نیمه شب شرعی: ' + str(data['result']['nimeshab']) + '\n'

    else:
        return None


def get_ingredient_slots(tokens, labels):
    slot = ''
    flag = False
    slots = []
    start_slot = 'B-INGREDIENT'
    mid_slot = 'I-INGREDIENT'

    for i in range(len(tokens)):
        if labels[i] == start_slot:
            slot = tokens[i]
            flag = True

        elif labels[i] == mid_slot:
            slot = slot + ' ' + tokens[i]

        else:
            if flag:
                slots.append(slot)
                slot = ''

    if slot != '':
        slots.append(slot)

    return slots


def recipe_from_api_answer(data, turn):
    recipe_title = data[turn]['title']
    recipe_ingredients = ', '.join(data[turn]['ingredients'])
    recipe = data[turn]['recipe']

    return 'اسم غذا: ' + recipe_title + '\nمواد لازم: ' + recipe_ingredients + '\nدستورپخت: ' + recipe

def get_first_operand(tokens, labels):
    start_slot = "B-OPERAND"
    mid_slot = "I-OPERAND"
    slot = ''
    flag = False

    for i in range(len(tokens)):
        if flag and labels[i] != mid_slot:
            break

        elif labels[i] == start_slot:
            slot = tokens[i]
            flag = True

        elif flag and labels[i] == mid_slot:
            slot += tokens[i]


    return slot


def get_second_operand(tokens, labels):
    start_slot = "B-OPERAND"
    mid_slot = "I-OPERAND"
    slot = ''
    flag = False
    counter = 0

    for i in range(len(tokens)):
        if labels[i] == start_slot and counter == 1:
            slot = tokens[i]
            flag = True
            counter += 1

        elif labels[i] == start_slot and counter == 0:
            counter += 1

        elif labels[i] == mid_slot and counter == 2:
            slot = slot + ' ' + tokens[i]

        elif counter >= 2:
            if flag:
                return slot

    return slot


def calculate_string(op1, operator, op2):
    plus = ['+', 'جمع', 'علاوه', 'با', 'اضافه', 'بعلاوه']
    minus = ['-', 'تفریق', 'منها', 'کم']
    multi = ['*', 'x', 'ضرب', 'تا', 'در']
    div = ['/', 'تقسیم', 'بر']

    op1 = int(op1)
    op2 = int(op2)
    res = "ERROR"
    if operator in plus:
        res = op1 + op2
    elif operator in minus:
        res = op1 - op2
    elif operator in multi:
        res = op1 * op2
    elif operator in div:
        res = op1/op2

    return str(res)
